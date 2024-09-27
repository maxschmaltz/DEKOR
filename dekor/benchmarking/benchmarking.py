import os
import json
from itertools import product
from sklearn.model_selection import train_test_split
from io import BytesIO
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from typing import Optional, List, Dict, Iterable

import dekor.splitters
from dekor.splitters.base import BaseSplitter
from dekor.utils.gecodb_parser import parse_gecodb, Compound, Tuple, Union
from dekor.eval.evaluator import CompoundEvaluator, EvaluationResult

evaluator = CompoundEvaluator()


def predict_with_splitter(
    splitter: BaseSplitter,
    test_compounds: Iterable[Compound]
) -> Tuple[List[Compound], EvaluationResult]:
    
    """
    Predict lemmas with a splitter.

    Parameters
    ----------
    splitter : `BaseSplitter`
        fit splitter

    test_compounds : `Iterable[Compound]`
        compounds to predict; note that it is not an `Iterable[str]`
        as in the splitters because this method is used for evaluation,
        not inference

    Returns:
    `List[Compound]`
        preds
    `EvaluationResult`
        the result of evaluation
    """

    test_lemmas = [
        compound.lemma for compound in test_compounds
    ]
    pred_compounds = splitter.predict(test_lemmas)
    scores = evaluator.evaluate(test_compounds, pred_compounds)
    return pred_compounds, scores


def eval_splitter(
    *,
    splitter: BaseSplitter,
    test_compounds: List[Compound]
) -> Tuple[EvaluationResult, List[Compound]]:
    
    """
    Evaluate splitter on test compounds.

    Parameters
    ----------
    splitter : `BaseSplitter`
        fit splitter

    test_compounds : `Iterable[Compound]`
        compounds to predict; note that it is not an `Iterable[str]`
        as in the splitters because this method is used for evaluation,
        not inference

    Returns:
    `EvaluationResult`
        the result of evaluation
    `List[Compound]`
        preds
    """

    pred_compounds, scores = predict_with_splitter(splitter, test_compounds)
    return scores, pred_compounds


def benchmark_splitter(
    *,
    splitter_cls: type,
    min_counts: List[int],
    all_splitter_params: List[dict],
    gecodb_path: str,
    train_size: Optional[float]=0.8,
    verbose: Optional[bool]=True
) -> Tuple[List[dict], Tuple[List[Compound], List[Compound]], Union[BytesIO, None]]:
    
    """
    Benchmark a single splitter given different configurations.

    Parameters
    ----------
    splitter_cls : `type`
        splitter class

    min_counts : `List[int]`:
        minimal counts of compounds to keep;
        all compounds occurring less will be dropped at the respective iteration

    all_splitter_params : `List[dict]`
        list of tested configurations for the splitter,
        with each configuration being a mapping between parameter name and its value

    gecodb_path : `str`
        path to the TSV DECOW16-format compounds dataset

    train_size : `float`, optional, defaults to `0.8`
        train size

    verbose : `bool` optional, defaults to `True`
        whether to show progress bar when fitting and predicting compounds
    """

    outputs = []
    best_score = -1 # so that even 0 is instantly better
    best_pairs = None
    best_plot_buffer = None
    n_iter = 1  # for printout
    for min_count in min_counts:
        gecodb = parse_gecodb(
            gecodb_path,
            min_count=min_count
        )
        train_data, test_data = train_test_split(gecodb, train_size=train_size, shuffle=True)
        train_compounds = train_data["compound"].values
        test_compounds = test_data["compound"].values
        for splitter_params in all_splitter_params:
            if verbose: # print out params
                print(
                    '\n',
                    f"{splitter_cls.name}: iter {n_iter}/{len(all_splitter_params * len(min_counts))}"
                )
                print(
                    ', '.join(f"{key}: {value}" for key, value in splitter_params.items()),
                    f", min_count: {min_count}"
                )
            splitter = splitter_cls(
                **splitter_params,
                verbose=verbose
            ).fit(train_compounds=train_compounds)
            scores, pred_compounds = eval_splitter(
                splitter=splitter,
                test_compounds=test_compounds
            )
            mean_score = scores.mean()
            output = {
                "scores": scores,
                "mean_score": mean_score,
                "train_size": len(train_compounds),
                "test_size": len(test_compounds),
                "splitter_name": splitter_cls.name,
                "splitter_metadata": splitter._metadata()
            }
            outputs.append(output)
            if mean_score > best_score:
                best_score = mean_score
                best_pairs = (test_compounds, pred_compounds)
                if hasattr(splitter, "plot_buffer") and splitter.plot_buffer is not None:
                    best_plot_buffer = splitter.plot_buffer
            else:
                if hasattr(splitter, "plot_buffer") and splitter.plot_buffer is not None:
                    splitter.plot_buffer.truncate(0)
                    splitter.plot_buffer.close()
            n_iter += 1
    outputs = sorted(outputs, key=lambda entry: entry["mean_score"], reverse=True)
    return outputs, best_pairs, best_plot_buffer


def benchmark_splitters(
    *,
    cls2params: Dict[str, Tuple[List[int], List[dict], str]],
    gecodb_path: str,
    out_dir: str,
    main_suffix: Optional[str]="",
    train_size: Optional[float]=0.8,
    rec_best_of_all: Optional[bool]=True,
    verbose: Optional[bool]=True
) -> None:
    
    """
    Benchmark one or more splitters given their different configurations.

    Parameters
    ----------
    cls2params : `Dict[str, Tuple[List[int], List[bool], List[dict], str]]`
        mapping between name of the splitter (according to its `.name` attribute)
        and tuple out of 4 elements:
            1. `List[int]`: minimal counts of compounds to keep;
            all compounds occurring less will be dropped at the respective iteration
            2. `List[dict]`: list of tested configurations for the splitter,
            with each configuration being a mapping between parameter name and its value
            3. `str`: suffix to prepend to this model out directory (<class_name_x_suffix> below)

    gecodb_path : `str`
        path to the TSV DECOW16-format compounds dataset

    out_dir : `str`
        path to directory to put the results to

    main_suffix : `str`, optional
        suffix to append to all output files before the class suffices

    train_size: `float`, optional, defaults to `0.8`
        train size

    rec_best_of_all : `bool` optional, defaults to `True`
        whether to additionally compile all scores in one file and save
        the best of all predictions 

    verbose : `bool` optional, defaults to `True`
        whether to show progress bar when fitting and predicting compounds

    Notes
    -----
    The result of evaluation will be written to `out_dir` with the following structure:
        out_dir
        --- <name of the class_name_1 object>
            --- scores<main_suffix><class_name_1_suffix>.json: JSON with sorted scores for the class_name_1 model
            --- best_preds<main_suffix><class_name_1_suffix>.tsv:
                TSV with columns of golds and preds from the best config of the class_name_1 model
            --- best_plot<main_suffix><class_name_1_suffix>.png: (if given) training plot
                from the best config of the class_name_1 model
        --- <name of the class_name_2 object>
            ...
        --- ...
            ...
        all_scores<main_suffix>.json: JSON with all sorted scores
        best_preds<main_suffix>.tsv: TSV with columns of golds and preds from the best config of the best model
    """

    all_outputs = []
    best_pair_of_all = None
    best_score = -1
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    for splitter_name, (min_counts, all_splitter_params, suffix) in cls2params.items():
        splitter_cls = dekor.splitters.__all_splitters__[splitter_name]
        target_dir = os.path.join(out_dir, splitter_cls.name)
        if not os.path.exists(target_dir): os.mkdir(target_dir)
        outputs, best_pairs, best_plot_buffer = benchmark_splitter(
            splitter_cls=splitter_cls,
            min_counts=min_counts,
            all_splitter_params=all_splitter_params,
            gecodb_path=gecodb_path,
            train_size=train_size,
            verbose=verbose
        )
        all_outputs += outputs
        outputs_path = os.path.join(target_dir, f"scores{main_suffix}{suffix}.json")
        with open(outputs_path, 'w', encoding="utf8") as out:
            json.dump(outputs, out, indent=4)
        pairs_path = os.path.join(target_dir, f"best_preds{main_suffix}{suffix}.tsv")
        with open(pairs_path, 'w', encoding="utf8") as pairs:
            pairs.writelines([
                f"{gold.raw}\t{pred.raw}\n"
                for gold, pred in zip(*best_pairs)
            ])
        if best_plot_buffer:
            best_plot_path = os.path.join(target_dir, f"best_plot{main_suffix}{suffix}.png")
            best_plot_buffer.seek(0)
            plot = Image.open(best_plot_buffer)
            info = PngInfo()
            for key, value in plot.text.items(): info.add_text(key, value)
            plot.save(best_plot_path, format="png", pnginfo=info)
        if outputs[0]["mean_score"] > best_score:
            best_score = outputs[0]["mean_score"]
            best_pair_of_all = best_pairs
    if rec_best_of_all:
        all_outputs = sorted(all_outputs, key=lambda entry: entry["mean_score"], reverse=True)
        all_outputs_path = os.path.join(out_dir, f"all_scores{main_suffix}.json")
        with open(all_outputs_path, 'w', encoding="utf8") as out:
            json.dump(all_outputs, out, indent=4)
        all_pairs_path = os.path.join(out_dir, f"best_preds_of_all{main_suffix}.tsv")
        with open(all_pairs_path, 'w', encoding="utf8") as pairs:
            pairs.writelines([
                f"{gold.raw}\t{pred.raw}\n"
                for gold, pred in zip(*best_pair_of_all)
            ])