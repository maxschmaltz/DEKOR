import os
import json
from itertools import product
from sklearn.model_selection import train_test_split
from typing import Optional, List, Dict

from dekor.splitters.base import BaseSplitter
import dekor.splitters
from dekor.utils.gecodb_parser import parse_gecodb, Compound, Tuple
from dekor.eval.evaluator import CompoundEvaluator

evaluator = CompoundEvaluator()


def predict_with_splitter(
    splitter: BaseSplitter,
    test_compounds: List[Compound]
):
    test_lemmas = [
        compound.lemma for compound in test_compounds
    ]
    pred_compounds = splitter.predict(test_lemmas)
    scores = evaluator.evaluate(test_compounds, pred_compounds)
    return pred_compounds, scores


def eval_splitter(
    *,
    unfit_splitter: BaseSplitter,
    train_compounds: List[Compound],
    test_compounds: List[Compound]
):
    splitter = unfit_splitter.fit(train_compounds)
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
):
    outputs = []
    best_score = -1
    best_pairs = None
    eliminate_allomorphys = [True, False]
    for min_count, eliminate_allomorphy in product(min_counts, eliminate_allomorphys):
        gecodb = parse_gecodb(
            gecodb_path,
            eliminate_allomorphy=eliminate_allomorphy,
            min_count=min_count
        )
        train_data, test_data = train_test_split(gecodb, train_size=train_size, shuffle=True)
        train_compounds = train_data["compound"].values
        test_compounds = test_data["compound"].values
        for splitter_params in all_splitter_params:
            unfit_splitter = splitter_cls(
                **splitter_params,
                eliminate_allomorphy=eliminate_allomorphy,
                verbose=verbose
            )
            if verbose: # print out params
                print('\n', ', '.join(f'{key}: {value}' for key, value in splitter_params.items()))
            scores, pred_compounds = eval_splitter(
                unfit_splitter=unfit_splitter,
                train_compounds=train_compounds,
                test_compounds=test_compounds
            )
            mean_score = scores.mean()
            output = {
                "scores": scores,
                "mean_score": mean_score,
                "train_size": len(train_compounds),
                "test_size": len(test_compounds),
                "splitter_name": splitter_cls.name,
                "splitter_metadata": unfit_splitter._metadata()
            }
            outputs.append(output)
            if mean_score > best_score:
                best_score = mean_score
                best_pairs = (test_compounds, pred_compounds)
    outputs = sorted(outputs, key=lambda entry: entry["mean_score"], reverse=True)
    return outputs, best_pairs


def benchmark_splitters(
    *,
    cls2params: Dict[str, Tuple[List[int], List[dict], str]],
    gecodb_path: str,
    out_dir: str,
    main_suffix: Optional[str]="",
    train_size: Optional[float]=0.8,
    verbose: Optional[bool]=True
):
    all_outputs = []
    best_pair_of_all = None
    best_score = -1
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    for splitter_cls_name, (min_counts, all_splitter_params, suffix) in cls2params.items():
        splitter_cls = getattr(dekor.splitters, splitter_cls_name)
        target_dir = os.path.join(out_dir, splitter_cls.name)
        if not os.path.exists(target_dir): os.mkdir(target_dir)
        outputs, best_pairs = benchmark_splitter(
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
        if outputs[0]["mean_score"] > best_score:
            best_score = outputs[0]["mean_score"]
            best_pair_of_all = best_pairs
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