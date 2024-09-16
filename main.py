import argparse
import json
from itertools import product
from typing import List, Dict, Any

from dekor.benchmarking.benchmarking import benchmark_splitters


def grid_parameters(parameters: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    return [
        {key: v for key, v in zip(parameters.keys(), value)}
        for value in product(*parameters.values())
    ]


def main() -> None:

    """
    Run benchmarking of splitters. CLI-based.

    CLI Parameters
    --------------
    params_mapping_path : `str`
        path to a JSON with of following structure:
        {
            class_name_1: [
                [min_count_1, min_count_2, ...],
                {
                    param_1: [param_1_val_1, param_1_val_2, ...],
                    param_2: [param_2_val_1, param_2_val_2, ...],
                    ...
                },
                class_name_1_suffix
            ],
            class_name_2: ...,
            ...
        }
        where min_count_x is the minimum count for parsing GecoDB,
        param_x is a list of values for param_x; all param values for all params
        will be combined, so all the configurations will be tested. Note
        that both variant with eliminating allomorphy and not will be run

    gecodb_path : `str`
        path to GecoDB compound dataset

    out_dir : `str`
        path to directory to put the results to

    --suffix, -s : `str`, optional
        suffix to append to all output files before the class suffices

    --train_size", "-ts", `float`, optional, defaults to `0.8`
        train size

    -x : `bool` optional, defaults to `True`
        whether to suppress additional compilation all scores in one file and saving
        the best of all predictions; `True` if given, `False` if missing

    -q, optional, defaults to `False`,
        whether to suppress verbose; `True` if given, `False` if missing

    Notes
    -----
    The result of evaluation will be written to `out_dir` with the following structure:
        out_dir
        --- <name of the class_name_1 object>
            --- scores<--suffix><class_name_1_suffix>.json: JSON with sorted scores for the class_name_1 model
            --- best_preds<--suffix><class_name_1_suffix>.tsv:
                TSV with columns of golds and preds from the best config of the class_name_1 model
            --- best_plot<--suffix><class_name_1_suffix>.png: (if given) training plot
                from the best config of the class_name_1 model
        --- <name of the class_name_2 object>
            ...
        --- ...
            ...
        all_scores<--suffix>.json: JSON with all sorted scores
        best_preds<--suffix>.tsv: TSV with columns of golds and preds from the best config of the best model
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "params_mapping_path", help="""path to a JSON with of following structure:
        {
            class_name_1: [
                [min_count_1, min_count_2, ...],
                {
                    param_1: [param_1_val_1, param_1_val_2, ...],
                    param_2: [param_2_val_1, param_2_val_2, ...],
                    ...
                },
                class_name_1_suffix
            ],
            class_name_2: ...,
            ...
        }
        where min_count_x is the minimum count for parsing GecoDB,
        param_x is a list of values for param_x; all param values for all params
        will be combined, so all the configurations will be tested. Note
        that both variant with eliminating allomorphy and not will be run
        """
    )
    parser.add_argument("gecodb_path", help="path to GecoDB compound dataset")
    parser.add_argument("out_dir", help="path to directory to put the results to")
    parser.add_argument("--suffix", "-s", default="", help="suffix to append to all output files before the class suffices")
    parser.add_argument("--train_size", "-ts", type=float, default=0.8, help="train size")
    parser.add_argument("-x", action="store_false", default=True, help="whether to suppress `rec_best_of_all`")
    parser.add_argument("-q", action="store_false", default=True, help="whether to suppress verbose")

    args = parser.parse_args()

    with open(args.params_mapping_path) as pm: params_mapping = json.load(pm)
    cls2params = {
        key: [
            min_counts,
            elliminate_allomorphys,
            grid_parameters(params_options),    # combine param values
            suffix
        ]
        for key, (min_counts, elliminate_allomorphys, params_options, suffix) in params_mapping.items()
    }

    benchmark_splitters(
        cls2params=cls2params,
        gecodb_path=args.gecodb_path,
        out_dir=args.out_dir,
        train_size=args.train_size,
        main_suffix=args.suffix,
        rec_best_of_all=args.x,
        verbose=args.q
    )


# example: python3 ./dekor/benchmarking/main.py ./dekor/benchmarking/configs/test.json ./resources/gecodb_v04.tsv ./benchmarking_resuts --suffix _test -q
if __name__ == "__main__":
    main()