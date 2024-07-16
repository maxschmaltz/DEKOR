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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "params_mapping_path", help="""Path to a JSON with of following structure:
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
        """
    )
    parser.add_argument("gecodb_path", help="path to GecoDB compound dataset")
    parser.add_argument("out_dir", help="path to directory to put the results to")
    parser.add_argument("--suffix", "-s", help="suffix to append to all output files before the class suffices")
    parser.add_argument("--train_size", "-ts", type=float, default=0.8, help="train size")
    parser.add_argument("-q", action="store_false", help="suppress verbose")

    args = parser.parse_args()

    with open(args.params_mapping_path) as pm: params_mapping = json.load(pm)
    cls2params = {
        key: [
            min_counts,
            grid_parameters(params_options),
            suffix
        ]
        for key, (min_counts, params_options, suffix) in params_mapping.items()
    }

    benchmark_splitters(
        cls2params=cls2params,
        gecodb_path=args.gecodb_path,
        out_dir=args.out_dir,
        train_size=args.train_size,
        main_suffix=args.suffix,
        verbose=args.q
    )


# example: python3 ./dekor/benchmarking/main.py ./dekor/benchmarking/configs/test.json ./resources/gecodb_v04.tsv ./benchmarking_resuts -s _test
if __name__ == "__main__":
    main()