from itertools import product
from typing import List, Dict, Any

from dekor.benchmarking.benchmarking import benchmark_splitters


def grid_parameters(parameters: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    return [
        {key: v for key, v in zip(parameters.keys(), value)}
        for value in product(*parameters.values())
    ]


def main():

    ngram_options = {
        "n": [4],
        "record_none_links": [False]
    }
    rnn_options = {
        "n": [3],
        "hidden_size": [32],
        "embedding_dim": [8]
    }
    
    cls2params = {
        "NGramsSplitter": (
            [100000],
            grid_parameters(ngram_options),
            "_suka"
        ),
        "RNNSplitter": (
            [100000],
            grid_parameters(rnn_options),
            "_puka"
        )
    }

    benchmark_splitters(
        cls2params=cls2params,
        gecodb_path="./resources/gecodb_v04.tsv",
        out_dir="./benchmarking_resuts",
        main_suffix="_mazafaka",
        verbose=True
    )


if __name__ == "__main__":
    main()