"""
Base model for splitting German compounds based on the DECOW16 compound data.
"""

from dekor.splitters.ngrams import NGramsSplitter
from dekor.splitters.nns import FFNSplitter, RNNSplitter, GRUSplitter, CNNSplitter

__all__ = {
    NGramsSplitter.name: NGramsSplitter,
    FFNSplitter.name: FFNSplitter,
    RNNSplitter.name: RNNSplitter,
	GRUSplitter.name: GRUSplitter,
	CNNSplitter.name: CNNSplitter
}