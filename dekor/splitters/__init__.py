"""
Base model for splitting German compounds based on the DECOW16 compound data.
"""

from dekor.splitters.ngrams import NGramsSplitter
from dekor.splitters.nns import FFNSplitter, RNNSplitter

__all__ = {
    NGramsSplitter.name: NGramsSplitter,
    FFNSplitter.name: FFNSplitter,
    RNNSplitter.name: RNNSplitter
}