"""
RNN model for splitting German compounds based on the DECOW16 compound data.
"""

from dekor.splitters.nns.ffn import FFNSplitter
from dekor.splitters.nns.rnn import RNNSplitter
from dekor.splitters.nns.gru import GRUSplitter
from dekor.splitters.nns.cnn import CNNSplitter