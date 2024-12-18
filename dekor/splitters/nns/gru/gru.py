"""
GRU model for splitting German compounds based on the DECOW16 compound data.
"""

import torch.nn as nn
from typing import Optional

from dekor.splitters.base import DEVICE
from dekor.splitters.nns.base import BaseRecurrentNN, BaseRecurrentNNSplitter


class GRU(BaseRecurrentNN):

    def __init__(
            self,
            *,
            input_size: int,
            hidden_size: Optional[int]=16,
            output_size: int,
            dropout_rate: Optional[float]=0.1,
            num_layers: Optional[int]=1,
            require_softmax: Optional[bool]=False
        ) -> None:
        super(GRU, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout_rate=dropout_rate,
            num_layers=num_layers,
            require_softmax=require_softmax
        )

    def _build_self(self) -> None:
        self.recurrent = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout_rate,
            # using only forward: in German, compound links depend only on left constituent
			# so only left to right order matters
            bidirectional=False,
            num_layers=self.num_layers,
            batch_first=True,
            device=DEVICE
        )
        self.dense = nn.Linear(
            in_features=(self.D * self.hidden_size) * 3,    # 3 gru outputs concatenated,
            out_features=self.output_size,
            device=DEVICE
        )
        self.softmax = nn.Softmax(dim=1)
    

class GRUSplitter(BaseRecurrentNNSplitter):

    """
    GRU model for splitting German compounds based on the DECOW16 compound data.

    Parameters
    ----------
    input_size : `int`
        input size (number of input features)

    hidden_size : `int`, optional, defaults to 16
        size of the hidden layer

    output_size : `int`
        output size (number of output classes)

    dropout_rate : `float`, optional, defaults to `0.1`
        dropout rate for `torch.nn.GRU`

    num_layers : `int`, optional, defaults to `1`
        number of layers in `torch.nn.GRU` 
    """

    name = "gru"
    path = ".pretrained/nns/gru.pt"

    def _build_nn(self) -> None:
        self.nn = GRU(
            input_size=self.embeddings._embedding_dim,
            output_size=len(self.vocab_links),
            **self.nn_params,
            # CrossEntropy is supposed to be used with raw logits
            require_softmax=self.criterion != "crossentropy"
        )