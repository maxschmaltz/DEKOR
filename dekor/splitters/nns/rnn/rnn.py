"""
RNN model for splitting German compounds based on the DECOW16 compound data.
"""

import torch.nn as nn
from typing import Optional, Literal

from dekor.splitters.base import DEVICE
from dekor.splitters.nns.base import BaseRecurrentNN, BaseRecurrentNNSplitter


class RNN(BaseRecurrentNN):

    def __init__(
            self,
            *,
            input_size: int,
            hidden_size: Optional[int]=64,
            output_size: int,
            activation: Optional[Literal["relu", "tanh"]]="tanh",
            dropout_rate: Optional[float]=0.0025,
            num_layers: Optional[int]=2,
            require_softmax: Optional[bool]=False
        ) -> None:
        assert activation in ["relu", "tanh"]
        super(RNN, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation=activation,
            dropout_rate=dropout_rate,
            num_layers=num_layers,
            require_softmax=require_softmax
        )

    def _build_self(self) -> None:
        self.recurrent = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            nonlinearity=self.activation,
            dropout=self.dropout_rate,
			# using only forward: in German, compound links depend only on left constituent
			# so only left to right order matters
            bidirectional=False,
            num_layers=self.num_layers,
            batch_first=True,
            device=DEVICE
        )
        self.dense = nn.Linear(
            in_features=(self.D * self.hidden_size) * 3,    # 3 rnn outputs concatenated
            out_features=self.output_size,
            device=DEVICE
        )
        self.softmax = nn.Softmax(dim=1)
    

class RNNSplitter(BaseRecurrentNNSplitter):

    name = "rnn"

    def _build_nn(self) -> None:
        self.nn = RNN(
            input_size=self.embeddings._embedding_dim,
            output_size=len(self.vocab_links),
            **self.nn_params,
            # CrossEntropy is supposed to be used with raw logits
            require_softmax=self.criterion != "crossentropy",
            batch_size=self.batch_size  # for hidden init
        )