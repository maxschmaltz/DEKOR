"""
GRU model for splitting German compounds based on the DECOW16 compound data.
"""

import torch.nn as nn
from typing import Optional, Literal

from dekor.splitters.base import DEVICE
from dekor.splitters.nns.base import BaseRecurrentNN, BaseRecurrentNNSplitter


class GRU(BaseRecurrentNN):

    def __init__(
            self,
            *,
            vocab_size: int,
            embedding_dim: Optional[int]=16,
            hidden_size: Optional[int]=64,
            output_size: int,
            bidirectional: Optional[bool]=True,
            num_layers: Optional[int]=2,
            dropout_rate: Optional[float]=0.0025,
            require_softmax: Optional[bool]=False,
            batch_size: Optional[int]=4096
        ) -> None:
        super(GRU, self).__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            output_size=output_size,
            embedding=nn.Embedding(vocab_size, embedding_dim),
            bidirectional=bidirectional,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            require_softmax=require_softmax,
            batch_size=batch_size
        )    

    def _build_self(self):
        self.recurrent = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout_rate,
            batch_first=True,
            device=DEVICE
        )
        self.linear = nn.Linear(
            in_features=self.D * self.hidden_size,
            out_features=self.output_size,
            device=DEVICE
        )
        self.softmax = nn.Softmax(dim=1)
    

class GRUSplitter(BaseRecurrentNNSplitter):

    name = "gru"

    def _build_nn(self) -> None:
        self.model = GRU(
            vocab_size=len(self.vocab_chars) + 1,   # plus ignore index
            output_size=len(self.vocab_links),
            **self.model_params,
            # CrossEntropy is supposed to be used with raw logits
            require_softmax=self.criterion != "crossentropy"
        )