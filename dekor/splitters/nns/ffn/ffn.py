"""
FFN model for splitting German compounds based on the DECOW16 compound data.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

from dekor.splitters.base import DEVICE
from dekor.splitters.nns.base import BaseNN, BaseForwardNNSplitter


class FFN(BaseNN):

    def __init__(
            self,
            *,
            input_size: int,
            hidden_size: Optional[int]=64,
            output_size: int,
            activation: Optional[Literal["relu", "tanh"]]="relu",
            dropout_rate: Optional[float]=0.025,
            require_softmax: Optional[bool]=False
        ) -> None:
        super(FFN, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation=activation,
            dropout_rate=dropout_rate,
            require_softmax=require_softmax
        )      

    def _build_self(self) -> None:
        self.linear = nn.Linear(
            in_features=self.input_size,
            out_features=self.hidden_size,
            device=DEVICE
        )
        self.activation = nn.ReLU() if self.activation == "relu" else nn.Tanh()
        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        self.dense = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.output_size,
            device=DEVICE
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor: torch.Tensor, force_softmax: Optional[bool]=False):
        # input: # b x emb x 3, already embedded
        # for FFN, we want to pass a single embedding for the whole context
        # so first, we concatenate them
        input_tensor = input_tensor.reshape(-1, self.input_size)    # b x (emb x 3)
        output = self.linear(input_tensor)   # b x h
        output = self.activation(output)    # b x h
        if self.dropout_rate:
            output = self.dropout(output)   # b x h
        output = self.dense(output)    # b x 0
        if self.require_softmax or force_softmax:
            output = self.softmax(output)  # b x o
        return output
    

class FFNSplitter(BaseForwardNNSplitter):

    name = "ffn"

    def _build_nn(self) -> None:
        self.nn = FFN(
            input_size=self.embeddings._embedding_dim * 3,   # triplets are concatenated
            output_size=len(self.vocab_links),
            **self.nn_params,
            # CrossEntropy is supposed to be used with raw logits
            require_softmax=self.criterion != "crossentropy"
        )