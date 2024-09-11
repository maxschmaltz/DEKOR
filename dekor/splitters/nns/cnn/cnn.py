"""
CNN model for splitting German compounds based on the DECOW16 compound data.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

from dekor.splitters.base import DEVICE
from dekor.splitters.nns.base import BaseNN, BaseForwardNNSplitter


class CNN(BaseNN):

    def __init__(
            self,
            *,
            input_size: int,
            vocab_size: int,
            embedding_dim: Optional[int]=16,
            window_size: Optional[int]=3,
            hidden_size: Optional[int]=64,
            output_size: int,
            activation: Optional[Literal["relu", "tanh"]]="relu",
            reduction: Optional[Literal["max", "conv"]]="max",
            dropout_rate: Optional[float]=0.025,
            require_softmax: Optional[bool]=False,
            batch_size: Optional[int]=4096
        ) -> None:

        assert activation in ["relu", "tanh"]
        assert reduction in ["max", "conv"]
        super(CNN, self).__init__(
            input_size=input_size,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            window_size=window_size,
            hidden_size=hidden_size,
            output_size=output_size,
            embedding=nn.Embedding(vocab_size, embedding_dim),
            activation=activation,
            reduction=reduction,
            dropout_rate=dropout_rate,
            require_softmax=require_softmax,
            batch_size=batch_size
        )      

    def _build_self(self):
        self.convolution = nn.Conv1d(
            in_channels=self.embedding_dim,
            out_channels=self.hidden_size,
            kernel_size=self.window_size,   # size of sliding window to convolute
            stride=1,   # step the window slides
            padding="same", # this pads the input so that the number of output seqs equals the number of input seqs
            device=DEVICE
        )
        self.activation = nn.ReLU() if self.activation == "relu" else nn.Tanh()
        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        if self.reduction == "conv":
            self.reduction = nn.Conv1d(
            in_channels=self.input_size,    # because first convolution gives b x h x i and we need to reduce i
            out_channels=1,
            kernel_size=self.window_size,   # size of sliding window to convolute
            stride=1,   # step the window slides
            padding="same", # this pads the input so that the number of output seqs equals the number of input seqs
            device=DEVICE
        )
        self.dense = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.output_size,
            device=DEVICE
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor, force_softmax=False):
        # input: b x i
        # in the base class, the X tensor is float; but for embeddings, we need integers
        output = self.embedding(input_tensor.long())   # b x i x emb
        # convolutional layer expects channels (`embedding_dim` in our case) first so we have to permute
        output = output.permute([0, 2, 1])  # b x emb x i
        output = self.convolution(output)   # b x h x i
        # revert permutation
        output = output.permute([0, 2, 1])    # b x i x h
        output = self.activation(output)    # b x i x h
        if self.dropout_rate:
            output = self.dropout(output)   # b x i x h
        # at this point, convolution & activation captured properties of different input slices;
        # if we apply the dense layer now, this will connect all of these slices with the outputs;
        # we cannot however have a distribution for each single slice, we need one for the input as a whole;
        # that is why we should either take the most "informative" (weight-wise) slice, or convolute them down
        if self.reduction == "max":
            output, _ = torch.max(output, dim=1)    # b x h
        else:   # if self.reduction == "conv":
            output = self.reduction(output) # b x 1 x h
            output = output.squeeze(1)  # b x h, for softmax as it works on dimension 1
        output = self.dense(output)    # b x o
        if self.require_softmax or force_softmax: output = self.softmax(output)  # b x o
        return output
    

class CNNSplitter(BaseForwardNNSplitter):

    name = "cnn"

    def _build_nn(self) -> None:
        self.model = CNN(
            input_size=self.maxlen,
            vocab_size=len(self.vocab_chars) + 1,   # plus ignore index
            output_size=len(self.vocab_links),
            **self.model_params,
            # CrossEntropy is supposed to be used with raw logits
            require_softmax=self.criterion != "crossentropy"
        )