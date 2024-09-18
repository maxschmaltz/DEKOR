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
            convolution_size: Optional[int]=3,
            hidden_size: Optional[int]=64,
            output_size: int,
            activation: Optional[Literal["relu", "tanh"]]="relu",
            reduction: Optional[Literal["max", "conv"]]="max",
            dropout_rate: Optional[float]=0.025,
            require_softmax: Optional[bool]=False
        ) -> None:
        super(CNN, self).__init__(
            input_size=input_size,
            convolution_size=convolution_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation=activation,
            reduction=reduction,
            dropout_rate=dropout_rate,
            require_softmax=require_softmax
        )      

    def _build_self(self) -> None:
        self.convolution = nn.Conv1d(
            in_channels=self.input_size,
            out_channels=self.hidden_size,
            kernel_size=self.convolution_size,   # size of sliding window to convolute
            stride=1,   # step the window slides
            padding="same", # this pads the input so that the number of output seqs equals the number of input seqs
            device=DEVICE
        )
        self.activation = nn.ReLU() if self.activation == "relu" else nn.Tanh()
        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        if self.reduction == "conv":
            self.reduction_conv = nn.Conv1d(
            in_channels=3,    # because first convolution gives b x h x 3 and we need to reduce 3
            out_channels=1,
            kernel_size=2,   # size of sliding window to convolute
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

    def forward(self, input_tensor: torch.Tensor, force_softmax: Optional[bool]=False):
        # input: # b x emb x 3, already embedded
        # just as for RNN, we want to proces the 3 parts separately and concatenate the result
        # before linear layer; see more in `BaseRecurrentNNSplitter`
        # convolutional layer expects channels (`embedding_dim` in our case) and then
        # input size for each batch; we have sequence length of 1 (a single embedding) so
        # we have to introduce a new dimension
        input_tensor = input_tensor.permute([0, 2, 1]) # b x 3 x emb
        output = self.convolution(input_tensor)   # b x h x 3
        # revert permutation
        output = output.permute([0, 2, 1])    # b x 3 x h
        output = self.activation(output)    # b x 3 x h
        if self.dropout_rate:
            output = self.dropout(output)   # b x 3 x h
        # at this point, convolution & activation captured properties of different input slices;
        # if we apply the dense layer now, this will connect all of these slices with the outputs;
        # we cannot however have a distribution for each single slice, we need one for the input as a whole;
        # that is why we should either take the most "informative" (weight-wise) slice, or convolute them down
        if self.reduction == "max":
            output, _ = torch.max(output, dim=1)    # b x h
        else:   # if self.reduction == "conv":
            output = self.reduction_conv(output) # b x 1 x h
            output = output.squeeze(1)  # b x h, for softmax as it works on dimension 1
        output = self.dense(output)    # b x o
        if self.require_softmax or force_softmax: output = self.softmax(output)  # b x o
        return output
    

class CNNSplitter(BaseForwardNNSplitter):

    name = "cnn"
    path = ".pretrained/nns/cnn.pt"

    def _build_nn(self) -> None:
        self.nn = CNN(
            input_size=self.embeddings._embedding_dim,   # triplets are concatenated
            output_size=len(self.vocab_links),
            **self.nn_params,
            # CrossEntropy is supposed to be used with raw logits
            require_softmax=self.criterion != "crossentropy"
        )