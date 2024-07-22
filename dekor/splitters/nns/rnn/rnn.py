"""
RNN model for splitting German compounds based on the DECOW16 compound data.
"""

import torch
import torch.nn as nn
from typing import Optional, Optional, Literal

from dekor.splitters.base import DEVICE
from dekor.splitters.nns.base import BaseNN, BaseNNSplitter


class RNN(BaseNN):

    def __init__(
            self,
            *,
            vocab_size: int,
            output_size: int,
            hidden_size: Optional[int]=64,
            embedding_dim: Optional[int]=16,
            bidirectional: Optional[bool]=True,
            num_layers: Optional[int]=2,
            activation: Optional[Literal["relu", "tanh"]]="tanh",
            dropout: Optional[float]=0.05,
            require_softmax: Optional[bool]=False,
            batch_size: Optional[int]=4096
        ) -> None:
        assert activation in ["relu", "tanh"]
        super(RNN, self).__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            output_size=output_size,
            embedding=nn.Embedding(vocab_size, embedding_dim),
            bidirectional=bidirectional,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            require_softmax=require_softmax,
            batch_size=batch_size
        )    

    def _build_self(self):
        self.model = nn.RNN(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            nonlinearity=self.activation,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            batch_first=True,
            device=DEVICE
        )
        self.linear = nn.Linear(
            in_features=self.D * self.hidden_size,
            out_features=self.output_size,
            device=DEVICE
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor, hidden_tensor, force_softmax=False):
        # input: b x 1 (x 1), hidden: D * nl x b x h
        embedded_tensor = self.embedding(input_tensor)  # b x 1 x emd
        output, hidden = self.model(embedded_tensor, hidden_tensor)   # b x 1 x D * h, D * nl x b x h
        output = self.linear(output)    # b x 1 x o
        output = output.squeeze(1)  # b x o, for softmax as it works on dimension 1
        if self.require_softmax or force_softmax: output = self.softmax(output)   # b x o
        return output, hidden
    
    @property
    def D(self):
        return 1 if not self.bidirectional else 2

    def init_hidden(self):
        return torch.zeros(self.D * self.num_layers, self.batch_size, self.hidden_size)
    

class RNNSplitter(BaseNNSplitter):

    name = "rnn"

    def _build_nn(self) -> None:
        self.model = RNN(
            vocab_size=len(self.vocab_chars) + 1,   # plus ignore index
            output_size=len(self.vocab_links),
            **self.model_params,
            # CrossEntropy is supposed to be used with raw logits
            require_softmax=self.criterion != "crossentropy"
        )

    def _train_on_batch(self, x: torch.Tensor, y: torch.Tensor) -> int:
        
        hidden = self.model.init_hidden()

        # iterate over features
        for i in range(x.size(1)):
            # in the base class, the X tensor is float; but for embeddings,
            # we need 1) integers and 2) they must be wrapped into 1 x 1 tensors
            input = x[:, i].long().unsqueeze(-1)
            output, hidden = self.model(input, hidden)

        loss = self.criterion(output, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _pad_batch(self, batch: torch.Tensor) -> torch.Tensor:
        diff = self.model.batch_size - len(batch)
        pads = [self.pad([])] * diff
        # we need integers, but that will be done in `_predict_batch()`
        pads = torch.tensor(pads, device=DEVICE)
        padded_batch = torch.cat((batch, pads), dim=0)
        return padded_batch
    
    def _predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        
        with torch.no_grad():

            hidden = self.model.init_hidden()

            # iterate over features
            for i in range(x.size(1)):
                # in the base class, the X tensor is float; but for embeddings,
                # we need 1) integers and 2) they must be wrapped into 1 x 1 tensors
                input = x[:, i].long().unsqueeze(-1)
                output, hidden = self.model(input, hidden, force_softmax=True)

            output = output.detach()

        return output