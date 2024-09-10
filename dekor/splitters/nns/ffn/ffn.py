"""
FFN model for splitting German compounds based on the DECOW16 compound data.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

from dekor.splitters.base import DEVICE
from dekor.splitters.nns.base import BaseNN, BaseNNSplitter


class FFN(BaseNN):

    def __init__(
            self,
            *,
            input_size: int,
            output_size: int,
            hidden_size: Optional[int]=64,
            activation: Optional[Literal["relu", "tanh"]]="relu",
            dropout_rate: Optional[float]=0.025,
            require_softmax: Optional[bool]=False,
            batch_size: Optional[int]=4096
        ) -> None:

        assert activation in ["relu", "tanh"]
        super(FFN, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            activation=activation,
            dropout_rate=dropout_rate,
            require_softmax=require_softmax,
            batch_size=batch_size
        )      

    def _build_self(self):
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
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor, force_softmax=False):
        # input: b x i
        output = self.linear(input_tensor)   # b x h
        output = self.activation(output)    # b x h
        if self.dropout_rate:
            output = self.dropout(output)   # b x h
        output = self.dense(output)    # b x o
        if self.require_softmax or force_softmax: output = self.softmax(output)  # b x o
        return output
    

class FFNSplitter(BaseNNSplitter):

    name = "ffn"

    def _build_nn(self) -> None:
        self.model = FFN(
            input_size=self.maxlen,
            output_size=len(self.vocab_links),
            **self.model_params,
            # CrossEntropy is supposed to be used with raw logits
            require_softmax=self.criterion != "crossentropy"
        )

    def _train_on_batch(self, x: torch.Tensor, y: torch.Tensor) -> int:
        output = self.model(x)
        loss = self.criterion(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _pad_batch(self, batch: torch.Tensor) -> torch.Tensor:
        diff = self.model.batch_size - len(batch)
        pads = [self.pad([])] * diff
        pads = torch.tensor(pads, dtype=torch.float, device=DEVICE)
        padded_batch = torch.cat((batch, pads), dim=0)
        return padded_batch
    
    def _predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(x, force_softmax=True)
            output = output.detach()
        return output