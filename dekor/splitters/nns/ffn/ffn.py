"""
FFN model for splitting German compounds based on the DECOW16 compound data.
"""

import re
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from io import BytesIO
from tqdm import tqdm
from typing import Optional, Iterable, Optional, List, Tuple, Literal, Dict

from dekor.splitters.base import BaseSplitter, DEVICE
from dekor.splitters.nns.base import BaseNN, BaseNNSplitter
from dekor.utils.gecodb_parser import (
    Compound,
    Link,
    UMLAUTS_REVERSED
)
from dekor.utils.vocabs import StringVocab
from dekor.utils.datasets import XYDataset


class FFN(BaseNN):

    def __init__(
            self,
            *,
            input_size: int,
            output_size: int,
            hidden_size: Optional[int]=64,
            activation: Optional[Literal["relu", "tanh"]]="relu",
            dropout: Optional[float]=0.05,
            require_softmax: Optional[bool]=False,
            batch_size: Optional[int]=4096
        ) -> None:

        assert activation in ["relu", "tanh"]
        super(FFN, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            activation=activation,
            dropout=dropout,
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
        self.dense = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.output_size,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor, force_softmax=False):
        # input: b x i
        output = self.linear(input_tensor)   # b x h
        output = self.activation(output)    # b x h
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
    

if __name__ == "__main__":

    from dekor.utils.gecodb_parser import parse_gecodb
    from sklearn.model_selection import train_test_split
    from dekor.benchmarking.benchmarking import eval_splitter

    gecodb_path = "./resources/gecodb_v04.tsv"
    gecodb = parse_gecodb(gecodb_path, eliminate_allomorphy=True, min_count=10000)
    train_data, test_data = train_test_split(gecodb, train_size=0.75, shuffle=True)
    train_compounds = train_data["compound"].values
    test_compounds = test_data["compound"].values
    test_lemmas = [
        compound.lemma for compound in test_compounds
    ]
    splitter = FFNSplitter(
        n=3,
        eliminate_allomorphy=True,
        batch_size=4096,
        verbose=True,
        criterion="margin"
    ).fit(train_compounds)
    _, pred_compounds = eval_splitter(
        splitter=splitter,
        test_compounds=test_compounds
    )
    pass