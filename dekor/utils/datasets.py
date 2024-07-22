"""
Module with custom dataset formats.
"""

from torch.utils.data import Dataset

from typing import Optional, Iterable, Any


class XYDataset(Dataset):

    """
    Simple dataset for batching.

    Parameters:
    X : `Iterable[Any]`
        train sequence

    Y : `Iterable[Any]`, optional
        target sequence; optional, if not provided, the dataset will return
        only train subsequences upon iteration
    """

    def __init__(self, X: Iterable[Any], Y: Optional[Iterable[Any]]=None) -> None:
        self.X = X
        self.Y = Y
        if self.Y is not None: assert len(X) == len(Y)

    def __getitem__(self, index: int) -> Any:
        return (self.X[index], self.Y[index]) if self.Y is not None else self.X[index]
    
    def __len__(self) -> int:
        return len(self.X)