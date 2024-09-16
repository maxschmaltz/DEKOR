import asyncio
import numpy as np
import torch
import torch.nn as nn
from typing import Iterable, Awaitable, Tuple

from dekor.splitters.base import DEVICE
from dekor.embeddings.base import BaseEmbeddings
from dekor.utils.vocabs import StringVocab


class TorchEmbeddings(BaseEmbeddings, nn.Module):

	name = "torch"

	def __init__(
		self,
		*
		vocab: StringVocab,
		embedding_dim: int
	) -> None:
		super(TorchEmbeddings, self).__init__()
		self.vocab = vocab
		self.embedding_dim = embedding_dim
		self.undelying_embedding = nn.Embedding(
			num_embeddings=len(vocab),
			embedding_dim=embedding_dim
		)
	
	async def aembed_single(self, text: str, *args, **kwargs) -> Awaitable[torch.Tensor]:
		raise NotImplementedError("Torch embeddings implements `embed` function directly")

	async def aembed(self, texts: Iterable[str], *args, **kwargs) -> torch.Tensor:
		raise NotImplementedError("Torch embeddings implements `embed` function directly")
	
	def embed(self, texts: Iterable[str]) -> torch.Tensor:
		pass	# use `forward()`	!!!
	
	@property
	def _embedding_dim(self) -> int:
		return self.embedding_dim