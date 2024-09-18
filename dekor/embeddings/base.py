from abc import ABC, abstractmethod
import asyncio
import torch
from typing import Iterable, Awaitable, Any


class BaseEmbeddings(ABC):

	name: str
	underlying_embeddings: Any
	requires_vocab: bool
	trainable: bool
	_embedding_dim: int

	@abstractmethod
	def train(self) -> None:
		pass

	@abstractmethod
	def eval(self) -> None:
		pass

	@abstractmethod
	async def aembed_single(self, text: str, *args, **kwargs) -> Awaitable[torch.Tensor]:
		pass

	async def aembed(self, texts: Iterable[str], *args, **kwargs) -> torch.Tensor:
		tasks = [self.aembed_single(text, *args, **kwargs) for text in texts]
		embeddings = await asyncio.gather(*tasks)
		embeddings = torch.stack(embeddings, dim=0)
		return embeddings

	def embed(self, texts: Iterable[str], *args, **kwargs) -> torch.Tensor:
		embeddings = asyncio.run(self.aembed(texts, *args, **kwargs))
		return embeddings	# b x emb