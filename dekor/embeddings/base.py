from abc import ABC, abstractmethod
import asyncio
import torch
from typing import Iterable, Awaitable, Optional

from dekor.utils.vocabs import StringVocab


class BaseEmbeddings(ABC):

	name: str
	vocab: Optional[StringVocab]=None

	@abstractmethod
	async def aembed_single(self, text: str, *args, **kwargs) -> Awaitable[torch.Tensor]:
		pass

	async def aembed(self, texts: Iterable[str], *args, **kwargs) -> torch.Tensor:
		tasks = [self.aembed_single(text, *args, **kwargs) for text in texts]
		embeddings = await asyncio.gather(*tasks)
		return embeddings

	def embed(self, texts: Iterable[str], *args, **kwargs) -> torch.Tensor:
		embeddings = asyncio.run(self.aembed(texts, *args, **kwargs))
		return embeddings
	
	@abstractmethod
	@property
	def _embedding_dim(self) -> int:
		pass