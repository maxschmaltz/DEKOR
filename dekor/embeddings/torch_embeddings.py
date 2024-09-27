import torch
import torch.nn as nn
from typing import Awaitable

from dekor.splitters.base import DEVICE
from dekor.embeddings.base import BaseEmbeddings
from dekor.utils.vocabs import StringVocab


class TorchEmbeddings(nn.Module, BaseEmbeddings):

	name = "torch"
	requires_vocab = True
	trainable = True

	def __init__(
		self,
		*,
		vocab: StringVocab,
		embedding_dim: int
	) -> None:
		super(TorchEmbeddings, self).__init__()
		self.vocab = vocab
		self.embedding_dim = embedding_dim
		self.underlying_embeddings = nn.Embedding(
			num_embeddings=len(vocab),
			embedding_dim=embedding_dim,
			device=DEVICE
		)

	# `train()` and `eval()` are implemented in `nn.Module`
	
	async def aembed_single(self, text: str) -> Awaitable[torch.Tensor]:
		# Even though in torch, we can operate over whole tensors fast and efficiently,
		# that requires that we have the same length of texts every time, hence, padding.
		# The texts are already too short and it might be too noisy to add padding;
		# That is why we will better embed each text separately and just ensure the same
		# shape of the output.
		# Our implementation here is character-wise; to get the most significant
		# embedding values from different characters, we use max pooling;
		# in cases like RNN it won't affect anything because RNN input
		# is already character-wise. This will also ensure the same output shape
		# no matter what the input text length is.
		text = text or ['']	# '' is also added manually
		char_ids = [self.vocab.encode(char) for char in text]	# l
		char_ids = torch.tensor(char_ids, dtype=torch.long, device=DEVICE)	# l
		char_embeddings = self.underlying_embeddings(char_ids)	# l x emb
		text_embedding, _ = torch.max(char_embeddings, dim=0)	# emb
		return text_embedding
	
	@property
	def _embedding_dim(self) -> int:
		return self.embedding_dim