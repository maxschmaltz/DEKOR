import torch
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings as OriginalFlairEmbeddings
from typing import Awaitable

from dekor.embeddings.base import BaseEmbeddings
from dekor.splitters.base import DEVICE


class FlairEmbeddings(BaseEmbeddings):

	name = "flair"
	requires_vocab = False
	trainable = False

	def __init__(self) -> None:
		# `FlairEmbeddings` fit better when the meaning of words plays a role;
		# for short "meaningless" sequences where the character-level information is needed,
		# `CharacterEmbeddings` fit better. However, `CharacterEmbeddings` turns out
		# to be just a wrapper over `nn.Embeddings` + `nn.LSTM` which makes no sense
		# to use because it just pretty much repeats our own implementation, just with LSTM instead of RNN.
		# `FlairEmbeddings` are though intended for larger chunks, still might be useful
		# because they are 1) pretrained but can be fine-tuned; 2) they are (citing)
		# "trained without any explicit notion of words and ... model words as sequences of characters" and
		# "contextualized by their surrounding text".
		# Even though 1), we will still use the embeddings frozen because fine-tuning
		# requires operations that diverge from our goal and just require a lot of
		# additional work which will probably won't pay off.
		self.underlying_embeddings = OriginalFlairEmbeddings(
			# using only forward: in German, compound links depend only on left constituent
			# so only left to right order matters
			"de-forward", 
			with_whitespace=False,
			tokenized_lm=False,
			has_decoder=False,	# only for generation
			is_lower=True	# lowercase
		)

	def train(self) -> None:
		pass    # are static

	def eval(self) -> None:
		pass

	async def aembed_single(self, text: str) -> Awaitable[torch.Tensor]:
		# if an empty link candidate is passed, `Sentence` cannot handle it;
		# we will heuristically substitute it with a marker which will not affect
		# the result because empty candidates will consistently be substituted
		# by the same marker, hence, have the same embeddings
		text = text or "!"
		sentence = Sentence(text, language_code="de")
		self.underlying_embeddings.embed(sentence)
		# is pushed to CUDA automatically but for Mac's mps, need to enforce
		embedding = sentence[0].embedding.to(DEVICE)
		return embedding

	@property
	def _embedding_dim(self) -> int:
		return self.underlying_embeddings.embedding_length