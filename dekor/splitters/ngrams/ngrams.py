"""
Ngram model for splitting German compounds based on the DECOW16 compound data.
"""

import re
import numpy as np
from tqdm import tqdm
import pickle
from typing import Iterable, Optional, List

from dekor.splitters.base import BaseSplitter
from dekor.utils.gecodb_parser import Compound
from dekor.utils.vocabs import StringVocab


class NGramsSplitter(BaseSplitter):

	"""
	N-grams-based compound splitter that relies on the COW dataset format.
	First, fits train COW entries, then predicts lemma splits in this format.

	Parameters
	----------
	n : `int`, optional, defaults to `2`
		maximum n-grams length

	record_none_links : `bool`, optional, defaults to `False`
		whether to record contexts between which no links occur;
		hint: that could lead to a strong bias towards no link choice

	verbose : `bool`, optional, defaults to `True`
		whether to show progress bar when fitting and predicting compounds
	"""

	name = "ngrams"
	path = ".pretrained/ngrams.pkl"

	def __init__(
		self,
		*,
		n: Optional[int]=3,
		record_none_links: bool,
		verbose: Optional[bool]=True
	) -> None:
		self.n = n
		self.record_none_links = record_none_links
		self.verbose = verbose
		self.vocab_positions = StringVocab()
		if self.record_none_links:
			self.path = re.sub("\.pkl$", "_nl.pkl", self.path)

	def _metadata(self) -> dict:
		return {
			"n": self.n,
			**super()._metadata()
		}

	def _fit(
		self,
		train_compounds: Optional[Iterable[Compound]]=None
	) -> None:

		"""
		Feed DECOW16 compounds to the model. That includes iterating through each compound
		with a sliding window and counting occurrences of links between n-gram contexts
		to try to fit to the target distribution.

		Parameters
		----------
		train_compounds : `Iterable[Compound]`
			collection of `Compound` objects out of COW dataset to fit

		Returns
		-------
		`NGramsSplitter`
			fit model
		"""

		all_masks = []
		progress_bar = tqdm(train_compounds, desc="Fitting") if self.verbose else train_compounds
		for compound in progress_bar:
			# collect masks from a single compound
			pairs = []
			for masks in self._get_positions(compound, self.n):
				for (left, right, mid), link in masks:
					# A mask has a form (c_l, c_r, c_m, l), where
					#   * c_l is the left n-gram
					#   * c_r is the right n-gram
					#   * c_m is the mid n-gram
					#   * l is the link id (unknown link id if none)
					# We want to get a mapping from <c_l, c_r, c_m> of contexts C = (c1, c2, ...)
					# to distribution of link ids L = (l1, l2, ...).
					# If we do that with matrices, they become too large to operate over because they become extremely sparse
					# and take up the whole memory (they are also 4D matrices which makes it worse);
					# that is why we need a more compact way to encode positions.
					# What we can do is to encode positions <c_l, c_r, c_m> separately as strings and thus reduce
					# the 4D matrix of shape (|C|, |C|, |C|, |L|) to a 2D matrix of shape (|P|, |L|),
					# where P is the set of encoded positions. Moreover, we will this significantly
					# reduce the sparseness of the matrix because there will be no unknowns contexts
					# so no zeros between n-grams, only zeros on L distribution when a link never occurs in a position.
					# For that, we join the contexts into a single position.
					link_id = self.vocab_links.add(link.component)
					position = '|'.join([left, right, mid])
					position_id = self.vocab_positions.add(position)
					pairs.append((position_id, link_id))
			all_masks += pairs
		all_masks = np.array(all_masks, dtype=np.int32)

		# count unique position --> link id mapping entries
		_, unique_indices, counts = np.unique(all_masks, return_index=True, return_counts=True, axis=0)
		# `np.unique()` sorts masks so the original order is not preserved;
		# to recreate it, the method returns indices of masks in the original dataset
		# in order of their counts; for example, `unique_indices[5]` contains
		# the index of that mask in `masks_links` whose count is under `counts[5]`
		unique_masks_links = all_masks[unique_indices]
		positions_ids = unique_masks_links[:, 0]
		link_ids = unique_masks_links[:, 1]
		# heuristically force <unk> link from <unk> context
		positions_ids = np.append(positions_ids, self.vocab_positions.unk_id)
		link_ids = np.append(link_ids, self.vocab_links.unk_id)
		counts = np.append(counts, 1)

		n_pos, n_links = len(self.vocab_positions), len(self.vocab_links)
		counts_links = np.zeros((n_pos, n_links), dtype=np.float32)  #f or further division
		counts_links[positions_ids, link_ids] = counts

		# counts to frequencies; since all positions point at least to <unk>,
		# no zero division will be encountered
		counts_links /= counts_links.sum(axis=1, dtype=np.float32).reshape(
			(n_pos, 1)
		)
		# rewrite <unk> link from <unk> context close to 0 so that
		# if there is an actual data available, it is preferred
		counts_links[self.vocab_positions.unk_id, self.vocab_links.unk_id] = 1e-10
		self.freqs_links = counts_links
	
	def predict(self, lemmas: Iterable[str]) -> List[Compound]:

		"""
		Make prediction from lemmas to DECOW16-format `Compound`s

		Parameters
		----------
		lemmas : `Iterable[str]`
			lemmas to predict

		Returns
		-------
		`List[Compound]`
			preds in DECOW16 compound format
		"""

		# to gather positions where there are no links, we force set `record_none_links`
		# as there are no positions otherwise because no links in test;
		# it will not affect training as it's already passed as well as on predictions
		# because prediction depends on the weights model learned
		record_none_links_orig = bool(self.record_none_links)   # copy
		self.record_none_links = True

		preds = []
		progress_bar = tqdm(lemmas, desc="Predicting") if self.verbose else lemmas
		# note: async run is not efficient as one iteration is too fast
		for lemma in progress_bar:
			logits = []
			# we will map final positions in the prob matrix
			# to link indices to easily restore raw compound
			idx = 0 # mask index
			link_candidates = {}
			compound = Compound(lemma) # since there are no link in lemma, a single stem will be there
			# collect masks from a single compound
			for i, mask in enumerate(self._get_positions(compound, self.n)):
				for (left, right, mid), _ in mask:  # link is to be predicted
					position = '|'.join([left, right, mid])
					position_id = self.vocab_positions.encode(position)
					probs = self.freqs_links[position_id]
					logits.append(probs)
					# start link index and link representation
					link_candidates[idx] = (i + 1, mid); idx += 1   # i + 1: correction for BOS
			logits = np.stack(logits)
			pred = self._predict(lemma, logits, link_candidates)
			preds.append(pred)

		# return `record_none_links_orig`
		self.record_none_links = record_none_links_orig

		return preds
	
	def save(self) -> None:
		state_dict = {
			"freqs_links": self.freqs_links,
			"vocab_links": self.vocab_links
		}
		with open(self.path, "wb") as f:
			pickle.dump(state_dict, f)

	def load(self) -> None:
		with open(self.path, "rb") as f:
			state_dict = pickle.load(f)
		self.freqs_links = state_dict["freqs_links"]
		self.vocab_links = state_dict["vocab_links"]