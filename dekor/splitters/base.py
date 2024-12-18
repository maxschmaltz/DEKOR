"""
Base model for splitting German compounds based on the DECOW16 compound data.
"""

import os
import json
import platform
import re
from abc import ABC, abstractmethod
import numpy as np
import torch
import warnings
from typing import Iterable, Iterator, Tuple, List, Dict, Optional	# Self

from dekor.utils.gecodb_parser import Compound, Link, UMLAUTS
from dekor.utils.vocabs import StringVocab, UNK

# in Mac, M1 Chip is a GPU but it shares memory with CPU
# which is why `torch.cuda.is_available()` won't recognize it
# and it requires another device code;
# "Darwin" stands for MacOS
DEVICE = "mps" if platform.system() == "Darwin" else "cuda" if torch.cuda.is_available() else "cpu"


class BaseSplitter(ABC):

	"""
	Base class for splitters.
	"""

	# caching preprocessed data is impossible because it gets shuffled on every iteration

	name: str
	path: str
	record_none_links: bool # enforce
	
	vocab_links = StringVocab()
	_elink = Link(  # for analyzing positions
		UNK,
		span=(-1, -1),
		type=UNK
	)

	@property
	def _metadata(self) -> dict:
		# for parameter tracking
		return {
			"record_none_links": self.record_none_links
		}

	def _get_positions(
		self,
		compound: Compound,
		context_window: int
	) -> Iterator[Tuple[Tuple[str], Compound]]:

		# Analyze a single compound; performed as a sliding window
		# with a sliding window inside over the compound lemma, where for each position it is stored,
		# which left and right context in N-grams there is and what is in between and
		# whether that "in between" is a link and, if yes, which one.
		# Example:
		#   "bundestag" with 2-grams
		#   ">b", "un", "" --> no link
		#   ">b", "nd", "u" --> no link
		#   ">b", "de", "un" --> no link
		#   ...
		#   "nd", "es", "" --> no link
		#   "nd", "st", "e" --> no link
		#   "nd", "ta", "es" --> link "_+s_"
		#   ...
		# Later, uniques context-link triples will be used in prediction.

		lemma = f'>{compound.lemma}<'    # BOS and EOS
		n = context_window  # shorthand
		l = len(lemma) - 1  # -1 because indexing starts at 0

		# as we know which links to expect, we will track them 
		next_link_idx = 0
		# Masks will be of a form (c_l, c_r, c_m, l), where
		#   * c_l is the left N-gram
		#   * c_r is the right N-gram
		#   * c_m is the middle N-gram
		#   * l is the link id (unknown id if none)
		# Then this masks can be used differently dependent on the splitter

		# Make sliding window; however, we want to start not directly with
		# N-grams, but first come from 1-grams to N-grams at the left of the compound
		# and then slide by N-grams; same with the end: not the last N-gram,
		# but N-gram to 1-gram. To be more clear: having 'Bundestag' and 3-grams, we don't want contexts
		# to be directly (("bun", "des"), ("und", "est"), ..., ("des", "tag")), 
		# but we rather want (("b", "und"), ("bu", "nde"), ("bun", "des"), ..., ("des", "tag"), ("est", "ag"), ("sta", "g")).
		# As we process compounds unidirectionally and move left to right,
		# we want subtract max N-gram length to achieve this effect; thus, with a window of length
		# max N-gram length, we will begin with 1-grams, ..., reach N-grams, ..., and end with ..., 1-grams
		for i in range(1 - n + 1, l - n):  # 1 from both sides because there can not be a link right after BOS
			masks = []
			# next expected link; we use empty link in case there are no links anymore to unify the workflow below
			next_link = compound.links[next_link_idx] if next_link_idx < len(compound.links) else self._elink
			s = max(0, i)   # start of left
			m = i + n   # end of left = start of mid
			# break cycle if case left forces the right to be the single EOS
			# which it makes no sense to record because link can not appear there or any further
			if m > l - 1: break # -1 for special symbols
			for r in range(4):  # max length of a link representation is 3 as in -ens-
				e = m + r   # end of mid = start of right
				f = m + r + n   # end of right
				# break cycle if case right context is going to be the single EOS
				if e > l - 1: break   # -1 for special symbols
				left = lemma[s: m]
				mid = lemma[m: e]
				right = lemma[e: f]
				# define if there is a link incoming at this index;
				# because of such implementation, mid will always be the same
				# with the link realization
				if (m - 1, e - 1) == next_link.span:    # -1 is correction because of special symbols
					link = next_link
					# increment
					next_link_idx += 1
				else:
					# the idea behind recording none links is that it will give us
					# the proportion between a link and no link on a certain position;
					# if not recorded, 5 occurrences of link A will be the same
					# as 25 occurrences of link B in different positions (if no other
					# links have been encountered), but with none links recorded,
					# 995-5 will be less probable than 975-25
					if self.record_none_links:
						link = self._elink
					else: continue
				masks.append(((left, right, mid), link))
			# This will be a generator because all of the splitters will have
			# an additional processing of masks so generating will prevent double iteration
			# over the same list; moreover, this is needed in prediction for some splitters
			# to be able to "meddle" in the middle of processing.
			# Also, we will yield masks in packs, because in further processing we will
			# need to define the most probable link for a single position
			# and not all the windows from this one position;
			# for example, in "bundestag" we want to predict not 4 windows separately as in
			# ("und", "", "est"), ("und", "e", "sta"), ("und", "es", "tag") ...
			# but rather whether there is a link after "und" given all windows.
			yield masks

	@abstractmethod
	def _fit(
		self,
		train_compounds: Iterable[Compound]=None,
		dev_compounds: Optional[Iterable[Compound]]=None,
		**kwargs
	) -> None:
		pass

	def fit(
		self,
		*,
		train_compounds: Optional[Iterable[Compound]]=None,
		dev_compounds: Optional[Iterable[Compound]]=None,
		test: Optional[bool]=False,
		**kwargs
	):	# -> Self:	# won't work in python3.10 or older
		
		"""
		Feed DECOW16 compounds to the model. That includes iterating through each compound
		with a sliding window, collecting and encoding occurrences of links between N-gram contexts
		and training the backbone NN on them to try to fit to the target distribution.

		Parameters
		----------
		train_compounds : `Iterable[Compound]`
			collection of `Compound` objects out of COW dataset to train on

		dev_compounds : `Iterable[Compound]`, optional
			collection of `Compound` objects out of COW dataset to validate the model
			on during training / fine-tuning

		test : `bool`, optional, defaults to `False`
			will not load / save the model if `True` (e.g. for benchmarking)

		Returns
		-------
		A subclass of `BaseSplitter`
			fit model
		"""

		# we want to have different models depending on their parameters so we
		# replace the path here
		path, extension = os.path.splitext(self.path)
		s = lambda md: md if not isinstance(md, dict) else '_'.join([
			f"{''.join([p[:3].capitalize() for p in param.split('_')])}-{s(value)}"
			for param, value in md.items()
		])
		path += '_' + s(self._metadata)
		path += f"_{0 if train_compounds is None else len(train_compounds)}"
		path += f"_{0 if dev_compounds is None else len(dev_compounds)}"
		self.path = path + extension
		if os.path.exists(self.path) and not test:
			warnings.warn(f"A pretrained model found. Loading from {self.path}.")
			self.load()
		else:
			assert train_compounds is not None
			if not test:
				warnings.warn(f"No pretrained model found. Training from scratch and saving to {self.path}.")
			self._fit(train_compounds, dev_compounds, **kwargs)
			if not test:
				self.save()
		return self
	
	def _passes_filter(
		self, 
		link_type: str,
		realization: str,
		first_noun: str
	) -> bool:
		
		# using `if` so that no further checks are performed once one has failed
		if (    # use if so that no further checks are performed once one has failed
			# deletion type with addition realization
			(
				link_type == "deletion"
				and (
					len(realization)
					or first_noun[-1] != "e"	# only schwas get deleted
				)
			) or
			# concatenation type with addition realization
			(
				link_type == "concatenation"
				and len(realization)
			) or
			# impossible addition; there might be or not be an e-,
			# depends on whether we eliminate allomorphy (legacy)
			(
				link_type == "addition" and
				# not re.match(f"^e?{realization}$", best_realization)
				not realization in ["s", "es", "n", "en", "e", "er", "ns", "ens"]
			) or
			# umlaut link where there is no rightmost (!) umlaut before
			(
				"umlaut" in link_type and
				(
					not re.search(f"({'|'.join(UMLAUTS.values())})(?!.+({'|'.join(UMLAUTS.keys())}))", first_noun)
					or not realization in ["", "e", "er"]
				)
			)
		):
			return False
		
		return True

	def _predict(
		self,
		lemma: str,
		logits: np.ndarray,
		link_candidates: Dict[int, Tuple[int, str]]
	) -> Compound:

		# For a single lemma, this function gets a list of link candidates,
		# i.e. each (possibly link-) ngram at each lemma index, and
		# respective to the candidates probability distribution
		# over links, so which link (or none) this ngram is with which prob.
		# This info is used to determine the most probable link and its position
		# which will satisfy the grammar constrains.
		# We know for sure that we works with N+N compounds which means
		# we can heuristically restrict all improbable additional
		# links it will predict. Thus, for the compound, we will
		# gather probabilities of all non-none links with respect to their
		# positions and then choose the most probable one.
			
		# at this point, none links have done their job and gave
		# the proportion (if they were even recorded) so
		# we should zero them to easily detect the most probable
		# non-none link
		logits[:, self.vocab_links.unk_id] = 0

		# There is a set of heuristics that have to be filtered out in place
		# in order to get cleaner result; for example, there can not be an umlaut link
		# in case there is no umlaut before it. This can mostly be checked once a link with
		# its representation is parsed; however, it is highly inefficient to do that
		# will all links whose probability is more that 0. Instead, we will treat the
		# probabilities as a stack with probs ordered from highest to lowest;
		# thus, at each iteration, we will check if the current max probable link
		# passes the filter and if yes, we'll break the cycle, if no, zero this prob
		# and take the next highest probable one. Thus, at the end we will output
		# the most probable link of all that passed the filter.  
		while True:

			# it cannot exceed because even if all non-zero links will appear invalid,
			# there will be no probs anymore and the function will exit

			max_prob = logits.max()
			if not max_prob:
				return Compound(lemma)   # no link detected, so return with no links
			
			best_idx, best_link_id = np.where(logits == max_prob)
			best_idx, best_link_id = best_idx[0], best_link_id[0]   # unpack
			i, best_realization = link_candidates[best_idx]
			best_link = self.vocab_links.decode(best_link_id)
			component, realization, link_type = Compound.get_link_info(best_link)

			first_noun = lemma[:i] # left constituent

			# heuristically filter out predictions that cannot be correct
			if not self._passes_filter(link_type, realization, first_noun):
				# zero this impossible link prob
				logits[best_idx, best_link_id] = 0
				continue

			# NOTE: reconstruct from components?
			# If allomorphy is eliminated, we can predict cases like
			# tag_+s_ticket correctly and know that the realization is -es-;
			# however, since we dont reconstruct the compound from components,
			# if we pass tag_+s_ticket to `Compound`, it will think that the
			# realization is -s- even though we know it is not the case.
			# That is why, if eliminated allomorphy encountered,
			# we must reconstruct the link as if allomorphy does not get eliminated,
			# and then `Compound` will still parse the link with elimination
			# but will receive the correct realization we predicted.
			if link_type == "addition_umlaut":
				first_noun = Compound.reverse_umlaut(first_noun)
			elif link_type == "deletion":
				to_delete = Compound.get_deletion(component)
				first_noun += to_delete
			if best_realization != realization:
				# component = re.sub(realization, best_realization, component)
				component = Compound.return_allomorphy(component)

			first_noun = first_noun + component + lemma[i + len(best_realization):]
			pred = Compound(first_noun)
			return pred

	@abstractmethod
	def predict(self, lemmas: List[str], *args, **kwargs) -> List[Compound]:

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

		pass

	@abstractmethod
	def save(self) -> None:

		"""
		Save the model.

		The model will be saved into the path specified in the `path` attribute.
		"""

		pass

	@abstractmethod
	def load(self) -> None:

		"""
		Load the model.

		The model will be loaded from the path specified in the `path` attribute.
		"""

		pass

	def __repr__(self) -> str:
		return f"{self.__class__.__name__}\n{json.dumps(self._metadata, indent=4)}"