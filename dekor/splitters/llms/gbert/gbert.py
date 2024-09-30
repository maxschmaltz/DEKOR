import re
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
	BertTokenizer,
	BertForSequenceClassification,
	EvalPrediction
)
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from typing import Optional, Iterable, Dict, List

from dekor.splitters.base import DEVICE
from dekor.splitters.llms.base import BaseHFSplitter
from dekor.utils.gecodb_parser import Compound

BASE_MODEL_NAME = "deepset/gbert-base"


class GBERTSplitter(BaseHFSplitter):

	name = "gbert"
	path = ".pretrained/llms/gbert/"

	def __init__(
		self,
		*,
		context_window: Optional[int]=3,
		record_none_links: bool,
		learning_rate: Optional[float]=0.001,
		n_epochs: Optional[int]=3,
		batch_size: Optional[int]=16,
		verbose: Optional[bool]=True
	):
		super().__init__(
			learning_rate=learning_rate,
			n_epochs=n_epochs,
			batch_size=batch_size,
			verbose=verbose
		)
		self.context_window = context_window
		self.record_none_links = record_none_links

	def _metadata(self) -> dict:
		return {
			"context_window": self.context_window,
			**super()._metadata(),
			"learning_rate": self.learning_rate,
			"n_epochs": self.n_epochs
		}
	
	def _build_tokenizer(self, path: str) -> None:
		self.tokenizer = BertTokenizer.from_pretrained(
			path,
			padding_side="right",
			truncation_side="left"
		)

	def _build_llm(self, path: str) -> None:
		self.llm = BertForSequenceClassification.from_pretrained(
			path,
			num_labels=len(self.vocab_links),
			problem_type="multi_label_classification",
			cache_dir=self.cache_path
		)

	def _tokenize(self, examples: List[str]) -> List[torch.Tensor]:
		return self.tokenizer(
			examples,
			padding=True,
			truncation=False,
			return_tensors="pt",
			return_token_type_ids=True,
			return_attention_mask=True,
			# return_special_tokens_mask=True,	# it is important where the separator is (only in processing)
			verbose=self.verbose
		)
	
	def _compute_eval_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
		# Since we cannot have the pure compound macro accuracy
		# here (it requires the whole compounds), we will simulate it here;
		# since all compounds necessarily have 1 link, all of the links
		# that need to be predicted will be here at some arbitrary positions,
		# and they will have non-zero link ids. In the final prediction from logits,
		# we ignore zero link probabilities, which we will replicate here.
		# So the approach here is to compare all non-zero gold links
		# with the corresponding predictions and calculate accuracy only over them.
		# Thus, we will monitor if the actually relevant links are predicted,
		# which will simulate whether the compounds were predicted correctly.
		golds = np.argmax(eval_pred.label_ids, axis=1)
		preds = np.argmax(eval_pred.predictions, axis=1)
		# non-zero links
		non_zero_indices, = np.where(golds != self.vocab_links.unk_id)
		non_zero_golds = golds[non_zero_indices]
		non_zero_preds = preds[non_zero_indices]
		accuracy = accuracy_score(non_zero_golds, non_zero_preds)
		return {
			"label_accuracy": accuracy
		}
	
	def _get_training_args(self, dev_available: bool) -> Dict:
		training_args = super()._get_training_args(dev_available)
		training_args["metric_for_best_model"] = "label_accuracy",	# from `_compute_eval_metrics()`
		training_args["greater_is_better"] = True
		return training_args

	def _fit(
		self,
		train_compounds: Iterable[Compound],
		dev_compounds: Optional[Iterable[Compound]]=None
	) -> None:

		self._build_tokenizer(BASE_MODEL_NAME)

		train_triplets = []
		train_link_ids = []
		progress_bar = tqdm(train_compounds, desc="Preprocessing train") if self.verbose else train_compounds
		for compound in progress_bar:
			# collect masks from a single compound
			triplets, link_ids = [], []
			for masks in self._get_positions(compound, self.context_window):
				for (left, right, mid), link in masks:
					# A mask has a form (c_l, c_r, c_m, l), where
					#   * c_l is the left n-gram
					#   * c_r is the right n-gram
					#   * c_m is the mid n-gram
					#   * l is the link id (unknown link id if none)
					# BERT makes an efficient use of the [SEP] token
					# to separate sequences so in contrary to
					# NN-based models where we had to get embeddings
					# for each triplet part separately because we risked
					# to blur the boundary between them otherwise,
					# here we can be sure [SEP] will be enough 
					link_id = self.vocab_links.add(link.component)
					# gather
					triplets.append(f"{left}[SEP]{right}[SEP]{mid}")
					link_ids.append(link_id)
			train_triplets += triplets
			train_link_ids += link_ids

		if dev_compounds is not None:

			dev_triplets = []
			dev_link_ids = []
			progress_bar = tqdm(dev_compounds, desc="Preprocessing dev") if self.verbose else dev_compounds
			for compound in progress_bar:
				# collect masks from a single compound
				triplets, link_ids = [], []
				for masks in self._get_positions(compound, self.context_window):
					for (left, right, mid), link in masks:
						link_id = self.vocab_links.encode(link.component)
						# gather
						triplets.append(f"{left}[SEP]{right}[SEP]{mid}")
						link_ids.append(link_id)
				dev_triplets += triplets
				dev_link_ids += link_ids

		self._build_llm(BASE_MODEL_NAME)

		# make target the same shape as the input for different losses; treat
		# each row as a holistic distribution with only correct link probability equal to 1;
		# that is needed for the BCE loss;
		# for example, if there are 10 links and the correct link is 3,
		# then instead of [3] the target will be [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
		Y_train = torch.zeros(len(train_link_ids), len(self.vocab_links), dtype=torch.float16, device=DEVICE)
		Y_train[range(len(train_link_ids)), train_link_ids] = 1
		train_dataset = Dataset.from_dict({
			"triplets": train_triplets,
			"labels": Y_train
		})

		train_dataset_tokenized = train_dataset.map(
			self._tokenize,
			input_columns="triplets",
			batched=True,
			drop_last_batch=False
		)
		train_dataset_tokenized.set_format(type="torch")

		if dev_compounds is not None:

			Y_dev = torch.zeros(len(dev_link_ids), len(self.vocab_links), dtype=torch.float16, device=DEVICE)
			Y_dev[range(len(dev_link_ids)), dev_link_ids] = 1
			dev_dataset = Dataset.from_dict({
				"triplets": dev_triplets,
				"labels": Y_dev
			})

			dev_dataset_tokenized = dev_dataset.map(
				self._tokenize,
				input_columns="triplets",
				batched=True,
				drop_last_batch=False
			)
			dev_dataset_tokenized.set_format(type="torch")

		else: dev_dataset_tokenized = None

		self._train(
			train_dataset_tokenized=train_dataset_tokenized,
			dev_dataset_tokenized=dev_dataset_tokenized
		)

	def predict(self, lemmas: List[str]) -> List[Compound]:
		
		# to gather positions where there are no links, we force set `record_none_links`
		# as there are no positions otherwise because no links in test;
		# it will not affect training as it's already passed as well as on predictions
		# because prediction depends on the weights model learned
		record_none_links_orig = bool(self.record_none_links)   # copy
		self.record_none_links = True

		test_triplets = []
		m = 0
		all_link_candidates = OrderedDict()
		progress_bar = tqdm(lemmas, desc="Preprocessing test") if self.verbose else lemmas
		for lemma in progress_bar:
			all_link_candidates[m] = {}
			idx = 0 # masks inside a single lemma
			compound = Compound(lemma) # since there are no link in lemma, a single stem will be there
			# collect masks from a single compound
			triplets = []
			for i, masks in enumerate(self._get_positions(compound, self.context_window)):
				# for each mask, we will store corresponding start index and link realization,
				# so that we can all start indices and realizations alongside with their
				# respective probs to make the final prediction just like in the Ngrams model;
				# `all_link_candidates` will store both boundaries of a single lemma as well as
				# indices and realizations inside of those boundaries
				for (left, right, mid), _ in masks:
					# probs = ...   # ngram in-place implementation
					triplets.append(f"{left}[SEP]{right}[SEP]{mid}")
					all_link_candidates[m][idx] = (i + 1, mid); idx += 1   # i + 1: correction for BOS
			test_triplets += triplets    # this will go to prediction inside of a batch
			# prepare for the next lemma
			m += len(triplets)
		all_link_candidates[m] = {} # "close the bracket"
		lc_keys = list(all_link_candidates.keys())

		# eval mode
		self.llm.eval()

		# return `record_none_links_orig`
		self.record_none_links = record_none_links_orig

		test_dataset = Dataset.from_dict({
			"triplets": test_triplets
		})

		test_dataset_tokenized = test_dataset.map(
			self._tokenize,
			input_columns="triplets",
			batched=True,
			drop_last_batch=False,
			# as opposed to using Trainer, here we have to remove the excessive columns manually
			remove_columns=["triplets"]
		)
		test_dataset_tokenized.set_format(type="torch")

		test_dataloader = DataLoader(
			# will only output batches of x's
			test_dataset_tokenized,
			batch_size=self.batch_size,
			drop_last=False # we cannot drop last because then not all the lemmas will be predicted
		)

		all_logits = []

		n_steps = len(test_dataloader)
		if self.verbose: progress_bar = tqdm(total=n_steps, desc="Predicting")
		with torch.no_grad():
			for batch in test_dataloader:
				if self.verbose: progress_bar.update()
				logits = self.llm(**batch).logits
				all_logits.append(logits)

		all_logits = torch.concat(all_logits, dim=0).detach().numpy()

		preds = []
		progress_bar = tqdm(lemmas, desc="Postprocessing") if self.verbose else lemmas
		for i, lemma in enumerate(progress_bar):
			start, end = lc_keys[i], lc_keys[i + 1]
			# get logits for the lemma
			logits = all_logits[start: end]
			link_candidates = all_link_candidates[start]
			pred = self._predict(lemma, logits, link_candidates)
			preds.append(pred)

		return preds