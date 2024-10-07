import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
	ByT5Tokenizer,
	T5ForConditionalGeneration,
	EvalPrediction
)
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from typing import Optional, Iterable, Dict, List

from dekor.splitters.base import DEVICE
from dekor.splitters.llms.base import BaseHFSplitter
from dekor.utils.gecodb_parser import Compound

BASE_MODEL_NAME = "google/byt5-base"


class ByT5Splitter(BaseHFSplitter):

	name = "byt5"
	path = ".pretrained/llms/byt5/"

	@property
	def _metadata(self) -> dict:
		return {
			"learning_rate": self.learning_rate,
			"n_epochs": self.n_epochs
		}
	
	def _build_tokenizer(self, path: str) -> None:
		self.tokenizer = ByT5Tokenizer.from_pretrained(
			path,
			padding_side="right",
			truncation_side="right"
		)

	def _build_llm(self, path: str) -> None:
		self.llm = T5ForConditionalGeneration.from_pretrained(
			path,
			cache_dir=self.cache_path,
			device_map=DEVICE
		)
	
	def _tokenize(self, observations: Dict[str, List[str]]) -> torch.Tensor:
		
		lemmas = observations["lemma"]
		lens = [len(lemma) for lemma in lemmas]
		inputs = self.tokenizer(
			lemmas,
			padding=True,
			truncation=True,
			# max_length=max(lens),
			return_tensors="pt",
			return_token_type_ids=False,
			return_attention_mask=True,
			verbose=self.verbose
		)
		output = {
			"input_ids": inputs["input_ids"],
			"attention_mask": inputs["attention_mask"]
		}

		if "raw" in observations:	# fine-tuning
			raws = observations["raw"]
			lens = [len(raw) for raw in raws]
			targets = self.tokenizer(
				raws,
				padding=True,
				truncation=True,
				# max_length=max(lens),
				return_tensors="pt",
				return_token_type_ids=False,
				return_attention_mask=True,
				verbose=self.verbose
			)
			output["labels"] = targets["input_ids"]

		return output

	def _compute_eval_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
		# Even thought dealing with a seq2seq task, the most important thing for us
		# at the end of the day is whether the compound was generated correctly so we
		# stick to the accuracy. Here though, in `eval_pred`, we receive
		# the gold labels (tokenized raws) and logits that contain
		# distribution over tokenizer vocab for each position;
		# if we convert golds ids back to strings, we'll just get the target raws,
		# and after argmax'ing the logits and converting the most probable ids
		# to strings, we'll get the predicted raws. Thus, we can get compound accuracy.
		gold_ids = eval_pred.label_ids
		golds = self.tokenizer.batch_decode(gold_ids, skip_special_tokens=True)	# target raws
		# T5-based models return logits and hidden state
		logits, _ = eval_pred.predictions	# b x i x llm_vocab_size, distribution over vocab
		pred_ids = np.argmax(logits, axis=2)
		preds = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)	# pred raws
		accuracy = accuracy_score(golds, preds)
		return {
			"compound_accuracy": accuracy
		}
	
	def _get_training_args(self, dev_available: bool) -> Dict:
		training_args = super()._get_training_args(dev_available)
		training_args["metric_for_best_model"] = "compound_accuracy"	# from `_compute_eval_metrics()`
		training_args["greater_is_better"] = True
		return training_args

	def _fit(
		self,
		train_compounds: Iterable[Compound],
		dev_compounds: Optional[Iterable[Compound]]=None
	) -> None:

		self._build_tokenizer(BASE_MODEL_NAME)
		self._build_llm(BASE_MODEL_NAME)
		
		# here, we have a seq2seq problem, so lemmas will become our input ids
		# and raw compounds - labels
		train_dataset = Dataset.from_dict({
			"lemma": [compound.lemma for compound in train_compounds],
			"raw": [compound.raw for compound in train_compounds]
		})

		# ByT5 works on raw UTF-8 bytes and can be used without a tokenizer,
		# but for batched inference & training it is recommended using a tokenizer class for padding
		# ("ByT5 simply uses raw bytes utf-8 encoding")
		train_dataset_tokenized = train_dataset.map(
			self._tokenize,
			batched=True,
      		batch_size=self.batch_size * 16,	# to make dataloader compatible
			drop_last_batch=False
		)
		train_dataset_tokenized.set_format(type="torch")

		if dev_compounds is not None:

			dev_dataset = Dataset.from_dict({
				"lemma": [compound.lemma for compound in dev_compounds],
				"raw": [compound.raw for compound in dev_compounds]
			})

			dev_dataset_tokenized = dev_dataset.map(
				self._tokenize,
				batched=True,
        		batch_size=self.batch_size * 16,	# to make dataloader compatible
				drop_last_batch=False
			)
			dev_dataset_tokenized.set_format(type="torch")

		else: dev_dataset_tokenized = None

		# make ByT5 parameter tensors contiguous before training to avoid the error:
		# ```text
		# ValueError: You are trying to save a non contiguous tensor:
		# `encoder.block.0.layer.0.SelfAttention.q.weight` which is not allowed. 
		# It either means you are trying to save tensors which are reference of each other 
		# in which case it's recommended to save only the full tensors, and reslice at load time, 
		# or simply call `.contiguous()` on your tensor to pack it before saving.
		# ```
		for param in self.llm.parameters(): param.data = param.data.contiguous()
		self._train(
			train_dataset_tokenized=train_dataset_tokenized,
			dev_dataset_tokenized=dev_dataset_tokenized
		)

	def _predict(
		self,
		raw: str,
		lemma: str
	) -> Compound:
		
		try:
			pred = Compound(raw)
		except:
			return Compound(lemma)

		# heuristically filter out predictions that cannot be correct
		if (
			#	1. not a single valid link
			len(l := pred.links) != 1
			#	2. not a correct lemma
			or pred.lemma != lemma
			# 	3. the link is impossible
			or not self._passes_filter(l[0].type, l[0].realization, pred.stems[0].component)
		):
			pred = Compound(lemma)
		
		return pred

	def predict(self, lemmas: List[str]) -> List[Compound]:

		# eval mode
		self.llm.eval()

		test_dataset = Dataset.from_dict({
			"lemma": lemmas
		})

		test_dataset_tokenized = test_dataset.map(
			self._tokenize,
			batched=True,
			batch_size=self.batch_size * 16,	# to make dataloader compatible
			drop_last_batch=False,
			# as opposed to using Trainer, here we have to remove the excessive columns manually
			remove_columns=["lemma"]
		)
		test_dataset_tokenized.set_format(type="torch")

		test_dataloader = DataLoader(
			test_dataset_tokenized,
			shuffle=False,
			batch_size=self.batch_size,
			drop_last=False # we cannot drop last because then not all the lemmas will be predicted
		)

		preds = []
		n_steps = len(test_dataloader)
		if self.verbose: progress_bar = tqdm(total=n_steps, desc="Predicting")
		with torch.no_grad():
			for batch in test_dataloader:
				if self.verbose: progress_bar.update()
				input_ids = batch["input_ids"]
				# move batch to CUDA as it is not done manually here as opposed to Trainer
				batch = {
					key: tensor.to(DEVICE)
					for key, tensor in batch.items()
				}
				pred_ids = self.llm.generate(**batch)
				for pred_raw, lemma in zip(
					(self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)),
	  				(self.tokenizer.batch_decode(input_ids, skip_special_tokens=True))
				):
					pred = self._predict(pred_raw, lemma)
					preds.append(pred)

		return preds