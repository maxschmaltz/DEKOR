import os
from abc import abstractmethod
import gc
import re
import torch
from datasets import Dataset
from transformers import (
	PreTrainedTokenizer,
	PreTrainedModel,
	TrainingArguments, 
	Trainer,
	EvalPrediction
)
import pickle
from typing import List, Dict, Union, Optional, Iterable

from dekor.splitters.base import BaseSplitter
from dekor.utils.gecodb_parser import Compound


class BaseHFSplitter(BaseSplitter):

	cache_path = ".cache/"
	logs_path = ".logs/"
	checkpoints_path = ".checkpoints/"

	def __init__(
		self,
		*,
		learning_rate: Optional[float]=0.001,
		n_epochs: Optional[int]=10,
		batch_size: Optional[int]=16,
		verbose: Optional[bool]=True
	) -> None:
		self.learning_rate = learning_rate
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.verbose = verbose

	@abstractmethod
	def _build_tokenizer(self, path: str, *args, **kwargs) -> None:
		self.tokenizer: PreTrainedTokenizer
		pass

	@abstractmethod
	def _build_llm(self, path: str, *args, **kwargs) -> None:
		self.llm: PreTrainedModel
		pass

	@abstractmethod
	def _tokenize(
		self,
		observations: Union[List[str], Dict[str, List[str]]],
		*args,
		**kwargs
	) -> Union[List[torch.Tensor], Dict[str, List[torch.Tensor]]]:
		pass

	@abstractmethod
	def _compute_eval_metrics(self, eval_pred: EvalPrediction, *args, **kwargs) -> Dict[str, float]:
		pass

	def _get_training_args(self, dev_available: bool) -> dict:
		return {
			"output_dir": self.checkpoints_path,
			"learning_rate": self.learning_rate,
			"num_train_epochs": self.n_epochs,
			"eval_strategy": "epoch" if dev_available else "no",
			"load_best_model_at_end": dev_available,	# will only work if there is dev
			"include_inputs_for_metrics": True,	# pass inputs for compute metrics
			# NOTE: add the two fields below in subclasses!!!
			# "metric_for_best_model": ...,	# from `_compute_eval_metrics()`
			# "greater_is_better": ...,
			"per_device_train_batch_size": self.batch_size,
			"per_device_eval_batch_size": self.batch_size,
			"auto_find_batch_size": False,
			"group_by_length": True,
			"logging_dir": self.logs_path,
			"logging_strategy": "epoch",
			"save_strategy": "epoch",	# save every epoch
			"save_total_limit": 3,	# saves 3 last epochs
			"save_only_model": True,	# don't save optimizer etc.
			# "optim": ...	# only Adams of different types are supported
		}
	
	def _train(
		self,
		train_dataset_tokenized: Dataset,
		dev_dataset_tokenized: Optional[Dataset]=None,
		**kwargs
	) -> None:

		torch.cuda.empty_cache()
		gc.collect()

		training_args = TrainingArguments(
			**self._get_training_args(dev_available=dev_dataset_tokenized is not None)
		)

		trainer = Trainer(
			self.llm,
			training_args,
			tokenizer=self.tokenizer,
			train_dataset=train_dataset_tokenized,
			eval_dataset=dev_dataset_tokenized,
			compute_metrics=self._compute_eval_metrics,
			**kwargs
		)

		trainer.train()
		self.trainer = trainer

		# remove checkpoints to save space; we are not training in several passes and will not
		# continue finetuning from checkpoint; the best model is saved anyways
		for _, dirnames, _ in os.walk(self.checkpoints_path):
			for dirname in dirnames:
				if dirname.startswith("checkpoint-"):
					full_dirname = os.path.join(self.checkpoints_path, dirname)
					for filename in os.listdir(full_dirname):
						os.remove(os.path.join(full_dirname, filename))
					os.rmdir(full_dirname)

	def fit(
		self,
		*,
		train_compounds: Optional[Iterable[Compound]]=None,
		dev_compounds: Optional[Iterable[Compound]]=None,
		test: Optional[bool]=False
	):
		# we want to have different models depending on their parameters so we
		# replace the path here
		path = re.sub(r"\/$", "", self.path)
		s = lambda md: md if not isinstance(md, dict) else '_'.join([
			f"{''.join([p[:3].capitalize() for p in param.split('_')])}-{s(value)}"
			for param, value in md.items()
		])
		path += '_' + s(self._metadata)
		path += f"_{0 if train_compounds is None else len(train_compounds)}"
		path += f"_{0 if dev_compounds is None else len(dev_compounds)}"
		self.path = path + "/"
		# with transformers we need to save models necessarily to load them afterwards
		# so we will keep track on whether it's test or not in order to
		# write the model in a temp directory and then remove if it is
		self._test = test
		if test:
			self.path = re.sub(r"\/$", "_tmptest/", self.path)
		super().fit(
			train_compounds=train_compounds,
			dev_compounds=dev_compounds,
			test=test
		)	# will call `save()`
		# del self._test
		return self
	
	def save(self) -> None:
		self.trainer.save_model(self.path)
		# now load the resulting model
		self._build_llm(self.path)
		if not self._test:
			self.tokenizer.save_pretrained(self.path)
			with open(os.path.join(self.path, "vocab_links.pkl"), "wb") as f:
				pickle.dump(self.vocab_links, f)
		else:
			# remove tmp after loading the model
			for filename in os.listdir(self.path):
				os.remove(os.path.join(self.path, filename))
			os.rmdir(self.path)

	def load(self) -> None:
		with open(os.path.join(self.path, "vocab_links.pkl"), "rb") as f:
			self.vocab_links = pickle.load(f)
		self._build_tokenizer(self.path)
		self._build_llm(self.path)