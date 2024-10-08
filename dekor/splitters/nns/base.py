"""
Base NN-based model for splitting German compounds based on the DECOW16 compound data.
"""

import os
from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from io import BytesIO
from tqdm import tqdm
from typing import Optional, Iterable, Optional, List, Literal, Tuple, Union

import dekor.embeddings
from dekor.splitters.base import BaseSplitter, DEVICE
from dekor.utils.gecodb_parser import Compound
from dekor.utils.vocabs import StringVocab
from dekor.utils.datasets import XYDataset
from dekor.eval.evaluate import EvaluationResult, evaluate


class BaseNN(nn.Module, ABC):

	"""
	Backbone network for NN-based splitters.
	"""

	def __init__(self, **kwargs) -> None:
		super(BaseNN, self).__init__()
		for param, value in kwargs.items():
			setattr(self, param, value)
		self._build_self()

	@abstractmethod
	def _build_self(self) -> None:
		pass

	@abstractmethod
	def forward(self, input_tensor: torch.Tensor, *args, **kwargs) -> int:
		pass


class BaseRecurrentNN(BaseNN):  # RNN, GRU

	def forward(
		self,
		input_tensor: torch.Tensor,
		hidden_tensor: Optional[torch.Tensor]=None,
		r: Optional[bool]=False,
		force_softmax: Optional[bool]=False
	):
		# in our implementation, we get the accumulative hidden representation from 
		# the parts of the triplet separately and then concatenate them into a single hidden
		# representation, which we use for prediction; since embeddings are outside of the
		# NN, we cannot have both recurrent and dense parts of the process work in one pass
		# so we separate this function onto two modes
		if r:
			# input: b x emb, already embedded, hidden: (D * nl) x h
			# assert isinstance(hidden_tensor, torch.Tensor)
			output, hidden = self.recurrent(input_tensor, hidden_tensor)   # b x 1 x (D * h), (D * nl) x b x h
			return output, hidden
		else:
			# input: b x ((D * h) * 3), concatenated hidden representations of the 3 parts
			output = self.dense(input_tensor)    # b x o
			if self.require_softmax or force_softmax:
				output = self.softmax(output)   # b x o
			return output
	
	@property
	def D(self) -> int:
		self.bidirectional = False	# legacy from when it could be bidirectional
		return 1 if not self.bidirectional else 2

	def init_hidden(self) -> torch.Tensor:
		# return torch.zeros(self.D * self.num_layers, self.batch_size, self.hidden_size)
		# in our implementation, we don't process the whole batch to avoid padding
		# so it is a 2D tensor
		return torch.zeros(self.D * self.num_layers, self.hidden_size, device=DEVICE)


class BaseNNSplitter(BaseSplitter):

	"""
	Base class for NN-based splitters.

	Parameters
	----------

	n : `int`, optional, defaults to `3`
		length of the contexts to encode on the left and on the right from
		target position (which is either a link or significant absence of it)
		for fitting and prediction

	record_none_links : `bool`, optional, defaults to `False`
		whether to record contexts between which no links occur;
		hint: that could lead to a strong bias towards no link choice

	optimizer : `str`, one of ["sgd", "adamw"], optional, defaults to "adamw"
		name of the optimizer for the backbone NN training; `torch.nn.SGD` or `torch.nn.AdamW`
		are used respectively

	criterion : `str`, one of ["crossentropy", "bce", "margin"], optional, defaults to "crossentropy"
		name of the loss function for the backbone NN training; `torch.nn.CrssEntropyLoss`,
		`torch.nn.BCEWithLogitsLoss`, or `torch.nn.MultiLabelSoftMarginLoss` are used respectively

	learning_rate : `float`, optional, defaults to `0.001`
		learning rate for the backbone NN training; must be in an interval (0; 1)

	n_epochs : `int`, optional, defaults to `3`
		number of epochs for the backbone NN to train; must be positive

	save_plot : `bool`, optional, defaults to `False`
		whether to save training plot with losses; if `True`, binary representation
		of the plot in PNG will be stored in the `plot_buffer` attribute

	verbose : `bool`, optional, defaults to `True`
		whether to show progress bar when fitting and predicting compounds

	kwargs:
		parameters to pass to the backbone NN
	"""

	# When doing benchmarking, it is computationally inefficient to combine all the 
	# parameters that we can pass both to the wrapper and the backbone NN 
	# (too much configurations). Correspondingly, we should separate the parameters
	# into groups and run benchmarking inside those groups separately, "freezing"
	# parameters of other groups; thus, we can reassemble intuitively the best parameters
	# group by group.
	# We decided to divide the parameters by their "functionality". The final order is:
	#   1. hyperparameters of the backbone NN;
	#   2. parameters of NN training;
	#   3. wrapper parameters, i.e. parameters of feature retrieval and managements.
	# In ordering the parameters, we were guided by the following logic:
	#   * Whatever the wrapper parameters are, the extracted features will still bear
	#   information about contexts vs links distribution; that is, whatever wrapper parameters
	#   we set, the "competition" of different configurations from the two remaining groups will be fair
	#   because they will compete over the same information whatever this information is.
	#   In other words, if one configuration of parameters wins another on one set of features,
	#   it will probably win over another features, because both these features
	#   describe the same distribution (just with different quality).
	#   So this group should be tested the last.
	#   * From the 2 remaining groups, hyperparameters are the core of the whole model;
	#   if the hyperparameters are chosen poorly, the performance will be bad
	#   no matter which training parameters are picked. However, even with a poor choice of
	#   training parameters the model can capture the patterns.
	#   It therefore makes sense to first test different hyperparameters, because different hyperparameters
	#   will show different performance even with badly matching training parameters and, hence,
	#   will be able to be ranked, but different training parameters will all result in 
	#   low performance if the hyperparameters are chosen erroneously.

	def __init__(
			self,
			*,
			context_window: Optional[int]=3,
			record_none_links: bool,
			embeddings_params: dict,
			nn_params: dict,
			optimizer: Optional[Literal["sgd", "adamw"]]="adamw",
			criterion: Optional[Literal["crossentropy", "bce", "margin"]]="crossentropy",
			learning_rate: Optional[float]=0.001,
			n_epochs: Optional[int]=3,
			target_metric: Optional[str]="f1",
			batch_size: Optional[int]=128,
			save_plot: Optional[bool]=False,
			verbose: Optional[bool]=True
		) -> None:
		self.context_window = context_window
		self.record_none_links = record_none_links
		self.embeddings_params = embeddings_params.copy()
		embeddings_name = self.embeddings_params.pop("name")
		self.embeddings_cls = dekor.embeddings.__all_embeddings__[embeddings_name]
		self.nn_params = nn_params.copy()
		self.optimizer = optimizer
		self.criterion = criterion
		self.learning_rate = learning_rate
		self.n_epochs = n_epochs
		self.target_metric = target_metric
		self.batch_size = batch_size
		self.plot_buffer = BytesIO() if save_plot else None
		self.verbose = verbose
		# vocabs
		if self.embeddings_cls.requires_vocab:
			self.vocab_chars = StringVocab()
			self.vocab_chars.add('')    # manually add empty char
		# self.path = re.sub(r"\.pt$", f"_{embeddings_name}.pt", self.path)	# so embeddings are compatible

	@property
	def _metadata(self) -> dict:
		return {
			"context_window": self.context_window,
			**super()._metadata,
			"embeddings_params": {
				"name": self.embeddings_cls.name,
				**self.embeddings_params
			},
			"nn_params": self.nn_params,
			"optimizer": self.optimizer.__class__.__name__,
			"criterion": self.criterion.__class__.__name__,
			"learning_rate": self.learning_rate,
			"n_epochs": self.n_epochs
		}
	
	# The thing is all the NN-based models work according to the same scenario,
	# that is why this base model mostly implements all the needed methods.
	# It works as follows:
	#   1 `fit()`. Fit compounds:
	#       1.1 `_get_positions()`. For each compound, <left, right, mid> contexts are collected iteratively,
	#       and they get encoded with embeddings; for each context, the link is stored;
	#       1.2 `_build_nn()`. The backbone NN is built;
	#       1.3. `_embed()`. Embeddings become X, encoded links - Y;
	#       1.4. `_train_on_batch()`. The backbone NN trains batch-wise on X, Y
	#   2. `predict()`. Predict lemmas:
	#       2.1 `_get_positions()`. For each lemma, <left, right, mid> contexts are collected iteratively,
	#       and they get encoded with embeddings; no links are stored;
	#       2.2. `_embed()`. Embeddings become X, encoded links - Y;
	#       2.3 `_predict_batch()`. The backbone NN predicts Y on X and yields generations
	#       for all positions for the whole batch
	#       2.4. `_predict()`. For each lemma, only its predictions from the batch are taken
	#       (for all its positions), and the prediction is unscrambled.
	#
	# The fact is, all of those processes are absolutely identical for the NN-based models,
	# so we decided to implement them in one place so here, all algorithmic procedures are covered.
	# Even the training and prediction goes the same way:
	# gather X, (Y) and train / predict batch-wise in the same manner.
	# What differentiates the models is the backbone NNs, and so such this base model
	# allows to concentrate only on their aspects in their respective classes, such as
	# `_embed()` and backbone NN methods.
	# This allows us to create a universal and flexible interface and add NN-based models
	# really quickly by just adjusting a few methods to match expected input and output
	# while maintaining its own workflow in between.
	
	def _build_embeddings(self):
		if self.embeddings_cls.requires_vocab:
			self.embeddings_params["vocab"] = self.vocab_chars
		# "name" is already removed at this point
		self.embeddings = self.embeddings_cls(**self.embeddings_params)
		self.embeddings_params.pop("vocab", None)
	
	@abstractmethod
	def _build_nn(self) -> None:
		self.nn: BaseNN
		pass

	@abstractmethod
	def _embed(self, inputs: Iterable[Union[Tuple[str], str]]) -> torch.Tensor:
		pass
	
	def _train_on_batch(self, triplets: Iterable[Tuple[str]], link_ids: Iterable[int]) -> float:

		self.optimizer.zero_grad()

		# here, we will receive an iterable of observations, 
		# each being a string context triplet and the corresponding link id
		x = self._embed(triplets)   # b x (emb * 3)
		
		# make target the same shape as the input for different losses; treat
		# each row as a holistic distribution with only correct link probability equal to 1;
		# that is needed for the BCE and margin losses;
		# for example, if there are 10 links and the correct link is 3,
		# then instead of [3] the target will be [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
		y = torch.zeros(self.batch_size, len(self.vocab_links), dtype=torch.float16, device=DEVICE)
		y[range(self.batch_size), link_ids] = 1

		output = self.nn(x)
		loss = self.criterion(output, y)
		loss.backward()
		self.optimizer.step()
		return loss.item()
	
	def _validate(self, dev_compounds: Iterable[Compound]) -> EvaluationResult:
		dev_lemmas = [
			compound.lemma for compound in dev_compounds
		]
		preds = self.predict(dev_lemmas)
		eval_res = evaluate(dev_compounds, preds, "any")
		# return train model
		self.nn.train()
		self.embeddings.eval()
		return eval_res

	def _fit(
		self,
		train_compounds: Iterable[Compound],
		dev_compounds: Optional[Iterable[Compound]]=None
	) -> None:

		"""
		Feed DECOW16 compounds to the model. That includes iterating through each compound
		with a sliding window, collecting and encoding occurrences of links between n-gram contexts
		and training the backbone NN on them to try to fit to the target distribution.

		Parameters
		----------
		compounds : `Iterable[Compound]`
			collection of `Compound` objects out of COW dataset to fit

		Returns
		-------
		A subclass of `BaseNNSplitter`
			fit model
		"""

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
					# Since this category of splitters uses NNs as the backbone,
					# we need to embed triplets to be able to efficiently work with them further.
					# Different NN-based splitters will encode different
					# parts of the contexts so no preprocessing is needed here.
					# However, some embeddings will need not strings but
					# ids so for those, we will populate character vocabs.
					link_id = self.vocab_links.add(link.component)
					if self.embeddings_cls.requires_vocab:
						for char in left + right + mid:
							_ = self.vocab_chars.add(char)

					# gather
					triplets.append((left, right, mid))
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
						triplets.append((left, right, mid))
						link_ids.append(link_id)
				dev_triplets += triplets
				dev_link_ids += link_ids

		# init embeddings
		self._build_embeddings()    # at this point, vocab is populated if needed

		# init model, adds .nn attribute
		self._build_nn()

		# train mode
		self.embeddings.train()
		self.nn.train()

		# init optimizer, criterion
		trainable_parameters = [{"params": self.nn.parameters()}]
		# if embeddings are trainable, we need to update their parameters as well
		if self.embeddings_cls.trainable:
			trainable_parameters += [{
				"params": self.embeddings.parameters(),
				# prevent compressing embedding parameters towards zero
				# to make the representations more distinctive
				"weight_decay": 0
			}]
		optimizer_class = torch.optim.SGD if self.optimizer == "sgd" else torch.optim.AdamW
		self.optimizer = optimizer_class(trainable_parameters, lr=self.learning_rate)

		# pass weights to handle disbalance
		# class_weights = [self.vocab_links.counts[id] for id in self.vocab_links._vocab_reversed]
		# class_weights = torch.tensor(class_weights, device=DEVICE)
		# class_weights = 1 - class_weights	# higher weights for rarer classes
		criterion_class = (
			nn.CrossEntropyLoss if self.criterion == "crossentropy"
			else nn.BCEWithLogitsLoss if self.criterion == "bce"
			else nn.MultiLabelSoftMarginLoss
		)
		# ignore index not supported with Y being a distribution
		# self.criterion = criterion_class(weight=class_weights)
		self.criterion = criterion_class()

		train_dataloader = DataLoader(
			# will output batches of x's and y's;
			# triplets will be handled differently depending
			# on embeddings and NNs
			XYDataset(train_triplets, train_link_ids),
			batch_size=self.batch_size,
			drop_last=True
		)

		# for plotting
		losses = []

		# save the best model
		best_metric = -1
		_state_dict_nn = None
		_state_dict_embeddings = None

		n_steps = len(train_dataloader) * self.n_epochs
		if self.verbose: progress_bar = tqdm(total=n_steps)
		for i in range(self.n_epochs):
			if self.verbose: progress_bar.set_description_str(f"Epoch {i + 1} / {self.n_epochs}")
			for j, (x, y) in enumerate(train_dataloader):
				# accumulative_loss = accumulative_loss + (loss := self._fit_example(x, y))
				if self.verbose: progress_bar.update()
				loss = self._train_on_batch(x, y)
				losses.append(loss)
			# evaluate at the end of every epoch
			if dev_compounds is not None:
				eval_res = self._validate(dev_compounds)
				metric = eval_res.link_metrics[self.target_metric]
				if metric >= best_metric:
					_state_dict_nn = self.nn.state_dict()
					if self.embeddings_cls.trainable:
						_state_dict_embeddings = self.embeddings.state_dict()
					best_metric = metric

		if dev_compounds is not None:
			self.nn.load_state_dict(_state_dict_nn)

		self._state_dict = {
			# "nn_params": self.nn_params,	# to make sure architectures will be compatible
			# "embeddings_params": self.embeddings_params,
			"vocab_links": self.vocab_links,
			"nn": _state_dict_nn,
			"embeddings": _state_dict_embeddings	# None if not trainable
		}

		# save PNG of the plot
		if self.plot_buffer:
			plt.figure()
			plt.plot(losses)
			plt.xlabel(f"Step")
			plt.ylabel(f"{self.criterion.__class__.__name__} on {self.batch_size}-sized batch")
			# the plot is saved to the plot buffer and not to a file;
			# if you need the plot, you can easily get it from the buffer, e.g.
			# ```python
			# from PIL import Image
			# from PIL.PngImagePlugin import PngInfo

			# splitter.plot_buffer.seek(0)
			# plot = Image.open(splitter.plot_buffer)
			# info = PngInfo()
			# for key, value in splitter._metadata.items():
			# 	info.add_text(key, str(value))
			# plot.save(path, format="png", pnginfo=info)
			# ``` 
			plt.savefig(self.plot_buffer, format="png")
			plt.close()

		return self

	def _predict_batch(self, triplets: Iterable[Tuple[str]]) -> torch.Tensor:
		
		with torch.no_grad():

			x = self._embed(triplets)

			# force softmax is used with CrossEntropyLoss because the loss function
			# doesn't require softmax so it's omitted but it is required during
			# prediction to get probability distribution
			output = self.nn(x, force_softmax=True)
			# output = output.detach()  # will be detached later

		return output

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

		# Unlike N-gram model, we cannot predict every lemma separately.
		# To be more precise, we can, but that would just be very inefficient. 
		# Instead, we will predict batch-wise; now, the tricky thing here is that
		# batches do not correspond to the bounds or the number of the input lemmas
		# anyhow; because each lemma is of different lengths, it will yield an 
		# arbitrary number of contexts; for example a batch of 16 can have
		# 3 last positions of lemma1, then all 7 of lemma2, all 4 of lemma3 and the last one of lemma5.
		# For that reason, when collecting the positions of input lemmas, we will
		# store the boundaries of the masks belonging to them:
		# 0-4 is the 4 positions of lemma1, 4-7 is from lemma2, ...
		# Thus, when we obtain all the probabilities from the backbone NN
		# (we will concatenate all batches), we will know for sure which probs
		# correspond to which lemma.
		# Just to make it straightforward: what is done in-place in the N-gram model
		# (get position and predict the most probable link), is done outside
		# in NN-based models because it is executed not for a single lemma,
		# but for the whole batch; these results are then interpreted to get the final prediction.

		# to gather positions where there are no links, we force set `record_none_links`
		# as there are no positions otherwise because no links in test;
		# it will not affect training as it's already passed as well as on predictions
		# because prediction depends on the weights model learned
		record_none_links_orig = bool(self.record_none_links)   # copy
		self.record_none_links = True

		all_triplets = []
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
					triplets.append((left, right, mid))
					all_link_candidates[m][idx] = (i + 1, mid); idx += 1   # i + 1: correction for BOS
			all_triplets += triplets    # this will go to prediction inside of a batch
			# prepare for the next lemma
			m += len(triplets)
		all_link_candidates[m] = {} # "close the bracket"
		lc_keys = list(all_link_candidates.keys())

		# eval mode
		self.nn.eval()
		self.embeddings.eval()

		# return `record_none_links_orig`
		self.record_none_links = record_none_links_orig
		
		test_dataloader = DataLoader(
			# will only output batches of x's;
			# triplets will be handled differently depending
			# on embeddings and NNs
			XYDataset(all_triplets),
			batch_size=self.batch_size,
			drop_last=False # we cannot drop last because then not all the lemmas will be predicted
		)

		all_logits = []

		n_steps = len(test_dataloader)
		if self.verbose: progress_bar = tqdm(total=n_steps, desc="Predicting")
		for x in test_dataloader:
			if self.verbose: progress_bar.update()
			if len(x[0]) < self.batch_size:    # last batch
				# in this case, we want to pad the whole batch to normal size
				# and then drop excessive predictions;
				# since any input is embedded in such a manner that 
				# the length of it does not change the output shape of the embeddings,
				# we can simply pad the batch with empty texts
				diff = self.batch_size - len(x[0])  # x is of shape 3 x b
				# empty triplet for each missing observation
				for i in range(3):
					x[i] += tuple(['']) * diff
				logits = self._predict_batch(x)
				logits = logits[:-diff]
			else:
				logits = self._predict_batch(x)    
			all_logits += logits

		all_logits = torch.stack(all_logits, dim=0).cpu().detach().numpy()

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
	
	def save(self) -> None:
		out_dir = os.path.dirname(self.path)
		os.makedirs(out_dir, exist_ok=True) 
		torch.save(self._state_dict, self.path)

	def load(self) -> None:
		# warnings.warn(
		# 	"To ensure architecture compatibility, `nn_params` and `embeddings_params` from "
		# 	"pretrained are used. Yours will be ignored."
		# )
		state_dict = torch.load(self.path, weights_only=False)
		# self.nn_params = state_dict["nn_params"]
		# self.embeddings_params = state_dict["embeddings_params"]
		# vocab
		self.vocab_links = state_dict["vocab_links"]
		# embeddings
		self._build_embeddings()
		if state_dict["embeddings"]:
			self.embeddings.underlying_embeddings.load_state_dict(state_dict["embeddings"])
		# NN
		self._build_nn()
		self.nn.load_state_dict(state_dict["nn"])
		# eval mode is set in `predict()`


class BaseForwardNNSplitter(BaseNNSplitter):

	def _embed(self, triplets: Iterable[Tuple[str]]) -> torch.Tensor:

		triplets_array = np.array(triplets, dtype=np.object_)

		# In this implementation, we need to pass info about the whole
		# context to the model (not recurrently); however, different parts are (in the math sense)
		# independent so instead of embedding the whole contexts,
		# we want to embed each part of the triplet and concatenate the results if needed later.
		triplet_parts_embeddings = []
		for i in range(3):
			triplet_part = triplets_array[i, :]
			triplet_part_embeddings = self.embeddings.embed(triplet_part)   # b x emb
			triplet_parts_embeddings.append(triplet_part_embeddings)
		x = torch.stack(triplet_parts_embeddings, dim=1)   # b x emb x 3
		
		return x
	

class BaseRecurrentNNSplitter(BaseNNSplitter):

	def _embed(self, triplets: Iterable[Tuple[str]]) -> torch.Tensor:
		
		triplets_array = np.array(triplets, dtype=np.object_)

		# make empty links candidates have length of 1 for embedding
		_len = lambda seq: len(seq) or 1
		_list = lambda seq: list(seq) or ['']

		# In this implementation, we need to pass info about the context iteratively;
		# we can pass parts of the context to the model but that will likely be little efficient
		# since there will be only 3 recurrent steps so the model is not likely to learn
		# relations between the parts.
		# Example: "bund" -> "land" -> "es" -> embedding
		#
		# Another approach would be to concatenate the parts with a separator and 
		# make a single embedding over the whole resulting sequence. Since the 
		# sequences are going to be short, the model might just lack the meaningfulness
		# of the signal from the separator, and the whole sequence will be treated as one string
		# (boundaries will blur out) so no patterns inside might be captured.
		# Example: "b" -> "u" -> ... -> "|" -> "e" -> ... -> embedding
		#
		# Moreover, it is important to mention that the parts are processed in such a way
		# that they become mathematically independent; to give the model a clear signal
		# that there are 3 distinct part and that they are separate from each other,
		# we will pass each part separately and then adjoin the representations.
		# Example: "b" -> "u" -> ... -> embedding0, "e" -> "s" -> ... -> embedding1, ...
		triplet_parts_embeddings = []
		for i in range(3):
			triplet_part = triplets_array[i, :]
			# since each triplet can be of different size, we should pad them to
			# be able to process in one pass; however, the sequences are so short
			# padding can be too much of a noise for them so instead,
			# we will separate them onto subtensors of the same length (the range
			# is narrow so it'll be 2-3 subtensors), pass them separately and
			# concatenate the result; because of iterative processing the 
			# output shape will be the same for the subtensors so the shapes will converge
			lengths = np.vectorize(_len)(triplet_part)
			# this will assign a unique index for all of the length and mark
			# each position within the original array in accordance with their length index;
			# then we can iterate over each length index and gather original
			# array members such that their length index is the current one;
			# example: ["aaa", "bb", "c", "aaa"]
			# length mapping: length 3: index 0, length 2: index 1, length 1: index 2
			# length_indices: [0, 1, 2, 0]
			# all length 3 members are under index 0 in length_indices
			triplet_parts_embedding = torch.zeros((self.batch_size, self.nn.D * self.nn.hidden_size), device=DEVICE)    # b x (D * h)
			unique_lengths, length_indices = np.unique(lengths, return_inverse=True)
			for l, length_index in zip(unique_lengths, np.unique(length_indices)):
				# if we just stack all the results in the end, we will shuffle all the parts
				# from each other, so we need to remember positions of entries of different length
				# and plug the corresponding final embeddings back in
				original_positions, = np.where(length_indices == length_index)
				length_subpart = triplet_part[original_positions]
				# reshape b x l into b x 1 x l to iterate over characters
				length_subpart = np.array([_list(seq) for seq in length_subpart])
				# hidden for this length subpart
				hidden = self.nn.init_hidden()
				# iterate l times character-wise
				for j in range(l):
					# b_l_0 + b_l_1 + ... = b, lengths of different length subparts
					length_subpart_input = self.embeddings.embed(length_subpart[:, j]) # b_l_j x emb
					final_embedding, hidden = self.nn(length_subpart_input, hidden, r=True)   # b_l_j x (D * h),  b_l_j x (D * h)
				# triplet_parts_embedding.append(final_embedding)
				triplet_parts_embedding[original_positions] = final_embedding
			# stack all length subparts back, this is final embedding for the whole triplet part
			triplet_parts_embeddings.append(triplet_parts_embedding)
				
		x = torch.concat(triplet_parts_embeddings, dim=1)   # b x ((D * h) * 3)

		return x