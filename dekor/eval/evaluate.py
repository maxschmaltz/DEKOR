"""
Module implementing compound splitter evaluator.
"""

from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import warnings
from typing import Iterable, Tuple, Union

from dekor.utils.gecodb_parser import Compound


class EvaluationResult:

	"""
	Evaluation result of compound splitter prediction against gold compounds.

	Attributes
	----------   
	df : `pandas.DataFrame`
		dataframe with golds, predictions, and compound class 
		retrievable under columns "golds", "preds", "comp_type" respectively

	confmat : `pandas.Dataframe`
		confusion matrix

	classification_report : `pandas.Dataframe`
		label-wise and weighted average accuracy

	metrics : `pandas.Dataframe`
		weighted average and absolute accuracy
	"""

	def __init__(
		self,
		golds: Iterable[Compound],
		preds: Iterable[Compound],
		comp_types: Union[Iterable[str], str]
	) -> None:
		
		data_df = pd.DataFrame(
			data={
				"gold": [gold.raw for gold in golds],
				"pred": [pred.raw for pred in preds],
				"comp_type": comp_types,
				"is_correct": np.equal(golds, preds),
				"gold_link": [gold.links[0].component for gold in golds],
				"pred_link": ["none" if not pred.links else pred.links[0].component for pred in preds]
			}
		)
		# define where the incorrect link was produced
		cond = (data_df["gold_link"] != data_df["pred_link"]) & (data_df["pred_link"] != "none")
		indexer = cond[cond].index.values
		data_df.loc[indexer, "pred_link"] = "err_link"
		# define where the correct link was placed wrong
		cond = (data_df["gold_link"] == data_df["pred_link"]) & (data_df["is_correct"] == False) & (data_df["pred_link"] != "none")
		indexer = cond[cond].index.values
		data_df.loc[indexer, "pred_link"] = "err_place"

		# confusion matrix, metrics
		confmat, classification_report, metrics = self._get_metrics(
			data_df["gold_link"].values, data_df["pred_link"].values
		)

		df = data_df.copy().drop(["is_correct", "gold_link", "pred_link"], axis=1)
		self.df = df
		self.confmat = confmat
		self.classification_report = classification_report
		self.metrics = metrics

	def _get_metrics(self, golds: Iterable[str], preds: Iterable[str]) -> Tuple[pd.DataFrame]:

		n = len(golds)
		
		# confusion matrix
		all_labels = np.unique((golds, preds))
		confmat_data = confusion_matrix(golds, preds, labels=all_labels)
		confmat = pd.DataFrame(
			data=confmat_data,
			index=all_labels,
			columns=all_labels
		)
		# there were no "none" and others in golds so we'l remove it for further classification report
		confmat = confmat.drop(["none", "err_link", "err_place"], axis=0, errors="ignore")
		# link-wise accuracy
		classification_report_data = {}
		for i, link in enumerate(all_labels):
			with warnings.catch_warnings():
				# temporarily turn off division by zero warnings
				warnings.simplefilter("ignore", RuntimeWarning)
				classification_report_data[link] = {
					# number of links with correctly assigned class i
					# over all links that are in class i
					"accuracy": confmat_data[i, i] / confmat_data[i, :].sum()
				}
		# Note: in the results, you'll see that the average precision is 1.0, whereas
		# the precision for rarer compound classes might be much lower. That originates from the
		# fact, that is no certain link could be predicted (e.g. no `_-e_` detected), the
		# precision becomes `NaN` because it is calculated as 0 / 0. Below, the `NaN`s are
		# replaces with zeros, which determines the final score below 1.0. So, if a compound
		# class is predicted with a precision lower than 1.0, it just means that some of the links were
		# not predicted at all; on the average, that doesn't hold because each link was predicted at least
		# once, so there are no "skipped links"; and the fact is, if a link was predicted, it was predicted
		# correctly.  
		classification_report = pd.DataFrame(classification_report_data)
		# classification_report = classification_report.fillna(0)	# remove NaNs
		# there were no "none" and others in golds so we'l remove it for weighted average
		classification_report = classification_report.drop(["none", "err_link", "err_place"], axis=1, errors="ignore")
		# now add average metrics; we'll use weighted accuracy and so on, where
		# classes with lower frequency will get higher weights
		class_weights = confmat.sum(axis=1)
		class_weights += 1	# add-1 smoothing to avoid 0 log values when there is a single link 
		class_weights = np.log(class_weights)	# smooth to handle too large disbalance
		class_weights = 1 / class_weights # higher weights for rarer classes
		class_weights = class_weights / class_weights.sum()	# normalize
		# inverse-weighted metrics
		metrics = np.matmul(classification_report, class_weights)
		metrics = metrics.rename({"accuracy": "weighted_accuracy"})
		classification_report["weighted_average"] = metrics
		# accuracy
		n_correct = confmat_data.diagonal().sum()
		links_accuracy = n_correct / n
		metrics["absolute_accuracy"] = links_accuracy

		return confmat, classification_report, metrics
	
	def __repr__(self) -> str:
		return str(self.metrics)


def evaluate(
	golds: Iterable[Compound],
	preds: Iterable[Compound],
	comp_types: Union[Iterable[str], str]
) -> EvaluationResult:

	"""
	Evaluate splitter predictions against gold compounds.

	Parameters
	----------
	golds : `Iterable[Compound]`
		gold compounds
	
	preds : `Iterable[Compound]`
		predictions

	Returns
	-------
	`EvaluationResult`
		result of evaluation
	"""

	assert len(golds) == len(preds)

	res = EvaluationResult(
		golds,
		preds,
		comp_types
	)

	return res