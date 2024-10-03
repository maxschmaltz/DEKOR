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
		dataframe with golds, predictions, whether the prediction is correct,
		target and pred link (all in raw string format),
		target and pred type, retrievable under columns "golds", "preds", "is_correct",
		"target_links", "pred_links", "target_types", "pred_types" respectively;
		note: if pred link is correct but not in the right place, "is_correct" equals `False`

	link_confmat : `pandas.Dataframe`
		overall link confusion matrix

	link_classification_report : `pandas.Dataframe`
		label-wise and weighted average precision, recall, f1 for links

	link_metrics : `pandas.Dataframe`
		weighted average precision, recall, f1, and overall accuracy for links

	type_confmat : `pandas.Dataframe`
		overall link type confusion matrix

	type_classification_report : `pandas.Dataframe`
		label-wise and weighted average precision, recall, f1 for link types

	type_metrics : `pandas.Dataframe`
		weighted average precision, recall, f1, and overall accuracy for link types

	placement : `pandas.Dataframe`
		percentage of correct placement of the link when predicted correctly
	"""

	def __init__(
		self,
		golds: Iterable[Compound],
		preds: Iterable[Compound],
		comp_types: Union[Iterable[str], str],
		target_links: Iterable[str],
		pred_links: Iterable[str],
		target_types: Iterable[str],
		pred_types: Iterable[str]
	) -> None:
		
		df = pd.DataFrame(
			index=[gold.raw for gold in golds],
			data={
				# "golds": [gold.raw for gold in golds],
				"pred": [pred.raw for pred in preds],
				"comp_type": comp_types,
				"is_correct": np.equal(golds, preds),
				"target_link": target_links,
				"pred_link": pred_links,
				"target_type": target_types,
				"pred_type": pred_types
			}
		)

		# confusion matrices, metrics
		link_confmat, link_classification_report, link_metrics = self._get_metrics(
			target_links, pred_links
		)
		type_confmat, type_classification_report, type_metrics = self._get_metrics(
			target_types, pred_types
		)
		# placement
		all_links = np.unique(target_links + pred_links).tolist()
		if "none" in all_links:
			all_links.remove("none")
		placement_data = {}
		for link in all_links:
			target_link_subdf = df[df["target_link"] == link]
			correct_pred_subdf = target_link_subdf[
				target_link_subdf["target_link"] == target_link_subdf["pred_link"]
			]
			correct_placement_subdf = correct_pred_subdf[correct_pred_subdf["is_correct"]]
			if not len(correct_pred_subdf):
				placement_percentage = 0
			else:
				placement_percentage = len(correct_placement_subdf) / len(correct_pred_subdf)
			placement_data[link] = placement_percentage
		placement = pd.Series(placement_data)

		self.df = df
		self.link_confmat = link_confmat
		self.link_classification_report = link_classification_report
		self.link_metrics = link_metrics
		self.type_confmat = type_confmat
		self.type_classification_report = type_classification_report
		self.type_metrics = type_metrics
		self.placement = placement

	def _get_metrics(self, golds: Iterable[str], preds: Iterable[str]) -> Tuple[pd.DataFrame]:

		n = len(golds)
		
		# confusion matrix
		all_labels = np.unique(golds + preds).tolist()
		confmat_data = confusion_matrix(golds, preds, labels=all_labels)
		confmat = pd.DataFrame(
			data=confmat_data,
			index=all_labels,
			columns=all_labels
		)
		# there were no "none" in golds so we'l remove it for further classification report
		if "none" in all_labels:
			all_labels.remove("none")
			confmat = confmat.drop("none", axis=0)
		# accuracy, precision, recall, f1 link-wise
		classification_report_data = {}
		for i, link in enumerate(all_labels):
			with warnings.catch_warnings():
				# temporarily turn off division by zero warnings
				warnings.simplefilter("ignore", RuntimeWarning)
				classification_report_data[link] = {
					# number of links with correctly assigned class i
					# over all links with assigned class i
					"precision": (precision := confmat_data[i, i] / confmat_data[:, i].sum()),
					# number of links with correctly assigned class i
					# over all links that are in class i
					"recall": (recall := confmat_data[i, i] / confmat_data[i, :].sum()),
					# harmonic mean of precision and recall
					"f1": 2 * ((precision * recall) / (precision + recall))
				}
		classification_report = pd.DataFrame(classification_report_data)
		classification_report = classification_report.fillna(0)	# remove NaNs
		# now add average metrics; we'll use weighted f1s and so on, where
		# classes with lower frequency will get higher weights
		class_weights = confmat.sum(axis=1)
		class_weights += 1	# add-1 smoothing to avoid 0 log values when there is a single link 
		class_weights = np.log(class_weights)	# smooth to handle too large disbalance
		class_weights = 1 / class_weights # higher weights for rarer classes
		class_weights = class_weights / class_weights.sum()	# normalize
		# inverse-weighted metrics
		metrics = np.matmul(classification_report, class_weights)
		classification_report["weighted_average"] = metrics
		# accuracy
		n_correct = confmat_data.diagonal().sum()
		links_accuracy = n_correct / n
		metrics["accuracy"] = links_accuracy

		return confmat, classification_report, metrics
	
	def __repr__(self) -> str:
		return str(self.link_metrics)


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

	target_link_components = []
	pred_link_components = []

	target_link_types = []
	pred_link_types = []

	for gold, pred in zip(golds, preds):

		target_link_components.append(
			gold.links[0].component
		)
		pred_link_components.append(
			"none" if not pred.links else pred.links[0].component
		)

		target_link_types.append(
			gold.links[0].type
		)
		pred_link_types.append(
			"none" if not pred.links else pred.links[0].type
		)

	res = EvaluationResult(
		golds,
		preds,
		comp_types,
		target_link_components,
		pred_link_components,
		target_link_types,
		pred_link_types
	)

	return res