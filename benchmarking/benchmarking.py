import os
import json
from itertools import product
import numpy as np
import pandas as pd
from typing import List

from dekor.utils.gecodb_parser import parse_gecodb
from dekor.splitters import __all_splitters__
from dekor.eval import evaluate
from dekor.eval.evaluate import EvaluationResult


def grid_parameters(parameters: dict, constants: dict, blacklist: List[dict]) -> List[dict]:
	# subgrids (for NNs)
	for key, value in parameters.items():
		if isinstance(value, dict):
			parameters[key] = grid_parameters(value)
	return [
		(comb := {
				**{key: v for key, v in zip(parameters.keys(), value)},
				**constants
		}) for value in product(*parameters.values())
		# to stop undesirable combinations like Flair + embedding_dim
		if not any([
			d.items() <= comb.items()
			for d in blacklist
		])
	]


def benchmark(config: dict) -> None:
	
	parameters = config["parameter_grid"]
	constants = config["constants"]
	blacklist = config["blacklist"]
	parameter_grid = grid_parameters(parameters, constants, blacklist)

	if (dev_path := config["dev_dataset"]):
		dev_dataset = parse_gecodb(dev_path, version="ds")
		dev_compounds = dev_dataset["compound"].values
	else: dev_compounds = None

	test_dataset = parse_gecodb(config["test_dataset"], version="ds")
	test_compounds = test_dataset["compound"].values
	test_lemmas = [compound.lemma for compound in test_compounds]
	test_comp_types = test_dataset["comp_type"].values

	results = []

	splitter_cls = __all_splitters__[config["model"]]
	n_iter = 1
	for param_comb in parameter_grid:

		for train_path in (train_paths := config["train_datasets"]):

			train_dataset = parse_gecodb(train_path, version="ds")
			train_compounds = train_dataset["compound"].values

			print(f"\n\n{config["model"]}: iter {n_iter}/{len(parameter_grid * len(train_paths))}")
			print(
				', '.join(f"{key}: {value}" for key, value in param_comb.items()),
				f", train_size: {len(train_dataset)}"
			)

			splitter = splitter_cls(**param_comb)

			splitter = splitter.fit(
				train_compounds=train_compounds,
				dev_compounds=dev_compounds,
				test=True	# will only save the best model if needed
			)

			pred_compounds = splitter.predict(test_lemmas)

			eval_res = evaluate(test_compounds, pred_compounds, test_comp_types)
			if config["save_full_report"]:
				# if the full report is needed, we need to infer some data
				# from `Compound` objects and not their raw representations
				# so to avoid re-parsing them from raws, we'll just store them
				extended_df = eval_res.df.copy()
				extended_df["gold_comp"] = test_compounds
				extended_df["pred_comp"] = pred_compounds
			else: extended_df = None
			sizes = {
				"train_size": len(train_compounds),
				"dev_size": 0 if not dev_compounds else len(dev_compounds),
				"test_size": len(test_compounds)
			}
			results.append((eval_res, extended_df, splitter, sizes))

			n_iter += 1

	target_metric = config["target_metric"]
	result_sorted = sorted(
		results,
		key=lambda x: x[0].link_metrics[target_metric],
		reverse=True
	)

	# save general scores
	out_scores = [
		{
			**{
				"model": config["model"],
				"parameters": splitter._metadata
			},
			**sizes,
			**eval_res.link_metrics.to_dict()
		}
		for eval_res, _, splitter, sizes in result_sorted
	]
	out_dir = config["out_dir"]
	os.makedirs(out_dir, exist_ok=True)
	with open(os.path.join(out_dir, "scores.json"), "w") as f:
		json.dump(out_scores, f, indent=4)

	# if needed, save the best model and make the full report
	best_result, best_extented_df, best_splitter, _ = result_sorted[0]

	if config["save_best_model"]:
		best_splitter.save()

	if config["save_full_report"]:

		def write_res(res: EvaluationResult, dir: str) -> None:
			write_kwargs = lambda x: {
				"sep": '\t',
				"header": isinstance(x, pd.DataFrame),	# don't need with Series
				"index": True
			}
			dfize = lambda attr: (obj := getattr(res, attr)).to_csv(
				os.path.join(dir, f"{attr}.tsv"),
				**write_kwargs(obj)
			)
			for attr in [
				"df",
				"link_confmat",
				"link_classification_report",
				"link_metrics",
				"type_confmat",
				"type_classification_report",
				"type_metrics",
				"placement"
			]:
				dfize(attr)
		
		# first, run evaluation on compounds from different categoties
		comp_types = np.unique(best_extented_df["comp_type"].values)
		for comp_type in comp_types:
			comp_type_subdf = best_extented_df[
				best_extented_df["comp_type"] == comp_type
			]
			comp_type_golds = comp_type_subdf["gold_comp"].values
			comp_type_preds = comp_type_subdf["pred_comp"].values
			comp_type_res = evaluate(comp_type_golds, comp_type_preds, comp_type)
			comp_type_dir = os.path.join(out_dir, comp_type)
			os.makedirs(comp_type_dir, exist_ok=True)
			write_res(comp_type_res, comp_type_dir)

		# now general
		write_res(best_result, out_dir)

	pass