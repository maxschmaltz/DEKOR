"""
Train and evaluation compound splitters.
"""

import argparse
import json

from benchmarking.benchmarking import benchmark


def main():

	"""
	Runs training + evaluation of a splitter according to the input configuration.
	The input configuration file is a JSON with the following fields:

	model : str
		the name of the splitter according to the `name` property of the respective class

	parameter_grid : Dict[str, Union[list, dict]]
		a list of values for each parameter of the splitter class; all the values will
		be combined, and the splitter will be trained and evaluated on each combination;
		you can also have a nested parameter grid i.e. a dictionary of the same structure
		instead of the list of values, if the respective parameter expects a `dict` value

	constants : dict
		a value for each parameter of the splitter that you want to keep the same for
		all the combinations in `parameter_grid`

	remove_keys : List[dict]
		if some of parameter combinations from `parameter_grid` require the absence
		of certain other parameters, this is described here; `remove_keys` is a list
		of dictionaries consisting exactly of 2 keys: `"condition"`, which is a `dict`
		with a subset of parameters that should trigger removal, and `"remove"`: a list
		of parameter keys to remove when the condition is met

	target_metric : str
		the metric by which the performance of the model with different configurations
		is compared; should be one of the metrics from the `metrics` property
		from the `EvaluationResult` class

	save_best_model : bool
		whether to save or not the model with the best configuration; if `True`,
		the splitter's method `save()` is called

	save_full_report : bool
		if `False`, saves only the metric scores for the different configurations,
		otherwise adds the confusion matrix, the classification report for each link,
		and the separate metrics for the best configuration, as well as the plot and the
		messages log if available

	train_datasets : List[str]
		a list of the paths to the train datasets you want to train the models on;
		if several are given, each configuration will be trained on each dataset
		and the scores will be collected separately

	dev_dataset : str
		the path to the dev dataset; if no dev dataset is used, a `null` should be given

	test_dataset : str
		the path to the test dataset

	out_dir : str
		the path to the directory to store the results to


	Note
	----
	Examples of configurations can be found under _./benchmarking/configs/_.


	Example
	-------
	```bash
	cd DEKOR
	python ./main.py ./path/to/you/config.json
	```
	"""

	parser = argparse.ArgumentParser()
	parser.add_argument("config_path", help="Path to configuration file to run.")
	args = parser.parse_args()

	with open(args.config_path) as cfg:
		config = json.load(cfg)
	benchmark(config)


if __name__ == "__main__":
	main()