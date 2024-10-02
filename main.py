import argparse
import json

from benchmarking.benchmarking import benchmark


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("config_path", help="Path to configuration file to run.")
	args = parser.parse_args()

	with open(args.config_path) as cfg:
		config = json.load(cfg)
	benchmark(config)


if __name__ == "__main__":
	main()