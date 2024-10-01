import os
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from dekor.utils.gecodb_parser import parse_gecodb


LINK_DISTRIBUTION = {
	# percentages are rounded to 4 digits;
	# in comments counts from (Schäfer & Pankratz 2018);
	# "_+er_" is given additional 0.0001 to add up to 1
	"_": 0.6046,	# 0.6025
	"_+s_": 0.2347,	# 0.2369
	"_+n_": 0.1289,	#  0.134
	"_-e_": 0.0128,	# 0.0078
	"_+e_": 0.0068,	# 0.0071
	"_+er_": 0.0046,	# 0.0038
	"_+=er_": 0.0037,	# 0.0036
	"_+=e_": 0.0018,	# 0.0022
	"_+ns_": 0.0017,	# 0.0016
	"_+=_": 0.0004	# 0.0005
}

COMPOUND_TYPE_DISTRIBUTION = {
	"common": 0.725,
	"infrequent": 0.225,
	"hapax_legomena": 0.05	# unique first constituents
}

DEFAULT_SIZES = {
	"train": [5000, 10000, 50000],
	"dev": [1000],
	"test": [500]
}


def make_datasets(
	gecodb_path: str,
	out_dir: str,
	sizes: Optional[Dict[str, List[float]]]=None
) -> None:
	
	gecodb = parse_gecodb(gecodb_path)	# takes some time
	gecodb["used"] = False	# track used items
	gecodb["link"] = [
		compound.links[0].component for compound in gecodb["compound"]
	]

	gecodb_splits = {}
	# split gecodb into frequent and infrequent parts (by first constituent)
	# 	the ~25% most frequent first constituents that make up
	#	~89% of all compounds
	gecodb_splits["common"] = gecodb[gecodb["fc_count"] >= 60]
	# 	the ~25% almost most infrequent first constituents that make up
	#	~2% of all compounds
	gecodb_splits["infrequent"] = gecodb[(gecodb["fc_count"] >= 3) & (gecodb["fc_count"] <= 12)]
	# 	the ~25% most infrequent first constituents that make up
	#	~0.5% of all compounds
	gecodb_splits["hapax_legomena"] = gecodb[gecodb["fc_count"] == 1]
	# compounds with highly allomorphic first constituents
	gecodb_allomorphic_fc = parse_gecodb("./dekor/gecodb_datasets/allomorphic_fc.tsv")
	gecodb_allomorphic_fc["comp_type"] = "allomorphic_fc"
	# invented compounds
	gecodb_invented = parse_gecodb("./dekor/gecodb_datasets/invented.tsv")
	gecodb_invented["comp_type"] = "invented"
	
	sizes = sizes or DEFAULT_SIZES

	# hapax legomena will be used in test to check zero-shot
	# model capabilities so in train and dev, we'll count their
	# share into infrequent
	type_distr = {"train": COMPOUND_TYPE_DISTRIBUTION.copy()}
	type_distr["train"]["infrequent"] += type_distr["train"].pop("hapax_legomena")
	type_distr["dev"] = type_distr["train"]
	type_distr["test"] = COMPOUND_TYPE_DISTRIBUTION

	samples = defaultdict(lambda: defaultdict(list))	# should be callable

	# first, we want to create the map of shares of the links to make sure no
	# links are crossing and not all links are taken up for a single split;
	# if needed, we'll have to adjust the number of shares (decrease to split up proportionally)
	train_total = max(sizes.get("train", DEFAULT_SIZES["train"]))
	dev_total = max(sizes.get("dev", DEFAULT_SIZES["dev"]))
	train_dev_total = sum([train_total, dev_total])
	test_total = max(sizes.get("test", DEFAULT_SIZES["test"]))
	total = sum([train_dev_total, test_total])
	for type, type_share in COMPOUND_TYPE_DISTRIBUTION.items():
		for link, link_share in LINK_DISTRIBUTION.items():
			if type in type_distr["train"]:
				n_of_type = total * type_share * link_share
			else:	# hapax legomena
				n_of_type = train_dev_total * type_share * link_share
			n_of_type = int(n_of_type)
			type_split = gecodb_splits[type]
			link_subset = type_split[type_split["link"] == link]
			if len(link_subset) < n_of_type:
				# correction to reduce sizes if needed
				correction = len(link_subset) / n_of_type
			else:
				correction = 1
			# splits of one type can share compounds in sets of different sizes 
			for split in ["train", "dev", "test"]:
				if not type in type_distr[split]: continue
				used_index = []
				for size in sizes.get(split, DEFAULT_SIZES[split]):
					n_links = int(size * type_share * link_share * correction)
					n_links = max(3, n_links) # min 3
					link_subset = link_subset[link_subset["used"] == False]
					try:
						link_sample = link_subset.sample(n_links)
					except ValueError:
						link_sample = link_subset
					link_sample["comp_type"] = type
					samples[split][size].append(link_sample)
					used_index.append(link_sample.index.values)
				used_index = np.concatenate(used_index, axis=0)
				type_split.loc[used_index, "used"] = True

	for split in ["train", "dev", "test"]:
		for size in sizes.get(split, DEFAULT_SIZES[split]):
			# add allomorphic and invented compounds
			sample = samples[split][size]
			if split == "test":
				sample += [
					gecodb_allomorphic_fc,
					gecodb_invented
				]
			sample = pd.concat(sample, axis=0)
			sample = sample.drop(["used", "link", "compound"], axis=1)
			sample.to_csv(
				os.path.join(out_dir, f"{split}_{size}.tsv"),
				sep='\t',
				header=False,
				index=False
			)