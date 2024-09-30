from typing import Dict, List, Optional

from dekor.utils.gecodb_parser import parse_gecodb


LINK_DISTRIBUTION = {
	# percentages are rounded to 4 digits
	# in comments counts from (Schäfer & Pankratz 2018)
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
	"common": 0.735,
	"infrequent": 0.17,
	"hapax_legomena": 0.03,	# unique first constituents
	"invented": 0.05,
	"allomorphic_first": 0.015
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

	# split gecodb into frequent and infrequent parts (by first constituent)
	# 	the ~10% most frequent ones
	gecodb_common = gecodb[gecodb["fc_count"] >= 200]
	# 	the ~20-50% range from most infrequent ones
	gecodb_rare = gecodb[(gecodb["fc_count"] >= 3) & (gecodb["fc_count"] <= 100)]
	
	sizes = sizes or DEFAULT_SIZES

	# "invented", "allomorphic_first", and "hapax_legomena" types
	# of compounds will be used in test to check zero-shot
	# model capabilities so in train and dev, we'll count their
	# share into rare
	type_distr_train_dev = COMPOUND_TYPE_DISTRIBUTION.copy()
	type_distr_train_dev["infrequent"] += type_distr_train_dev.pop("invented")
	type_distr_train_dev["infrequent"] += type_distr_train_dev.pop("allomorphic_first")
	type_distr_train_dev["infrequent"] += type_distr_train_dev.pop("hapax_legomena")

	type_distr_test = COMPOUND_TYPE_DISTRIBUTION
	
	# make train
	for train_size in sizes.get("train", DEFAULT_SIZES["train"]):
		
		n_common = train_size * type_distr_train_dev["common"]
		common_samples = []
		for link, share in LINK_DISTRIBUTION.items():
			n_links = int(n_common * share)	# floor
			n_links = max(3, n_links)	# at least 3 of a type
			link_subset = gecodb_common[gecodb_common["link"] == link]
			link_subset = link_subset[link_subset["used"] == False]
			link_sample = link_subset.sample(n_links)
			gecodb_common.loc[link_sample.index, "used"] = True
			common_samples.append(link_sample)