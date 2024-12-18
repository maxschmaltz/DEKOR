"""
Module for parsing the DECOW16 compound dataset.
"""

import re
from dataclasses import dataclass, field
import pandas as pd
from typing import Tuple, List, Optional, Union, Literal


DE = "[a-zäöüß]*"   # German alphabet
LINK_TYPES = {  # (legacy)
	# links are in parentheses so that when the
	# raw compound is split, links are returns as well
	"addition_umlaut": f"(_\+={DE}_)",  # gast_+=e_buch = Gästebuch, mutter_+=_rente = Mütterrente
	"addition": f"(_\+{DE}_)",          # bund_+es_land = Bundesland
	"deletion": f"(_\-{DE}_)",          # schule_-e_jahr = Schuljahr
	"concatenation": "(_)"              # zeit_punkt = Zeitpunkt
}
LINK_PATTERN = '|'.join(LINK_TYPES.values())    # any link
UMLAUTS = {
	'au': 'äu',
	'a': 'ä',
	'o': 'ö',
	'u': 'ü'
}
UMLAUTS_REVERSED = {v: k for k, v in UMLAUTS.items()}


@dataclass
class Stem:

	"""
	A stem component of a `Compound`.

	Attributes
	----------
	component : `str`
		morpheme representation of the component

	realization : `str`, optional
		concrete realization of the component; if not given, equals to `component`

	span : `Tuple[int]`
		span of the component realization in the `Compound` lemma

	Example
	-------
	>>> compound = Compound("schule_-e_jahr")
	>>> compound.components[0].component
	"schule"
	>>> compound.components[0].realization
	"schul"
	>>> compound.components[0].span
	(0, 5)
	"""

	component: str = field(compare=True)
	realization: Optional[str] = field(compare=True, default=None)
	span: Tuple[int] = field(compare=True, kw_only=True)
	# further features

	def __post_init__(self) -> None:
		if not self.realization:
			self.realization = self.component

	def __repr__(self) -> str:
		return self.component


@dataclass
class Link:

	"""
	A link component of a `Compound`.

	Attributes
	----------
	component : `str`
		decodb representation of the component, e.g. "_+er_"

	realization : `str`, optional
		concrete realization of the component, e.g. "er"; if not given, equals to an empty string

	span : `Tuple[int]`
		span of the component in the `Compound` lemma

	type : `str`
		type of the link according to the COW dataset

	Example
	-------
	>>> compound = Compound("schule_-e_jahr")
	>>> compound.components[1].component
	"_-e_"
	>>> compound.components[1].realization
	""
	>>> compound.components[1].span
	(5, 5)
	>>> compound.components[1].type
	"deletion"
	"""

	component: str = field(compare=True)
	realization: Optional[str] = field(compare=True, default=None)
	span: Tuple[int] = field(compare=True, kw_only=True)
	type: str = field(compare=True, kw_only=True)
	# further features

	def __post_init__(self) -> None:
		if not self.realization:
			self.realization = ""

	def __repr__(self) -> str:
		return self.component


@dataclass
class Compound:

	"""
	Class for representation of a COW compound entry.

	Parameters
	----------
	raw : `str`
		COW dataset entry to analyze

	Attributes
	----------
	lemma : `str`
		recovered lemma representation of the compound

	stems : `List[Stem]`
		list of the stems of the compound

	links : `List[Link]`
		list of the links of the compound
		
	components : `List[Union[Stem, Link]]`
		list of the stems and the links of the compound, sorted by span

	Example
	-------
	>>> compound = Compound("schule_-e_jahr")
	>>> compound.raw
	"schule_-e_jahr"
	>>> compound.lemma
	"schuljahr"
	>>> compound.components
	["schule", "", "jahr"]
	"""

	raw: str = field(compare=True)
	lemma: str = field(compare=True, init=False)
	stems: List[Stem] = field(compare=False, init=False)
	links: List[Link] = field(compare=False, init=False)
	components: List[Union[Stem, Link]] = field(compare=True, init=False)

	def __post_init__(self) -> None:
		self._analyze(self.raw) # defines .stems, .links, .components, .lemma

	def _get_stem_obj(self, component: str) -> Stem:
		stem = Stem(
			component=component,
			# realization is just as the component at first, will be modified later if needed
			span=(self.j, self.j + len(component)),
		)
		self.j += len(component)
		return stem
	
	@staticmethod
	def eliminate_allomorphy(link: str) -> str:

		"""
		Eliminate allomorphy of a linking element, e.g. _+es_ to _+s_.

		Parameters
		----------
		link : `str`
			link for elimination

		Returns
		-------
		`str`
			link with eliminated allomorphy (input link if not applicable)
		"""

		# if link == "_+es_": link = "_+s_" # -es vs -s
		# elif link == "_+en_": link = "_+n_" # -en vs -n
		# elif link == "_+ens_": link = "_+ns_" # -ens vs -ns
		link = re.sub(r"_\+e(?=(n|s|ns))", "_+", link)    # just remove the -e-
		return link
	
	@staticmethod
	def return_allomorphy(link: str) -> str:

		"""
		Return allomorphy of a linking element after `eliminate_allomorphy()`, e.g. _+es_ to _+s_.

		Parameters
		----------
		link : `str`
			link for returning

		Returns
		-------
		`str`
			link with returned allomorphy (input link if not applicable)
		"""

		# if link == "_+s_": link = "_+es_" # -es vs -s
		# elif link == "_+n_": link = "_+en_" # -en vs -n
		# elif link == "_+ns_": link = "_+ens_" # -ens vs -ns
		link = re.sub(r"_\+(?=(n|s|ns))", "_+e", link)    # just return the -e-
		return link
	
	@staticmethod
	def get_link_info(link: str) -> Tuple[str]:

		"""
		Determines realization and type of the link.

		Parameters
		----------
		link : `str`
			string to analyze

		Returns
		-------
		`Tuple[str]`
			the link itself, it's realization, and it's type
		"""

		for link_type, pattern in LINK_TYPES.items():
			match = re.match(
				pattern.replace(DE, f'(?P<r>{DE})'), # add parenthesis to pattern to capture links under name "r"
				link
			)
			if match:
				# match will return 3 spans: the span of the whole match,
				# the span of the first capturing group that we use to return
				# the links when splitting a raw compound (same as the whole match),
				# and the last span is the realization of the component
				# that we capture in (DE)
				realization = match.groupdict().get("r", "")  # in concatenation, there is no group "r"
				if link_type == "deletion":
					realization = ""
				# eliminate allophones
				link = Compound.eliminate_allomorphy(link)
				return link, realization, link_type

	def _get_link_obj(self, component: str) -> Link:
		component, realization, link_type = self.get_link_info(component)
		link = Link(
				component=component,
				realization=realization,
				span=(self.j, self.j + len(realization)),
				type=link_type
			)
		self.j += len(realization)
		return link
	
	@staticmethod
	def get_deletion(deletion_link: str) -> str:

		"""
		In a deletion link like _-e_, determines the deletion substring: "e" in this example.

		Parameters
		----------
		deletion_link : `str`
			link to analyze

		Returns
		-------
		`str`
			deletion substring
		"""

		to_delete = re.match(
			LINK_TYPES["deletion"].replace(DE, f"(?P<r>{DE})"),
			deletion_link
		).group("r")   # capturing groups as is in `_get_link_obj()`
		return to_delete
	
	@staticmethod
	def perform_umlaut(string: str) -> str:

		"""
		Performs rightmost (!) umlaut, like "altstadt" --> "altstädt".

		Parameters
		----------
		string : `str`
			string to perform umlaut over

		Returns
		-------
		`str`
			string after performing umlaut (input string if not applicable)
		"""

		match = re.search("(au|a|o|u)[^aou]+$", string)
		if match:
			# the whole suffix containing the vowel
			suffix_before_umlaut = match.group(0)
			# the vowel itself
			umlaut = match.group(1)
			# perform umlaut in the suffix
			suffix_after_umlaut = re.sub(
				umlaut,
				UMLAUTS[umlaut],
				suffix_before_umlaut
			)
			# adjust realization: perform umlaut
			string = re.sub(
				f'{suffix_before_umlaut}$',
				suffix_after_umlaut,
				string
			)
		return string

	@staticmethod
	def reverse_umlaut(string: str) -> str:

		"""
		Reverse rightmost (!) umlaut, like "altstädt" --> "altstadt".

		Parameters
		----------
		string : `str`
			string to reverse umlaut in

		Returns
		-------
		`str`
			string after reversing umlaut (input string if not applicable)
		"""

		match = re.search("(äu|ä|ö|ü)[^äöü]+$", string)
		if match:
			# the whole suffix containing the vowel
			suffix_after_umlaut = match.group(0)
			# the vowel itself
			umlaut = match.group(1)
			# perform umlaut in the suffix
			suffix_before_umlaut = re.sub(
				umlaut,
				UMLAUTS_REVERSED[umlaut],
				suffix_after_umlaut
			)
			# adjust realization: perform umlaut
			string = re.sub(
				f"{suffix_after_umlaut}$",
				suffix_before_umlaut,
				string
			)
		return string
	
	def _fuse_link(self, link: Link) -> None:
		# build the lemma by fusing currently processed part with incoming links;
		# that includes, for example, umlaut and deletion processing, concatenation and so on;
		# adjust previous stem
		previous_stem = self.stems[-1]
		if link.type == "deletion":
			to_delete = self.get_deletion(link.component)
			ld = len(to_delete)
			previous_stem.realization = re.sub(
				f'{to_delete}$',
				'',
				previous_stem.component
			)
			# adjust spans accordingly
			self.j -= ld
			previous_stem.span = (previous_stem.span[0], previous_stem.span[1] - ld)
			link.span = (link.span[0] - ld, link.span[1] - ld)
		elif link.type == "addition_umlaut":
			# search for the closest to the link "umlautable" vowel;
			# will return 2 matches (if finds anything):
			# the whole suffix with umlaut, and the vowel itself (in the capturing group)
			previous_stem.realization = self.perform_umlaut(previous_stem.component)

	def _analyze(self, raw: str) -> None:
		raw = raw.lower()
		self.stems = []
		self.links = []
		self.j = 0  # global scope of index to be available from anywhere in the class
		# split by links; capturing groups will store the links
		components = re.split(LINK_PATTERN, raw)
		for component in components:
			if not component: continue  # `None` from capturing group occasionally occurs
			if not '_' in component:    # stem
				stem = self._get_stem_obj(component)
				self.stems.append(stem)
			else:
				link = self._get_link_obj(component)
				self._fuse_link(link) # adjust previous stem if needed
				self.links.append(link)
		del self.j
		self.components = sorted(self.stems + self.links, key=lambda c: c.span) # sort by span => order of appearance
		self.lemma = ''.join([component.realization for component in self.components])

	def __len__(self) -> int:
		return len(self.components)

	def __repr__(self) -> str:
		return f"{self.lemma} <-- {self.raw}"


def parse_gecodb(
	gecodb_path: str,
	version: Optional[Literal["4", "5", "ds"]]="5"
) -> pd.DataFrame:

	"""
	Parse the DECOW16-format compounds dataset.

	Parameters
	----------
	gecodb_path : `str`
		path to the TSV DECOW16-format compounds dataset

	version : `str`, optional, one of `["4", "5", "ds"]`, defaults to `"5"`
		expects columns with raw DECOW16 entries if `version="4"`,
		those plus first constituent count if `version="5"`,
		and those plus compound type if `version="ds"`

	Returns
	-------
	`pandas.DataFrame`
		dataframe with columns respectively to `version` plus compound column:
		* "raw": `str`: DECOW16-format compound entry (`"4"`, `"5"`, and `"ds"`)
		* "count": `int`: number of occurrences in DECOW16 (`"4"`, `"5"`, and `"ds"`)
		* "fc_count": `int`: number of first constituent occurrences in the data (`"5"` and `"ds"`)
		* "comp_type": `str`: compound type by first constituent frequency (`"ds"`)
		* "compound": `Compound`: processed compounds (`"4"`, `"5"`, and `"ds"`)
	"""

	colnames = (
		["raw", "count"] if version == "4" else
		["raw", "count", "fc_count"] if version == "5"
		else ["raw", "count", "fc_count", "comp_type"]
	)
	gecodb = pd.read_csv(
		gecodb_path,
		sep='\t',
		names=colnames,
		encoding="utf-8"
	)
	gecodb["compound"] = gecodb["raw"].apply(Compound)
	return gecodb