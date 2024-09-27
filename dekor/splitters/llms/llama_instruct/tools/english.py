import re
import asyncio
import pandas as pd
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool
from typing import Tuple, Union

from dekor.splitters.llms.llama_instruct.tools.paradigm import get_paradigms
from dekor.utils.gecodb_parser import Compound


GENUS_MAPPING_ENGLISH = {
	"m": "masculine",
	"n": "neuter",
	"f": "feminine"
}

DESCRIPTION_TEMPLATE = """\
There is the word "{lemma}": a {genus} with{maybe_gen_type} {pl_type} plural {umlaut}:\
{maybe_gen_form} "{pl_form}".\
"""


class ParadigmToolEnglishInput(BaseModel):
	lemma: str = Field(..., description="the input lemma for analysis")


async def get_paradigm_desc_english(lemma: str, info: Union[Tuple[str, pd.DataFrame], None]) -> str:

	if not info:
		return (
			f"The nominative word \"{lemma}\" was found, but it has a defect paradigm "
			"and is therefore either a proper name, or a non-German word, or "
			"a very rare word."
		)
	
	genus, paradigm = info

	# Since the tendencies are bound to either phonological / semantic properties
	# which we cannot confidently retrieve from here, or to morphological properties,
	# namely, genus, declination type, genitive and plural type (only genitive and plural
	# due to historical development of the links), the most we can retrieve from here is the latter.
	# We will therefore form a string description of the word heuristically to save some
	# resources without quality loss as the task is almost deterministic.

	# first, we must see if it's a weak masculine as it's a separate case
	if (
		genus == "m"
		and paradigm["Singular"]["Genitiv"] == "des " + re.sub("e$", "", lemma) + "en"
	):
		wordform = paradigm['Singular']['Genitiv'].split()[1]	# remove article
		return (
			f"There is the word \"{lemma}\": a weak masculine with all wordforms "
			f"ending on -(e)n: \"{wordform}\" (except for singular nominative)."
		)

	# then, we care for genitive only if it's strong masculine or neutra with (e)s genitive
	if genus in ["m", "n"]:
		if paradigm["Singular"]["Genitiv"] in ("des " + lemma + "s", "des " + lemma + "es"):
			# heuristically add variativity for one-syllable nouns;
			# this won't work accurately with deverbatives with prefixes (e.g. Betrieb)
			if len(re.findall("ei|ie|eu|äu|a|u|o|y|e|i", lemma, flags=re.I)) == 1:
				gen_form = f"des {lemma}(e)s"
				gen_type_desc = " (e)s-genitive and"
				gen_form_desc = f" \"{gen_form}\","
			else:
				gen_form = f"des {lemma}s"
				gen_type_desc = " s-genitive and"
				gen_form_desc = f" \"{gen_form}\","
		elif paradigm["Singular"]["Genitiv"] in ("des " + lemma + "ns", "des " + lemma + "ens"):
			gen_form = paradigm["Singular"]["Genitiv"]
			gen_form_desc = f" \"{gen_form}\","
			gen_type = re.sub(f"^{lemma}", "", re.sub("^des ", "", gen_form))
			gen_type_desc = f" {gen_type}-genitive and"
		else: gen_form_desc = gen_type_desc = ""
	else: gen_form_desc = gen_type_desc = ""

	# lastly, plural is always important
	pl_form = pl_form_desc = paradigm["Plural"]["Nominativ"]

	umlauted_lemma = Compound.perform_umlaut(lemma)
	if umlauted_lemma in paradigm["Plural"]["Nominativ"] and umlauted_lemma != lemma:	# umlaut
		pl_type_desc = re.sub(f"^{umlauted_lemma}", "", re.sub("^die ", "", pl_form)) or "null"
		umlaut_desc = "with umlaut"

	else:
		pl_type_desc = re.sub(f"^{lemma}", "", re.sub("^die ", "", pl_form)) or "null"
		umlaut_desc = "without umlaut"

	paradign_desc = DESCRIPTION_TEMPLATE.format(
		lemma=lemma,
		genus=GENUS_MAPPING_ENGLISH[genus],
		maybe_gen_type=gen_type_desc,
		pl_type=pl_type_desc,
		umlaut=umlaut_desc,
		maybe_gen_form=gen_form_desc,
		pl_form=pl_form_desc
	)
	return paradign_desc


async def aget_paradigm_english(lemma: str) -> str:

	lemma = lemma.capitalize()

	infos = await get_paradigms(lemma)

	if not infos:
		return f"The nominative word \"{lemma}\" wasn't found and probably doesn't exist."
	
	descs = [
		await get_paradigm_desc_english(lemma, info)
		for info in infos
	]
	desc = '\n'.join(descs)
	return desc.strip()	# in case of failed empty string


def get_paradigm_english(lemma: str) -> str:
	return asyncio.run(aget_paradigm_english(lemma))


paradigm_tool_english = StructuredTool.from_function(
	get_paradigm_english,
	coroutine=aget_paradigm_english,
	name="paradigm_tool",
	description="Retrieve the grammatical paradigm of a word fom Wiktionary.",
	args_schema=ParadigmToolEnglishInput
)