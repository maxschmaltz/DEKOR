"""
Tools for Llama3.1-based pipelines for splitting German compounds. German version.
"""

import re
import asyncio
import pandas as pd
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool
from typing import Tuple, Union

from dekor.splitters.llms.llama_instruct.tools.paradigm import _get_paradigms
from dekor.utils.gecodb_parser import Compound


GENUS_MAPPING_GERMAN = {
	"m": "Maskulinum",
	"n": "Neutrum",
	"f": "Femininum"
}

DESCRIPTION_TEMPLATE = """\
Es gibt das Wort "{lemma}": Ein {genus} mit{maybe_gen_type} {pl_type}-Plural {umlaut}:\
{maybe_gen_form} "{pl_form}".\
"""


class ParadigmToolGermanInput(BaseModel):
	lemma: str = Field(..., description="die Input-Lemma zum Analysieren")


async def get_paradigm_desc_german(lemma: str, info: Union[Tuple[str, pd.DataFrame], None]) -> str:

	if not info:
		return (
			f"Das Nominativ Wort \"{lemma}\" wurde gefunden, aber es hat ein defektes Paradigma "
			"und ist daher entweder ein Eigenname, oder ein nichtdeutsches Wort, oder "
			"ein sehr seltenes Wort."
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
			f"Es gibt das Wort \"{lemma}\": Ein schwaches Maskulinum mit allen Formen "
			f"auf -(e)n: \"{wordform}\" (außer singularem Nominativ)."
		)

	# then, we care for genitive only if it's strong masculine or neutra with (e)s genitive
	if genus in ["m", "n"]:
		if paradigm["Singular"]["Genitiv"] in ("des " + lemma + "s", "des " + lemma + "es"):
			# heuristically add variativity for one-syllable nouns;
			# this won't work accurately with deverbatives with prefixes (e.g. Betrieb)
			if len(re.findall("ei|ie|eu|äu|a|u|o|y|e|i", lemma, flags=re.I)) == 1:
				gen_form = f"des {lemma}(e)s"
				gen_type_desc = " (e)s-Genitiv und"
				gen_form_desc = f" \"{gen_form}\","
			else:
				gen_form = f"des {lemma}s"
				gen_type_desc = " s-Genitiv und"
				gen_form_desc = f" \"{gen_form}\","
		elif paradigm["Singular"]["Genitiv"] in ("des " + lemma + "ns", "des " + lemma + "ens"):
			gen_form = paradigm["Singular"]["Genitiv"]
			gen_form_desc = f" \"{gen_form}\","
			gen_type = re.sub(f"^{lemma}", "", re.sub("^des ", "", gen_form))
			gen_type_desc = f" {gen_type}-Genitiv und"
		else: gen_form_desc = gen_type_desc = ""
	else: gen_form_desc = gen_type_desc = ""

	# lastly, plural is always important
	pl_form = pl_form_desc = paradigm["Plural"]["Nominativ"]

	umlauted_lemma = Compound.perform_umlaut(lemma)
	if umlauted_lemma in paradigm["Plural"]["Nominativ"] and umlauted_lemma != lemma:	# umlaut
		pl_type_desc = re.sub(f"^{umlauted_lemma}", "", re.sub("^die ", "", pl_form)) or "null"
		umlaut_desc = "mit Umlaut"

	else:
		pl_type_desc = re.sub(f"^{lemma}", "", re.sub("^die ", "", pl_form)) or "null"
		umlaut_desc = "ohne Umlaut"

	paradign_desc = DESCRIPTION_TEMPLATE.format(
		lemma=lemma,
		genus=GENUS_MAPPING_GERMAN[genus],
		maybe_gen_type=gen_type_desc,
		pl_type=pl_type_desc,
		umlaut=umlaut_desc,
		maybe_gen_form=gen_form_desc,
		pl_form=pl_form_desc
	)
	return paradign_desc


async def aget_paradigm_german(lemma: str) -> str:

	lemma = lemma.capitalize()

	infos = await _get_paradigms(lemma)

	if not infos:
		return f"Das Nominativ Wort \"{lemma}\" wurde nicht gefunden und existiert vielleicht nicht."
	
	descs = [
		await get_paradigm_desc_german(lemma, info)
		for info in infos
	]
	desc = '\n'.join(descs)
	return desc.strip()	# in case of failed empty string
	

def get_paradigm_german(lemma: str) -> str:

	"""
	Get a textual description of the paradigm(s) of a German noun from Wiktionary. 
	
	That includes the type of declination (strong / weak) with a focus on genitive
	and Plural form types. For that, the tool sends a request to https://de.wiktionary.org/wiki/{lemma}
	and heuristically forms a description. If the paradigm is defect or the word is not found,
	it is also returned as a string hint (e.g. "Das Nominativ Wort {lemma} wurde nicht
	gefunden und existiert vielleicht nicht.").

	Parameters
	----------
	lemma : `str`
		the word to look up on Wiktionary; note: Wiktionary looks up the words
		by their nominative form

	Returns
	-------
	`str`
		heuristic string description of the paradigm(s) 
	"""

	return asyncio.run(aget_paradigm_german(lemma))


paradigm_tool_german = StructuredTool.from_function(
	get_paradigm_german,
	coroutine=aget_paradigm_german,
	name="paradigm_tool",
	description="Vom Wiktionary das grammatische Paradigma eines Wortes abrufen.",
	args_schema=ParadigmToolGermanInput
)