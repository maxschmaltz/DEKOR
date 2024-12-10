"""
Prompts for Llama3.1-based pipelines for splitting German compounds, German version.
"""

"""
Copyright (C) 2024 Maksim Shmalts

This program is licensed under GNU AGPL v3.0 with Commons Clause. See [LICENSE.md](./LICENSE.md) file for details.
"""

import json
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal, Iterable, Optional

from dekor.utils.gecodb_parser import Compound


_link_types_enum_german = Literal["Einfache Fuge", "Fuge mit Umlautung", "Subtraktion", "Nullfuge"]
LINK_TYPES_MAPPING_GERMAN = {
	"addition": "Einfache Fuge",
	"addition_umlaut": "Fuge mit Umlautung",
	"deletion": "Subtraktion",
	"concatenation": "Nullfuge",
}
LINK_TYPES_MAPPING_GERMAN_REVERSED = {v: k for k, v in LINK_TYPES_MAPPING_GERMAN.items()}


class CompoundAnalysisGerman(BaseModel):
	# to make sure the pipeline returns a valid JSON, we make this Pydantic model
	erster_bestandteil: str = Field(..., description="Der erste Bestandteil des Kompositums")
	zweiter_bestandteil: str = Field(..., description="Der zweite Bestandteil des Kompositums")
	fugenelement: str = Field(..., description="Die Fuge zwischen den Bestandteilen")
	art_der_fuge: _link_types_enum_german = Field(..., description="Die Art der Fuge")


RETRIEVE_PARADIGMS_TEMPLATE_GERMAN = """\
Dir wurde das Kompositum "{lemma}" gegeben. Vermute mal, welche Wörter als \
sein erster Bestandteil gelten könnten. Du musst alle möglichen Varianten betrachten. \
Für **all** diese Wörter musst du ihre Paradigmen nachschlagen. \
Dafür verfügst du über die folgenden Tools:

{tool_schemas}

Forme Tool Calls für **jeden** Bestandteil, den du betrachtest. Als Output gib \
einen bloßen JSON-String ohne zusätzlichen Text, in dem sich eine List von Tool Calls findet. \
Jeder Tool Call ist ein Objekt mit Keys "name", wo der name des Tools steckt, und "arguments", \
wo die Argumente für das Tool nach dem Schema stehen.\
"""


GET_CANDIDATES_TEMPLATE_GERMAN = """\
In deutschen Komposita spielen für die Fugenauswahl die Eigenschaften des **ersten** Bestandteils \
eine entscheidende Rolle. Das bedeutet, dass der **erste** Bestandteil die mögliche Auswahl \
begrenzt, sodass dadurch das Fugenelement mit einer Konfidenz vorhergesagt werden kann.
Das beschreiben die folgenden Tendenzen:


====
{tendencies}
====


Dein Kollege analysiert das Kompositum "{lemma}". Deine Aufgabe ist, zu bestimmen, \
welche ersten Bestandteile bzw. Fugen in diesem Kompositum möglich sind, \
und diese nach Wahrscheinlichkeit zu sortieren mit Begründung dazu. Bei der Wahl \
preferiere frequente Wörter als Bestandteile. Der erste Bestandteil muss immer im Nominativ stehen.
Checke immer unbedingt, dass die von dir vorgeschlagenen Optionen der Zusammensetzung des Kompositums \
entsprechen, imdem es beim Zusammenstellung des vorgeschlagenen Bestandteils und der Fuge \
keine fehlenden bzw. extra Elemente ergibt.


Beispiel
====
Kompositum: "Bundesland"
Hinweis:
Es gibt 3 wahrscheinliche Analyse, sortiert nach Wahrscheinlichkeit:
1. Der erste Bestandteil ist "Bund". Das ist ein starker Maskulinum mit (e)s-Genitiv ist, \
was auf einfache Fuge "es" aufweist. Das ist die wahrscheinlichste Analyse, weil das Wort \
"Bund" frequent ist und die Analyse der Zusammensetzung des Kompositums entspricht.
2. Der erste Bestandteil ist "Bunde", was auf einfache Fuge "s" aufweist. Diese Analyse ist
wahrscheinlich, weil der Stadtname "Bunde" gar nicht frequent vorkommt.
3. Der erste Bestandteil ist "Bundes"; Es kann aber keine einfache Fugen "s" oder "es" haben, weil diese \
schon im Bestandteil eingeschlossen sind, und es steht daher eine Nullfuge, \
weil es in dem Fall keine anderen sein können. Das ist die unwahrscheinlichste Analyse, \
weil es das Wort mit Nominativ "Bundes" gar nicht gibt. 


Deine eigentliche Aufgabe
=====
Kompositum: "{lemma}"
Hinweis:
"""


async def get_candidates_instructions_german(candidates: Optional[str]=None) -> str:
	if candidates:
		candidates_str = (
			"Dein Kollege hat für dich die wahrscheinlichsten "
			"Paare von ersten Bestandteilen und Fugenelemente vorbereitet. "
			"Bei der Wahl achte darauf, dass Wahrscheinlichkeit nicht immer "
			"der Korrektheit korrelliert ist, also checke unbedingt vor der Wahl, "
			"welche Analyse wirklich passen. Der Hinweis vom Kollege ist:"
			f"\n\n{candidates}\n\n"
		)
	else: candidates_str = ""
	return candidates_str


# the default JSON format instructions from JSON parsers seem to
# confuse the LLM, so here's a lightweight variant of it 
JSON_FORMAT_TEMPLATE_GERMAN = """\
Der Output soll ein gut formatierter JSON-String sein, der genau dem folgenden JSON-Schema entspricht:

```
{schema}
```

Gib NUR den reinen JSON-String zurück, der von dreifachen Graviszeichen (```) umgeben ist.\
"""


ANALYSIS_INSTRUCTIONS_GERMAN = """\
Du wirst ein Kompositum auf Deutsch gegeben. Das ist ein N+N-Kompositum, das heißt, \
es besteht aus genau 2 substantivischen Bestandteilen mit einem Fugenelement inzwischen. \
Es gibt 4 Arten von Fugenelementen, die dir vorkommen dürfen:
1. "Einfache Fuge": Ein -s-, -es-, -n-, -en-, -e-, -er-, -ns- oder -ens- zwischen den Bestandteilen; \
Beispiel: Bund + Land --> Bund-es-land.
2. "Fuge mit Umlautung": Ein -e-, -er- oder ein leerer String zwischen den Bestandteilen, das \
Umlautung des Stammvokals des ersten Bestandteils auslöst; Beispiel: Buch + Regal --> Büch-er-regal.
3. "Subtraktion": Ein leerer String, der Subtraktion der Schluss-Schwa des ersten Bestandteils auslöst; \
Beispiel: Schule + Jahr --> Schul-jahr.
4. "Nullfuge": Die zwei Bestandteile werden ohne zusätzliche Laute inzwischen konkateniert. \
Beispiel: Garten + Tür --> Garten-tür.
"""


# n-shot prompt
ANALYSIS_TEMPLATE_GERMAN = """\
{instructions}

Deine Aufgabe ist, das Kompositum "{lemma}" zu analysieren, indem du die zwei ursprünglichen \
Bestandteile und das Fugenelement bzw. die Art von der Fuge von der Liste oben bestimmen sollst.

{candidates}
{format_instructions}
{examples}


Deine eigentliche Aufgabe
=====
Kompositum: "{lemma}"
Analyse:
"""


# not async because is called in non-async `_fit()`
def form_examples_german(example_compounds: Iterable[Compound]):
	examples = []
	es = '\n'	# for easy escaping
	for compound in example_compounds:
		json_string = json.dumps(
			{
				"erster_bestandteil": compound.stems[0].component.capitalize(),
				"zweiter_bestandteil": compound.stems[1].component.capitalize(),
				"fugenelement": compound.links[0].realization,
				"art_der_fuge": LINK_TYPES_MAPPING_GERMAN[compound.links[0].type]
			},
			ensure_ascii=False,
			indent=4
		)
		# add triple backsticks and additional brackets to escape the JSON
		json_string = f"```{es}{json_string}{es}```"
		example = f"Kompositum: \"{compound.lemma.capitalize()}\"{es}Analyse:{es}{json_string}"
		examples.append(example)
	examples_str = f"{es * 2}Beispiele{es}===={es}{(es * 2).join(examples)}"
	return examples_str


# in order to not confuse the model with wrapping the whole above prompt
# with format instructions and so on into yet another prompt below,
# we separate the tasks: not regenerate the analysis here, but rather
# find the errors here and regenerate in the next guess
FIND_ERRORS_TEMPLATE_GERMAN = """\
Deine Kollege hatte die folgende Aufgabe:

{instructions}


Er hat zum Kompositum "{lemma}" die folgende Analyse produziert:

{analysis}


Die Analyse enthält Fehler im Bezug zur Aufgabe. {find_errors_instructions}\
"""


async def get_find_errors_instructions_german(candidates: Optional[str]=None) -> str:
	if not candidates:
		find_errors_instructions = (
			"Finde den oder die Fehler und fasse in einem Satz zusammen, "
			"warum diese Analyse falsch ist, damit dein Kollege "
			"die korrigieren könnte. Achte dafür aufmerksamkeit darauf, "
			"welche Fugen an welchen Stellen vorkommen und nicht vorkommen dürfen laut der Aufgabe."
		)
	else:
		find_errors_instructions = (
			"Finde den oder die Fehler und wähle eine andere Option, die am besten passt, "
			"von den Kandidaten unden aus. Fasse in einem Satz zusammen, warum "
			"diese Analyse falsch ist und erkläre welche Option von unden dein Kollege betrachten muss."
			f"\n\n{candidates}"
		)
	return find_errors_instructions