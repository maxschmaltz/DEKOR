"""
Prompts for Llama3.1-based pipelines for splitting German compounds, English version.
"""

import json
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal, Iterable, Optional

from dekor.utils.gecodb_parser import Compound


_link_types_enum_english = Literal["Simple addition", "Addition with umlaut", "Deletion", "Concatenation"]
LINK_TYPES_MAPPING_ENGLISH = {
	"addition": "Simple addition",
	"addition_umlaut": "Addition with umlaut",
	"deletion": "Deletion",
	"concatenation": "Concatenation",
}
LINK_TYPES_MAPPING_ENGLISH_REVERSED = {v: k for k, v in LINK_TYPES_MAPPING_ENGLISH.items()}


class CompoundAnalysisEnglish(BaseModel):
	# to make sure the pipeline returns a valid JSON, we make this Pydantic model
	first_noun: str = Field(..., description="The first noun of the compound")
	second_noun: str = Field(..., description="The second noun of the compound")
	link_realization: str = Field(..., description="The link between the nouns")
	link_type: _link_types_enum_english = Field(..., description="Type of the link")


RETRIEVE_PARADIGMS_TEMPLATE_ENGLISH = """\
You were given the compound "{lemma}". Make guesses about which words could be considered \
to be its first constituent. You have to consider all possible variants. \
For **all** these words, you have to look up their paradigms. \
You have the following tools for this:

{tool_schemas}

Form tool calls for **every** constituent you are considering. As output, give \
a simple JSON string without additional text containing a list of tool calls. \
Each tool call is an object with keys "name", which contains the name of the tool, and "arguments", \
where the arguments for the tool are given according to the schema.\
"""


GET_CANDIDATES_TEMPLATE_ENGLISH = """\

In German compounds, the properties of the **first** constituent play a decisive role \
in choosing the linking element. This means that the **first** constituent limits the possible choices \
so that the link can be predicted with some confidence from that.
This is described by the following tendencies:


====
{tendencies}
====


Your colleague analyzes the compound "{lemma}". Your task is to define the most probable \
first constituents and respective linking elements and to sort them by probability \
with reasoning for that. When choosing, prefer more frequent words. The first constituent \
should always be nominative.
Always necessary check for the options you suggest to align with the composition of the compound, \
so that when concatenated the suggested first constituent and linking element, there are \
no missing or extra elements.


Example
====
Compound: "Bundesland"
Hint:
There are 3 probable analyses sorted by probability:
1. The first constituent is "Bund". That's a strong masculine with (e)s-Genitiv, \
which points to addition "es". That's the most probable analysis because the word "Bund" \
is really frequent and the analysis aligns with the composition of the word.
2. The first constituent is "Bunde", which points to addition "s". This analysis is not \
probable because the city name "Bunde" doesn't occur frequently at all.
3. The first constituent is "Bundes"; it can't have additions "s" or "es" then because they \
are already included in the first constituent, and then there is concatenation \
as there can't be anything else in this case. That's the most improbable analysis as \
there is no such word with nominative "Bundes".


Your actual task
=====
Compound: "{lemma}"
Hint:
"""


async def get_candidates_instructions_english(candidates: str) -> str:
	candidates_str = (
		"Your colleague has prepared the most likely "
		"pairs of first components and linking elements for you. "
		"When choosing, remember that probability is not always "
		"correlated with correctness, so be sure to necessarily check "
		"which analyses really fit. The hint from the colleague is:"
		f"\n\n{candidates}\n\n"
	)
	return candidates_str


# the default JSON format instructions from JSON parsers seem to
# confuse the LLM, so here's a lightweight variant of it 
JSON_FORMAT_TEMPLATE_ENGLISH = """\
The output should be a well-formatted JSON instance that strictly conforms to the JSON schema below:

```
{schema}
```

Return ONLY a pure JSON string surrounded by triple backticks (```).\
"""


ANALYSIS_INSTRUCTIONS_ENGLISH = """\
You are given a German compound. That is a N+N compound which means it consists \
of exactly 2 noun constituents with a link in between. There are 4 link types you might encounter: \
1. "Simple addition": an -s-, -n-, -en-, -e-, -er-, -ns-, or -ens- between the constituents; \
example: Bund + Land --> Bund-es-land.
2. "Addition with umlaut": an -e-, -er-, or an empty string between the constituents that trigger \
the umlaut of the stem vocal of the first constituent; example: Buch + Regal --> Büch-er-regal.
3. "Deletion": an empty string that triggers the deletion of the end schwa of the first constituent; \
example: Schule + Jahr --> Schul-jahr.
4. "Concatenation": the two constituents just get concatenated without additional sounds in between; \
example: arten + Tür --> Garten-tür.
"""


# n-shot prompt
ANALYSIS_TEMPLATE_ENGLISH = """\
{instructions}

Your task is to analyze the compound "{lemma}" in such a way that you define the original constituents \
and the link as well as the link type from the list above.

{candidates}
{format_instructions}
{examples}


Your actual task
=====
Compound: "{lemma}"
Analysis:
"""


# not async because is called in non-async `_fit()`
def form_examples_english(example_compounds: Iterable[Compound]):
	examples = []
	es = '\n'	# for easy escaping
	for compound in example_compounds:
		json_string = json.dumps(
			{
				"first_noun": compound.stems[0].component.capitalize(),
				"second_noun": compound.stems[1].component.capitalize(),
				"link_realization": compound.links[0].realization,
				"link_type": LINK_TYPES_MAPPING_ENGLISH[compound.links[0].type]
			},
			ensure_ascii=False,
			indent=4
		)
		# add triple backsticks and additional brackets to escape the JSON
		json_string = f"```{es}{json_string}{es}```"
		example = f"Compound: \"{compound.lemma.capitalize()}\"{es}Analysis:{es}{json_string}"
		examples.append(example)
	examples_str = f"{es * 2}Examples{es}===={es}{(es * 2).join(examples)}"
	return examples_str


# in order to not confuse the model with wrapping the whole above prompt
# with format instructions and so on into yet another prompt below,
# we separate the tasks: not regenerate the analysis here, but rather
# find the errors here and regenerate in the next guess
FIND_ERRORS_TEMPLATE_ENGLISH = """\
Your colleague had the following task:

{instructions}


They have produced the following analysis for the compound "{lemma}":

{analysis}


This analysis contains error(s) with respect to the task instructions. Find the \
error(s) and give a one-sentence summary what is wrong in this analysis so that \
your colleague corrects it. For that, take a careful look at which links \
may and may not occur in which positions according to the task instructions.\
"""


async def get_find_errors_instructions_english(candidates: Optional[str]=None) -> str:
	if not candidates:
		find_errors_instructions = (
			"Find the error(s) and give a one-sentence summary why this analysis is wrong, "
			"so that your colleague could correct it. For that, take a careful look at which links "
			"may and may not occur in which positions according to the task instructions."
		)
	else:
		find_errors_instructions = (
			"Find the error(s) and choose another option that fits best from the candidates below. "
			"Give a one-sentence summary why this analysis is wrong and explain which option below"
			"your colleague should consider."
			f"\n\n{candidates}"
		)
	return find_errors_instructions