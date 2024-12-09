"""
Custom format instructions for Llama3.1-based pipelines for splitting German compounds.
"""

import json

from langchain_core.pydantic_v1 import BaseModel


def _remove_title(schema: dict) -> dict:
	# recursively remove excessive field "title"
	if isinstance(schema, dict):
		schema.pop("title", None)
		for subschema in schema.values():
			_remove_title(subschema)
	return schema


def get_json_format_instructions(prompt: str, pydantic_object: BaseModel) -> str:

	"""
	Get custom format instructions for prompting Llama3.1-based pipelines for splitting German compounds.

	Parameters
	----------
	prompt : `str`
		prompt to insert the resulting instructions to

	pydantic_object : `langchain_core.pydantic_v1.BaseModel`
		Pydantic model of the desired structured output

	Returns
	-------
	`str`
		initial prompt with inserted format instructions
	"""

	reduced_schema = _remove_title(pydantic_object.schema())	# remove excessive fields "title"
	schema_str = json.dumps(reduced_schema)
	json_format_instructions = prompt.format(schema=schema_str)
	return json_format_instructions