import json

from langchain_core.pydantic_v1 import BaseModel


def remove_title(schema: dict) -> dict:
	# recursively remove excessive field "title"
	if isinstance(schema, dict):
		schema.pop("title", None)
		for subschema in schema.values():
			remove_title(subschema)
	return schema


def get_json_format_instructions(prompt: str, pydantic_object: BaseModel) -> str:
	reduced_schema = remove_title(pydantic_object.schema())	# remove excessive fields "title"
	schema_str = json.dumps(reduced_schema)
	json_format_instructions = prompt.format(schema=schema_str)
	return json_format_instructions