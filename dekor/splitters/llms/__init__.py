"""
LLM-based models for splitting German compounds based on the DECOW16 compound data.
"""

from dekor.splitters.llms.gbert import GBERTSplitter
from dekor.splitters.llms.byt5 import ByT5Splitter
from dekor.splitters.llms.llama_instruct import LlamaInstructSplitter