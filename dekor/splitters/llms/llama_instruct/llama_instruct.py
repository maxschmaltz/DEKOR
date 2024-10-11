import os
import asyncio
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, AIMessage
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.tools import render_text_description_and_args
from langchain_core.output_parsers import JsonOutputParser
# from langchain.output_parsers import OutputFixingParser
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from io import BytesIO
from tqdm import tqdm
from typing_extensions import TypedDict
from typing import Optional, List, Dict, Iterable, Awaitable, Annotated, Literal	# Self
from dotenv import load_dotenv

from dekor.splitters.base import BaseSplitter
from dekor.splitters.llms.llama_instruct.format_instructions import get_json_format_instructions
from dekor.utils.gecodb_parser import Compound

load_dotenv()	# load environment variables

sem = asyncio.Semaphore(10)	# limit the number of concurrent async tasks to 10

LINK_TYPES = {
    # to get the link component from its realization
    "addition_umlaut": "_+={c}_",	# gast_+=e_buch = Gästebuch, mutter_+=_rente = Mütterrente
    "addition": "_+{c}_",			# bund_+es_land = Bundesland
    "deletion": "_-e_",	# only e's	# schule_-e_jahr = Schuljahr
    "concatenation": "_"			# zeit_punkt = Zeitpunkt
}


class LlamaInstructSplitter(BaseSplitter):

	name = "llama-instruct"
	path = "meta/llama-3.1-405b-instruct"

	timeout = 15

	def __init__(
		self,	# base: 1 credit
		*,
		use_german_prompts: Optional[bool]=False,
		n_shots: Optional[int]=0,
		suggest_candidates: Optional[bool]=True,	# +2 credits
		retrieve_paradigm: Optional[bool]=False,	# +1 credits
		max_generations: Optional[int]=3,	# +2 credits per one regeneration (happens rarely)
		log_messages: Optional[bool]=True,
		save_graph: Optional[bool]=False,
		verbose: Optional[bool]=True
	) -> None:
		self.use_german_prompts = use_german_prompts
		self.n_shots = n_shots
		self.suggest_candidates = suggest_candidates
		if retrieve_paradigm: assert suggest_candidates
		self.retrieve_paradigm = retrieve_paradigm
		self.max_generations = max(max_generations, 1)
		self.messages_log = {} if log_messages else None
		self.plot_buffer = BytesIO() if save_graph else None	# unify names with NNs
		self.verbose = verbose

	@property
	def _metadata(self) -> dict:
		return {
			"use_german_prompts": self.use_german_prompts,
			"n_shots": self.n_shots,
			"suggest_candidates": self.suggest_candidates,
			"retrieve_paradigm": self.retrieve_paradigm,
			"max_generations": self.max_generations
		}

	def _build_llm(self) -> None:
		self.llm = ChatNVIDIA(
			model=self.path,
			nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
			temperature=0,	# ensure reproducibility
			max_tokens=1024,
			seed=42	# ensure reproducibility
		)

	def _build_graph(self) -> None:

		class SplitterState(TypedDict):

			# state just contains all the information needed to guide
			# the agent throughout the guide; at each node hit,
			# the whole state is passed to retrieve information from it
			messages: Annotated[List[BaseMessage], add_messages]	# for the agent
			all_messages: Annotated[List[BaseMessage], add_messages]	# for logging
			lemma: str	# will be rewritten any call
			tool_outputs: AIMessage	# paradigms if applicable
			candidates: str	# candidates retrieved from tendencies if applicable
			analysis: dict	# will be rewritten any call
			pred: Compound
			made_attempts: int	# track the number of attempts

		async def _retrieve_paradigm(state: SplitterState) -> Awaitable[dict]:

			lemma = state["lemma"]

			formatted_prompt = await self.retrieve_paradigms_prompt.ainvoke({
				"lemma": lemma
			})
			query: list[BaseMessage] = formatted_prompt.to_messages()
			ai_response = await self.llm.ainvoke(query)
			# since we are using ad hoc tool calling to enable multiple calls,
			# we won't be having any `tool_calls` and will have to parse them
			tool_calls = await self.json_parser.aparse(ai_response.content)
			tool_outputs = []
			for tool_call in tool_calls:
				try:
					tool_name = tool_call["name"]
					tool_args = tool_call["arguments"]
					tool_to_call = self.tools[tool_name]
					tool_output = await tool_to_call.ainvoke(tool_args)
					tool_outputs.append(tool_output)
				except: pass

			tool_outputs = '\n'.join(tool_outputs)
			# simulate AI message so that it can refer to itself
			# when giving candidates
			tool_outputs_message = AIMessage(content=tool_outputs)
			return {
				"all_messages": query + [ai_response],	# + [tool_outputs_message], 	# add later
				"tool_outputs": [tool_outputs_message]
			}

		async def _get_candidates(state: SplitterState) -> Awaitable[dict]:

			# Instead of doing usual RAG over the tendencies as it was planned, we decided to pass 
			# the whole list of tendencies to the model so that the model
			# decides what's relevant for prediction for several reasons:
			# 	1. After the guess, the model generates a JSON analysis that then gets evaluated;
			# 	it poses a difficulty in retrieval because the JSON itself has too little information
			# 	to be a good query, and generating the query is also problematic:
			#		1.1. The JSON has only the link and the constituents; constituents might
			# 		shift embeddings to an irrelevant section of the MD, and also the links
			# 		are too similar to be distinctive (-s- vs -es-). There is an option
			# 			to construct the vectorstore such that the search is performed over
			#		headers and the predicted constituents are omitted, but then see 2.
			# 		1.2. There might be another way: not searching by the link and then
			# 		checking if the predicted first constituent is one of the classes
			#		usual for that link like in 1.1, but vice versa: describing the class
			#		of the first constituent (generation), retrieving the links that are usual for it,
			#		and then checking if there is a match. However, this is problematic as
			#		many of the classes that prefer a certain link are too specific
			# 		to be described without knowing them; for example, it is hardly
			# 		obvious that one should check specifically for
			# 		"monosyllabic feminines with en-plural" for "Burg".
			# 	2. Even if there were a good way to create the query, retrieval would
			#	be precise but would lack recall, and thus would be arguably useful.
			# 	For example, if there are several tendencies that might work, A, B, and C,
			#	and the C is the needed one but it is not that close semantically to the
			#	query because there are some terms missing or something, and A and B
			# 	are retrieved, the whole retrieval is useless because C is not retrieved.
			#	3. For modern models, sequences of such lengths are not long.
			#	4. General tendencies are not less important but can be hardly retrieved
			#	with any of 1.1. or 1.2. They can be force pre-/appended but once again,
			#	they are not always relevant.

			lemma = state["lemma"]
			# if retrieving paradigms, prepend the tool outputs (paradigms)
			# to give the LLM a hint
			tool_outputs: list[AIMessage] = state.get("tool_outputs", None) or []

			formatted_prompt = await self.get_candidates_prompt.ainvoke({
				"lemma": lemma
			})
			query: list[BaseMessage] = formatted_prompt.to_messages()
			messages = query + tool_outputs	# first the task, then tool outputs
			ai_response = await self.llm.ainvoke(messages)

			return {
				"all_messages": messages + [ai_response],
				"candidates": ai_response.content
			}

		async def _guess(state: SplitterState) -> Awaitable[dict]:

			lemma = state["lemma"]
			candidates = state.get("candidates", None)	# if applicable, already populated by the guess
			messages = state["messages"]
			made_attempts = state["made_attempts"]

			if not messages:	# first guess
				candidates_instructions = await self._get_candidates_instructions(candidates)
				formatted_prompt = await self.analysis_prompt.ainvoke({
					"candidates": candidates_instructions,	# will be empty if not applicable
					"lemma": lemma
				})
				new_messages: list = formatted_prompt.to_messages()
			else: new_messages = []

			ai_response = await self.llm.ainvoke(messages + new_messages)
			analysis_str = ai_response.content
			# if the JSON is invalid, there will be an error here
			# and it will transfer us to the `except` block in `_apredict()`
			analysis = await self.analysis_output_parser.aparse(analysis_str)
			pred = self._predict(analysis)
			
			return {	# dict to update the state
				"all_messages": new_messages + [ai_response],
				"messages": new_messages + [ai_response],	# all we be appended
				"pred": pred,
				"analysis": analysis,
				"made_attempts": made_attempts + 1	# track the number of attempts
			}
		
		async def _validate(state: SplitterState) -> Literal["retry", "finish"]:

			# heuristically filter out predictions that cannot be correct;
			# this function routes the agent based on its prediction:
			# if the prediction is valid, then the agent exits,
			# otherwise it retries if the number of generations has not yet exceeded
			lemma = state["lemma"]
			pred = state["pred"]
			made_attempts = state["made_attempts"]

			valid = not (
				#	1. not a single valid link
				len(l := pred.links) != 1
				#	2. not a correct lemma
				or pred.lemma != lemma.lower()	# we capitalized it earlier
				# 	3. the link is impossible
				or not self._passes_filter(
					l[0].type,
					l[0].realization,
					(
						pred.stems[0].realization if pred.links[0].type != "deletion"
						else pred.stems[0].component
					)
				)
			)

			if valid:
				return "finish"	# success
			else:
				if made_attempts == self.max_generations:
					return "abort"	# don't have any more retries
				else: return "retry"

		async def _find_errors(state: SplitterState) -> Awaitable[dict]:

			lemma = state["lemma"]
			candidates = state.get("candidates", None)
			analysis = state["analysis"]

			# the graph can get here only if a pred was made and it was incorrect;
			# in that case, we just add to the messages history that the 
			# last analysis before this message (it will be recorded on each guess)
			# was incorrect and that he has to retry
			find_errors_instructions = await self._get_find_errors_instructions(candidates)
			formatted_prompt = await self.find_errors_prompt.ainvoke({
				"lemma": lemma,
				"analysis": analysis,
				"find_errors_instructions": find_errors_instructions
			})
			query: list[BaseMessage] = formatted_prompt.to_messages()
			ai_response = await self.llm.ainvoke(query)

			# from here, we need only the summary of what is wrong
			# in the last generation, so we update only messages
			return {
				"all_messages": query + [ai_response],
				"messages": [ai_response]
			}

		async def _abort(state: SplitterState) -> dict:
			lemma = state["lemma"]
			abort_message = AIMessage(content="Aborting.")
			return {
				"all_messages": [abort_message],
				"pred": Compound(lemma.lower())
			}
			
		# define start state
		graph_builder = StateGraph(SplitterState)

		# add nodes: functions in our case
		if self.suggest_candidates:
			graph_builder.add_node("get_candidates", _get_candidates)
		if self.retrieve_paradigm:
			graph_builder.add_node("retrieve_paradigm", _retrieve_paradigm)
		graph_builder.add_node("guess", _guess)
		graph_builder.add_node("abort", _abort)
		graph_builder.add_node("find_errors", _find_errors)

		# set nodes
		#	1. entrypoint: make the first guess
		if self.suggest_candidates:
			if self.retrieve_paradigm:
				# either retrieve the paradigm and first pass them to the candidate recruiter
				graph_builder.set_entry_point("retrieve_paradigm")
				graph_builder.add_edge("retrieve_paradigm", "get_candidates")
			else:
				# either get the candidates directly
				graph_builder.set_entry_point("get_candidates")
			# and pass the candidates to the guesser anyways
			graph_builder.add_edge("get_candidates", "guess")
		else:
			# or let it guess itself 
			graph_builder.set_entry_point("guess")
		#	2. the guess needs to be validated;
		# 	if the validation is successful,
		#	can return the result, otherwise retry if possible
		graph_builder.add_conditional_edges(
			"guess",	# start at guessing
			_validate,	# always go from guessing to validation
			# map from validation to the following action:
			# if "finish": end execution, otherwise retry if possible;
			# with such implementation, the agent can exit only
			# after validation
			{"finish": END, "abort": "abort", "retry": "find_errors"}
		)
		#	3. "finish": nothing to implement, will just exit
		#	4. "abort": make dummy compound from lemma, exit as well;
		#	to make it be able to exit from abortion, set the finish node
		graph_builder.set_finish_point("abort")
		#	5. "retry": first determine the error in the analysis,
		#	then pass the reasoning back to the agent and try to regenerate
		graph_builder.add_edge("find_errors", "guess")

		self._splitter = graph_builder.compile()

		# save PNG of the graph
		if self.plot_buffer:
			# the plot is saved to the plot buffer and not to a file;
			# if you need the plot, you can easily get it from the buffer, e.g.
			# ```python
			# from PIL import Image
			# from PIL.PngImagePlugin import PngInfo

			# splitter.plot_buffer.seek(0)
			# graph = Image.open(splitter.plot_buffer)
			# info = PngInfo()
			# for key, value in splitter._metadata.items():
			# 	info.add_text(key, str(value))
			# graph.save(path, format="png", pnginfo=info))
			# ``` 
			self.plot_buffer.write(self._splitter.get_graph().draw_mermaid_png())

	def _fit(
		self,
		train_compounds: Optional[Iterable[Compound]]=None,
		dev_compounds: Optional[Iterable[Compound]]=None	# unify params
	) -> None:
		
		self._build_llm()

		# first, build the prompt depending on the language;
		# here we have to do the import manually because 
		# conditioning like `prompts_source = prompts.german if ... else prompts.english`
		# and then importing from `prompts_source` cannot be resolved
		if self.use_german_prompts:

			from dekor.splitters.llms.llama_instruct.tools.german import (
				paradigm_tool_german as paradigm_tool
			)
			from dekor.splitters.llms.llama_instruct.prompts.german import (
				LINK_TYPES_MAPPING_GERMAN_REVERSED,
				CompoundAnalysisGerman as CompoundAnalysis,
				JSON_FORMAT_TEMPLATE_GERMAN as format_instructions_template,
				RETRIEVE_PARADIGMS_TEMPLATE_GERMAN as retrieve_paradigms_template,
				GET_CANDIDATES_TEMPLATE_GERMAN as get_candidates_template,
				get_candidates_instructions_german as get_candidates_instructions,
				ANALYSIS_INSTRUCTIONS_GERMAN as analysis_instructions,
				ANALYSIS_TEMPLATE_GERMAN as analysis_template,
				form_examples_german as form_examples,
				FIND_ERRORS_TEMPLATE_GERMAN as find_errors_template,
				get_find_errors_instructions_german as get_find_errors_instructions
			)
			self._first_noun_key = "erster_bestandteil"
			self._second_noun_key = "zweiter_bestandteil"
			self._link_realization_key = "fugenelement"
			self._link_type_key = "art_der_fuge"
			self._link_type_mapping = LINK_TYPES_MAPPING_GERMAN_REVERSED
			_tendencies_path = "./dekor/splitters/llms/llama_instruct/prompts/cat_tendencies_german.md"
		
		else:

			from dekor.splitters.llms.llama_instruct.tools.english import (
				paradigm_tool_english as paradigm_tool
			)
			from dekor.splitters.llms.llama_instruct.prompts.english import (
				LINK_TYPES_MAPPING_ENGLISH_REVERSED,
				CompoundAnalysisEnglish as CompoundAnalysis,
				JSON_FORMAT_TEMPLATE_ENGLISH as format_instructions_template,
				RETRIEVE_PARADIGMS_TEMPLATE_ENGLISH as retrieve_paradigms_template,
				GET_CANDIDATES_TEMPLATE_ENGLISH as get_candidates_template,
				get_candidates_instructions_english as get_candidates_instructions,
				ANALYSIS_INSTRUCTIONS_ENGLISH as analysis_instructions,
				ANALYSIS_TEMPLATE_ENGLISH as analysis_template,
				form_examples_english as form_examples,
				FIND_ERRORS_TEMPLATE_ENGLISH as find_errors_template,
				get_find_errors_instructions_english as get_find_errors_instructions
			)
			self._first_noun_key = "first_noun"
			self._second_noun_key = "second_noun"
			self._link_realization_key = "link_realization"
			self._link_type_key = "link_type"
			self._link_type_mapping = LINK_TYPES_MAPPING_ENGLISH_REVERSED
			_tendencies_path = "./dekor/splitters/llms/llama_instruct/prompts/cat_tendencies_english.md"

		# bind tools if we look up paradigm
		if self.retrieve_paradigm:
			self.tools = {paradigm_tool.name: paradigm_tool}
			tool_schemas = render_text_description_and_args(list(self.tools.values()))
			self.retrieve_paradigms_prompt = PromptTemplate.from_template(
				retrieve_paradigms_template,
				partial_variables={
					"tool_schemas": tool_schemas
				}
			)
			# Llama3.1 405B can't call a tool multiple times
			# so it will always return a single word.
			# ```
			# self.llm_with_paradigm = self.llm.bind_tools(
			# 	[paradigm_tool],
			# 	tool_choice="required"
			# )
			# ```
			# Instead we'll use prompt to ask it to generate
			# the name of the tool and the args for each call (ad hoc calling)
			self.json_parser = JsonOutputParser()

		if self.suggest_candidates:
			md_loader = UnstructuredMarkdownLoader(_tendencies_path)
			md_doc = md_loader.load()[0]	# reads the whole file into a single doc
			md_text = md_doc.page_content
			get_candidates_prompt = PromptTemplate.from_template(
				get_candidates_template,
				partial_variables={
					"tendencies": md_text
				}
			)
			self.get_candidates_prompt = get_candidates_prompt
		self._get_candidates_instructions = get_candidates_instructions	# we'll need it guess anyways

		# define format instructions
		format_instructions = get_json_format_instructions(
			format_instructions_template,
			CompoundAnalysis
		)

		# add examples if not zero-shot
		if self.n_shots:
			example_compounds = train_compounds[:self.n_shots]
			example_string = form_examples(example_compounds)
		else: example_string = ""

		analysis_prompt = PromptTemplate.from_template(
			analysis_template,
			partial_variables={
				"instructions": analysis_instructions,
				"format_instructions": format_instructions,
				"examples": example_string
			}
		)
		self.analysis_prompt = analysis_prompt

		# fixing parser with first try to parse the output with the
		# underlying parser (JSON parser over `CompoundAnalysis` in our case)
		# and will retry to fix invalid JSON with an LLM in case the first
		# parsing(s) fail(s)	# legacy
		analysis_output_parser = JsonOutputParser(pydantic_object=CompoundAnalysis)
		self.analysis_output_parser = analysis_output_parser
		# analysis_output_parser = OutputFixingParser.from_llm(
		# 	llm=self.llm,
		# 	parser=analysis_json_parser,
		# 	max_retries=2
		# )

		# prompt to correct invalid generations
		find_errors_prompt = PromptTemplate.from_template(
			find_errors_template,
			partial_variables={
				"instructions": analysis_instructions
			}
		)
		self.find_errors_prompt = find_errors_prompt
		self._get_find_errors_instructions = get_find_errors_instructions

		# in graph later, we will need to store the messages and be able 
		# to invoke the LLM separately on a list of messages (for retry)
		# so we won't have a single chain

		self._build_graph()

	def fit(
		self,
		*,
		train_compounds: Optional[Iterable[Compound]]=None,
		dev_compounds: Optional[Iterable[Compound]]=None,	# unify params
		test: Optional[bool]=False	# unify params
	):	# -> Self:	# won't work in python3.10 or older
		if self.n_shots:
			assert train_compounds is not None and len(train_compounds) >= self.n_shots
		self._fit(train_compounds=train_compounds)
		return self
	
	async def _apredict(self, lemma: str) -> Compound:

		async with sem:
		
			try:
				output = await asyncio.wait_for(
					self._splitter.ainvoke({
						"lemma": lemma.capitalize(),
						"made_attempts": 0	# will be incremented each guess
					}),
					timeout=self.timeout * self.max_generations
				)
				# compound_analysis = output["compound_analysis"]
				pred = output["pred"]
				if self._progress_bar: self._progress_bar.update()

				if self.messages_log is not None:
					messages = [
						{
							"type": message.__class__.__name__,
							"content": message.content
						}
						for message in output["all_messages"]
					]
					self.messages_log[lemma] = messages

			except Exception as e:
				# empty compound to distinguish between incorrect generations
				# and failed pipeline
				if "[402] Payment Required" in str(e):
					pass # change credits if needed and rebuild the LLM and graph
					# self.llm = ...
					# self._build_graph()
					# pred = ...

				pred = Compound("")
				print(e)
				if self.messages_log is not None:
					self.messages_log[lemma] = str(e)
				
			return pred

	def _predict(self, compound_analysis: Dict[str, str]) -> Compound:

		# get the prediction
		first_noun = compound_analysis[self._first_noun_key].lower()
		second_noun = compound_analysis[self._second_noun_key].lower()
		realization = compound_analysis[self._link_realization_key]
		link_type = self._link_type_mapping[compound_analysis[self._link_type_key]]

		link_component = LINK_TYPES[link_type].format(c=realization)
		pred_raw = first_noun + link_component + second_noun
		pred = Compound(pred_raw)

		return pred

	async def apredict(self, lemmas: List[str]) -> List[Compound]:
		tasks = [self._apredict(lemma) for lemma in lemmas]
		preds = await asyncio.gather(*tasks)	# async loop
		return preds
	
	def predict(self, lemmas: List[str]) -> List[Compound]:
		self._progress_bar = tqdm(total=len(lemmas), desc="Predicting") if self.verbose else None
		preds = asyncio.run(self.apredict(lemmas))
		return preds

	def save(self) -> None:
		# no save or load, the model is not fine-tuned and hosted remotely
		raise NotImplementedError(f"The method is not implemented for {self.__class__.__name__}")

	def load(self) -> None:
		# no save or load, the model is not fine-tuned and hosted remotely
		raise NotImplementedError(f"The method is not implemented for {self.__class__.__name__}")