# DEKOR: DEutscher KOmpositazerlegeR

Contents:
* [Abstract](#abstract)
* [Prerequisites](#prerequisites)
* [Usage of LLM-based Pipelines](#usage-of-llm-based-pipelines)
* [Evaluation](#evaluation)
* [Cite](#cite)


## Abstract

> Being an extremely productive word formation tool, compounding in German makes its natural language processing challenging due to the potentially unlimited vocabulary. Compound splitting as the most obvious solution is a non-trivial task since it requires analyzing unevenly and almost arbitrary distributed linking elements in compounds, and the ability to handle infrequent and unique compounds, which form the majority of compounds. In this thesis, I present LLM-based pipelines that address N+N compound splitting with an inventory of 12 links and are capable of handling rare and unknown compounds. The LLM based pipelines actively use reasoning, self-evaluation, and ad hoc referring to external sources to produce the best possible analyses and achieve a weighted accuracy of 0.677 and an absolute accuracy of 0.861.


## Prerequisites

The project is not published on PyPI and can be only cloned from Git:

```bash
git clone https://github.com/maxschmaltz/DEKOR.git
```

The Llama3.1 for the respective pipelines is run from [NVIDIA endpoints](https://build.nvidia.com/meta/llama-3_1-405b-instruct) so you need to set up a simple .env file with the `NVIDIA_API_KEY` system variable (see [template](./.env.template)).


## Usage of LLM-based Pipelines

The key point of the research is to bring an LLM-based solution to German compound splitting. To use the respective models, refer to the [documentation](./dekor/splitters/llms/llama_instruct/llama_instruct.py) in the respective code.

### Example

```python
from dekor.splitters import LlamaInstructSplitter

dekor = LlamaInstructSplitter(
	use_german_prompts=False,
	n_shots=0,
	suggest_candidates=True,
	retrieve_paradigm=True,
	max_generations=3
)
preds = dekor.predict(["belichtungsmöglichkeit"])
print(preds[0])
```

Output (in DECOW16 format):

```text
belichtung_+s_möglichkeit
```


## Evaluation

The overall scores are given below. For more detailed evaluation as well as discussion refer to the eponymous chapters of the [thesis paper](./papers/assessing_llm_based_pipelines_for_splitting_nominal_compounds_for_german.pdf).

| Model | Weighted | Absolute |
|-------|----------|----------|
| *N-gram* | 0.375 | 0.628 |
| *FFN* | 0.0 | 0.003 |
| *RNN* | 0.004 | 0.041 |
| *GRU* | 0.0 | 0.0 |
| *CNN* | 0.02 | 0.008 |
| *GBERT* | 0.136 | 0.552 |
| *ByT5* | 0.4 | 0.721 |
| *Llama-instruct₍base₎* | 0.572 | **0.861** |
| *Llama-instruct₍₊cand₎* | 0.605 | 0.836 |
| *Llama-instruct₍₊cand₊par₎* | **0.677** | 0.848 |

*Table: Weighted and absolute accuracy scores of the splitters on average (rounded to 3 digits after decimal point)*


## Cite

The official paper is yet to come.

An [unpublished thesis paper](./papers/assessing_llm_based_pipelines_for_splitting_nominal_compounds_for_german.pdf) is however available in the repo.