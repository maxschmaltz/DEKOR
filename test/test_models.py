import unittest
from sklearn.model_selection import train_test_split

from dekor.utils.gecodb_parser import parse_gecodb
from dekor.splitters import (

    NGramsSplitter,
    
	FFNSplitter,
    RNNSplitter,
    GRUSplitter,
    CNNSplitter,
	
	GBERTSplitter,
	ByT5Splitter,
	LlamaInstructSplitter

)
from dekor.benchmarking.benchmarking import eval_splitter


class TestLemmaCorrectness(unittest.TestCase):

    # In principle, the only thing we can test about the model
    # is it's adequacy in the sense that it does not remove
    # anything from or add anything from the compounds however it
    # splits them.

	# also, many parameters are shared between the models
	# (like `context_window` or embeddings in NN-based models),
	# so we distributed different values of share parameters
	# between different models in order to test more of them
	# more efficiently

	gecodb_path = "./resources/gecodb_v04.tsv"

	def get_data(self):
		gecodb = parse_gecodb(
			self.gecodb_path,
			min_count=100000
		)
		train_data, test_data = train_test_split(gecodb, train_size=0.75, shuffle=True)
		train_compounds = train_data["compound"].values
		test_compounds = test_data["compound"].values
		return train_compounds, test_compounds

	def test_ngrams(self):
		train_compounds, test_compounds = self.get_data()
		test_lemmas = [
			compound.lemma for compound in test_compounds
		]
		splitter = NGramsSplitter(
			n=3,
			record_none_links=False,
			verbose=False
		).fit(train_compounds=train_compounds, test=True)
		_, pred_compounds = eval_splitter(
			splitter=splitter,
			test_compounds=test_compounds
		)
		pred_lemmas = [
			compound.lemma for compound in pred_compounds
		]
		self.assertListEqual(test_lemmas, pred_lemmas)
        
	def test_ffn(self):
		train_compounds, test_compounds = self.get_data()
		test_lemmas = [
			compound.lemma for compound in test_compounds
		]
		splitter = FFNSplitter(
			context_window=4,
			record_none_links=False,
			embeddings_params={
				"name": "flair"
			},
			nn_params={
				"hidden_size": 128,
				"activation": "relu",
				"dropout_rate": 0.025
			},
			n_epochs=3,
			batch_size=4096,
			verbose=False
		).fit(train_compounds=train_compounds, test=True)
		_, pred_compounds = eval_splitter(
			splitter=splitter,
			test_compounds=test_compounds
		)
		pred_lemmas = [
			compound.lemma for compound in pred_compounds
		]
		self.assertListEqual(test_lemmas, pred_lemmas)

	def test_rnn(self):
		train_compounds, test_compounds = self.get_data()
		test_lemmas = [
			compound.lemma for compound in test_compounds
		]
		splitter = RNNSplitter(
			context_window=5,
			record_none_links=False,
			embeddings_params={
				"name": "torch",
				"embedding_dim": 16
			},
			nn_params={
				"hidden_size": 128,
				"activation": "relu",
				"dropout_rate": 0.025,
				"num_layers": 2
			},
			n_epochs=3,
			batch_size=4096,
			verbose=False
		).fit(train_compounds=train_compounds, test=True)
		_, pred_compounds = eval_splitter(
			splitter=splitter,
			test_compounds=test_compounds
		)
		pred_lemmas = [
			compound.lemma for compound in pred_compounds
		]
		self.assertListEqual(test_lemmas, pred_lemmas)

	def test_gru(self):
		train_compounds, test_compounds = self.get_data()
		test_lemmas = [
			compound.lemma for compound in test_compounds
		]
		splitter = GRUSplitter(
			context_window=6,
			record_none_links=False,
			embeddings_params={
				"name": "torch",
				"embedding_dim": 16
			},
			nn_params={
				"hidden_size": 128,
				"dropout_rate": 0.025,
				"num_layers": 2
			},
			n_epochs=3,
			batch_size=4096,
			verbose=False
		).fit(train_compounds=train_compounds, test=True)
		_, pred_compounds = eval_splitter(
			splitter=splitter,
			test_compounds=test_compounds
		)
		pred_lemmas = [
			compound.lemma for compound in pred_compounds
		]
		self.assertListEqual(test_lemmas, pred_lemmas)

	def test_cnn(self):
		train_compounds, test_compounds = self.get_data()
		test_lemmas = [
			compound.lemma for compound in test_compounds
		]
		splitter = CNNSplitter(
			context_window=7,
			record_none_links=False,
			embeddings_params={
				"name": "flair"
			},
			nn_params={
				"hidden_size": 128,
				"activation": "relu",
				"reduction": "conv",
				"dropout_rate": 0.025
			},
			n_epochs=3,
			batch_size=4096,
			verbose=False
		).fit(train_compounds=train_compounds, test=True)
		_, pred_compounds = eval_splitter(
			splitter=splitter,
			test_compounds=test_compounds
		)
		pred_lemmas = [
			compound.lemma for compound in pred_compounds
		]
		self.assertListEqual(test_lemmas, pred_lemmas)

	def test_gbert(self):
		train_compounds, test_compounds = self.get_data()
		test_lemmas = [
			compound.lemma for compound in test_compounds
		]
		splitter = GBERTSplitter(
			context_window=10,
			record_none_links=False,
			n_epochs=3,
			batch_size=64,
			verbose=False
		).fit(train_compounds=train_compounds, test=True)
		_, pred_compounds = eval_splitter(
			splitter=splitter,
			test_compounds=test_compounds
		)
		pred_lemmas = [
			compound.lemma for compound in pred_compounds
		]
		self.assertListEqual(test_lemmas, pred_lemmas)

	def test_byt5(self):
		train_compounds, test_compounds = self.get_data()
		test_lemmas = [
			compound.lemma for compound in test_compounds
		]
		splitter = ByT5Splitter(
			n_epochs=3,
			batch_size=4,
			verbose=False
		).fit(train_compounds=train_compounds, test=True)
		_, pred_compounds = eval_splitter(
			splitter=splitter,
			test_compounds=test_compounds
		)
		pred_lemmas = [
			compound.lemma for compound in pred_compounds
		]
		self.assertListEqual(test_lemmas, pred_lemmas)


	def test_llama_instruct(self):
		train_compounds, test_compounds = self.get_data()
		test_compounds = test_compounds[:5]	# make only 5 lemmas to spare credits
		test_lemmas = [
			compound.lemma for compound in test_compounds
		]
		splitter = LlamaInstructSplitter(
			n_shots=2,
			suggest_candidates=False,
			retrieve_paradigm=False,
			max_generations=1,
			verbose=False
		).fit(train_compounds=train_compounds, test=True)
		_, pred_compounds = eval_splitter(
			splitter=splitter,
			test_compounds=test_compounds
		)
		pred_lemmas = [
			compound.lemma for compound in pred_compounds
		]
		self.assertListEqual(test_lemmas, pred_lemmas)


if __name__ == '__main__':
    unittest.main()