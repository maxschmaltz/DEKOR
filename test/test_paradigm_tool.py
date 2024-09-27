import unittest

# german version is enough, they are basically the same in different languages
from dekor.splitters.llms.llama_instruct.tools.german import paradigm_tool_german


class TestParadigmTool(unittest.TestCase):
	
	# see https://de.wiktionary.org/wiki/{Word}

	def test_paradigm_tool_0(self):
		lemma = "Mensch"
		expected_paradigm_desc = (
			"Es gibt das Wort \"Mensch\": Ein schwaches Maskulinum mit allen Formen "
			"auf -(e)n: \"Menschen\" (außer singularem Nominativ)."
			"\nEs gibt das Wort \"Mensch\": Ein Neutrum mit (e)s-Genitiv und er-Plural ohne Umlaut: "
			"\"des Mensch(e)s\", \"die Menscher\"."
		)
		actual_paradigm_desc = paradigm_tool_german(lemma)
		self.assertEqual(expected_paradigm_desc, actual_paradigm_desc)

	def test_paradigm_tool_1(self):
		lemma = "Affe"
		expected_paradigm_desc = (
			"Es gibt das Wort \"Affe\": Ein schwaches Maskulinum mit allen Formen "
			"auf -(e)n: \"Affen\" (außer singularem Nominativ)."
		)
		actual_paradigm_desc = paradigm_tool_german(lemma)
		self.assertEqual(expected_paradigm_desc, actual_paradigm_desc)

	def test_paradigm_tool_2(self):
		lemma = "Hund"
		expected_paradigm_desc = (
			"Es gibt das Wort \"Hund\": Ein Maskulinum mit (e)s-Genitiv und e-Plural ohne Umlaut: "
			"\"des Hund(e)s\", \"die Hunde\"."
		)
		actual_paradigm_desc = paradigm_tool_german(lemma)
		self.assertEqual(expected_paradigm_desc, actual_paradigm_desc)

	def test_paradigm_tool_3(self):
		lemma = "Bund"
		expected_paradigm_desc = (
			"Es gibt das Wort \"Bund\": Ein Maskulinum mit (e)s-Genitiv und e-Plural mit Umlaut: "
			"\"des Bund(e)s\", \"die Bünde\"."
			"\nEs gibt das Wort \"Bund\": Ein Neutrum mit (e)s-Genitiv und e-Plural ohne Umlaut: "
			"\"des Bund(e)s\", \"die Bunde\"."
		)
		actual_paradigm_desc = paradigm_tool_german(lemma)
		self.assertEqual(expected_paradigm_desc, actual_paradigm_desc)

	def test_paradigm_tool_4(self):
		lemma = "Betrieb"
		expected_paradigm_desc = (
			"Es gibt das Wort \"Betrieb\": Ein Maskulinum mit s-Genitiv und e-Plural ohne Umlaut: "
			"\"des Betriebs\", \"die Betriebe\"."
		)
		actual_paradigm_desc = paradigm_tool_german(lemma)
		self.assertEqual(expected_paradigm_desc, actual_paradigm_desc)

	def test_paradigm_tool_5(self):
		lemma = "Kind"
		expected_paradigm_desc = (
			"Es gibt das Wort \"Kind\": Ein Neutrum mit (e)s-Genitiv und er-Plural ohne Umlaut: "
			"\"des Kind(e)s\", \"die Kinder\"."
		)
		actual_paradigm_desc = paradigm_tool_german(lemma)
		self.assertEqual(expected_paradigm_desc, actual_paradigm_desc)

	def test_paradigm_tool_6(self):
		lemma = "Rad"
		expected_paradigm_desc = (
			"Es gibt das Wort \"Rad\": Ein Neutrum mit (e)s-Genitiv und er-Plural mit Umlaut: "
			"\"des Rad(e)s\", \"die Räder\"."
			"\nEs gibt das Wort \"Rad\": Ein Neutrum mit null-Plural ohne Umlaut: \"die Rad\"."
		)
		actual_paradigm_desc = paradigm_tool_german(lemma)
		self.assertEqual(expected_paradigm_desc, actual_paradigm_desc)

	def test_paradigm_tool_7(self):
		lemma = "Herz"
		expected_paradigm_desc = (
			"Es gibt das Wort \"Herz\": Ein Neutrum mit ens-Genitiv und en-Plural ohne Umlaut: "
			"\"des Herzens\", \"die Herzen\"."
		)
		actual_paradigm_desc = paradigm_tool_german(lemma)
		self.assertEqual(expected_paradigm_desc, actual_paradigm_desc)

	def test_paradigm_tool_8(self):
		lemma = "Anfänger"
		expected_paradigm_desc = (
			"Es gibt das Wort \"Anfänger\": Ein Maskulinum mit s-Genitiv und null-Plural ohne Umlaut: "
			"\"des Anfängers\", \"die Anfänger\"."
		)
		actual_paradigm_desc = paradigm_tool_german(lemma)
		self.assertEqual(expected_paradigm_desc, actual_paradigm_desc)

	def test_paradigm_tool_9(self):
		lemma = "Vater"
		expected_paradigm_desc = (
			"Es gibt das Wort \"Vater\": Ein Maskulinum mit s-Genitiv und null-Plural mit Umlaut: "
			"\"des Vaters\", \"die Väter\"."
		)
		actual_paradigm_desc = paradigm_tool_german(lemma)
		self.assertEqual(expected_paradigm_desc, actual_paradigm_desc)

	def test_paradigm_tool_10(self):
		lemma = "Burg"
		expected_paradigm_desc = (
			"Es gibt das Wort \"Burg\": Ein Femininum mit en-Plural ohne Umlaut: \"die Burgen\"."
		)
		actual_paradigm_desc = paradigm_tool_german(lemma)
		self.assertEqual(expected_paradigm_desc, actual_paradigm_desc)

	def test_paradigm_tool_11(self):
		lemma = "Frage"
		expected_paradigm_desc = (
			"Es gibt das Wort \"Frage\": Ein Femininum mit n-Plural ohne Umlaut: \"die Fragen\"."
		)
		actual_paradigm_desc = paradigm_tool_german(lemma)
		self.assertEqual(expected_paradigm_desc, actual_paradigm_desc)

	def test_paradigm_tool_12(self):
		lemma = "Hand"
		expected_paradigm_desc = (
			"Es gibt das Wort \"Hand\": Ein Femininum mit e-Plural mit Umlaut: \"die Hände\"."
		)
		actual_paradigm_desc = paradigm_tool_german(lemma)
		self.assertEqual(expected_paradigm_desc, actual_paradigm_desc)


if __name__ == '__main__':
	unittest.main()