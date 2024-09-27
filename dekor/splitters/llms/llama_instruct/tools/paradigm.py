import requests
from bs4 import BeautifulSoup, Tag
import pandas as pd
from typing import Tuple, Union, List

GENUS_MAPPING = {
	"der": "m",
	"das": "n",
	"die": "f"
}


async def parse_paradigm(paradigm_section: Tag) -> Union[Tuple[str, pd.DataFrame], None]:
		
	try:

		table_data = {}
		
		rows = paradigm_section.find_all("tr")
		for i, row in enumerate(rows):
			cells = row.find_all(["th", "td"])
			row_data = []
			for cell in cells:
				if "<br/>" not in cell.decode():
					cell_data = cell.get_text().strip()
					row_data.append(cell_data)
				else:
					# several variants as in "des Bunds, des Bundes"
					cell_datas = cell.decode().split("<br/>")
					cell_datas = [BeautifulSoup(part, 'html.parser') for part in cell_datas]
					# cell_datas = [cell.get_text().strip() for cell in cell_datas]
					# cell_data = ', '.join(cell_datas)
					cell_data = cell_datas[0].get_text().strip()	# get more standard variant
					row_data.append(cell_data)
									
			table_data[i] = row_data

		table = pd.DataFrame(table_data)
		table.columns = table.iloc[0]  # first row as column names
		table = table.drop(table.index[0])  # drop the first row as it is now excessive
		table = table.set_index("").T  # first column as index, "" is read as default by Soup

		# some words contain additional columns (like "Singular 1" and "Singular 2" in "Herz");
		# the first column is the standard, so we need to remove others
		if "Singular 1" in table.columns.tolist():
			table = table.rename(columns={"Singular 1": "Singular"})
			other_sg_cols = [
				col for col in table.columns.tolist()
				if "Singular" in col and col != "Singular"
			]
			table = table.drop(other_sg_cols, axis=1)

		if "Plural 1" in table.columns.tolist():
			table = table.rename(columns={"Plural 1": "Plural"})
			other_pl_cols = [
				col for col in table.columns.tolist()
				if "Plural" in col and col != "Plural"
			]
			table = table.drop(other_pl_cols, axis=1)

		# we need a certain format in downstream functions so we should assure it here
		assert table.columns.tolist() == ["Singular", "Plural"]
		assert table.index.tolist() == ["Nominativ", "Genitiv", "Dativ", "Akkusativ"]
		# filter out defect paradigms ot proper noun tables
		assert table["Singular"]["Nominativ"].split()[0] in ["der", "das", "die"]
		assert table["Singular"]["Genitiv"].split()[0] in ["des", "der"]
		assert table["Singular"]["Dativ"].split()[0] in ["dem", "der"]
		assert table["Singular"]["Akkusativ"].split()[0] in ["den", "das", "die"]
		assert table["Plural"]["Nominativ"].split()[0] == "die"
		assert table["Plural"]["Genitiv"].split()[0] == "der"
		assert table["Plural"]["Dativ"].split()[0] == "den"
		assert table["Plural"]["Akkusativ"].split()[0] == "die"

		# retrieve genus if available
		genus = GENUS_MAPPING[table["Singular"]["Nominativ"].split()[0]]

		return genus, table

	except: return


async def get_paradigms(lemma: str) -> List[Union[Tuple[str, pd.DataFrame], None]]:
	
	# fetch the Wiktionary page for the word
	url = f"https://de.wiktionary.org/wiki/{lemma}"
	response = requests.get(url)
	
	if response.status_code != 200:
		return

	# parse the HTML content
	soup = BeautifulSoup(response.content, "html.parser")

	# paradigm table 
	paradigm_sections = soup.find_all("table", {"class": "inflection-table"})
	
	infos = [
		await parse_paradigm(paradigm_section)
		for paradigm_section in paradigm_sections
	]

	return infos