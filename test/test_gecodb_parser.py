import unittest

from dekor.gecodb_parser import Compound, Stem, Link


class TestGecoDBParser(unittest.TestCase):

    # region zero links

    def test_concatenation_0(self):
        # parse compound
        gecodb_entry = "Frage_Stellung"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("frage", index=0, is_noun=True),
            Link("", index=1, type="concatenation"),
            Stem("stellung", index=2, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Fragestellung"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_concatenation_1(self):
        # parse compound
        gecodb_entry = "viel_Zahl"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("viel", index=0, is_noun=False),
            Link("", index=1, type="concatenation"),
            Stem("zahl", index=2, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Vielzahl"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_concatenation_2(self):
        # parse compound
        gecodb_entry = "selbst_Wert_Gefühl"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("selbst", index=0, is_noun=False),
            Link("", index=1, type="concatenation"),
            Stem("wert", index=2, is_noun=True),
            Link("", index=3, type="concatenation"),
            Stem("gefühl", index=4, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Selbstwertgefühl"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_hyphen_0(self):
        # parse compound
        gecodb_entry = "Online_--_Shop"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("online", index=0, is_noun=True),
            Link("-", index=1, type="hyphen"),
            Stem("shop", index=2, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Online-shop"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_hyphen_1(self):
        # parse compound
        gecodb_entry = "E_--_Mail_--_Adresse"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("e", index=0, is_noun=True),
            Link("-", index=1, type="hyphen"),
            Stem("mail", index=2, is_noun=True),
            Link("-", index=3, type="hyphen"),
            Stem("adresse", index=4, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "E-mail-adresse"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_hyphen_2(self):
        # parse compound
        gecodb_entry = "Max_--_Planck_--_Institut"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("max", index=0, is_noun=True),
            Link("-", index=1, type="hyphen"),
            Stem("planck", index=2, is_noun=True),
            Link("-", index=3, type="hyphen"),
            Stem("institut", index=4, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Max-planck-institut"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_umlaut_0(self):
        # parse compound
        gecodb_entry = "Mangel_+=_Rüge"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("mangel", index=0, is_noun=True),
            Link("", index=1, type="umlaut"),
            Stem("rüge", index=2, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Mängelrüge"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_umlaut_1(self):
        # parse compound
        gecodb_entry = "Mutter_+=_Zentrum"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("mutter", index=0, is_noun=True),
            Link("", index=1, type="umlaut"),
            Stem("zentrum", index=2, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Mütterzentrum"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_umlaut_2(self):
        # parse compound
        gecodb_entry = "Nagel_+=_kauen"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("nagel", index=0, is_noun=True),
            Link("", index=1, type="umlaut"),
            Stem("kauen", index=2, is_noun=False)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Nägelkauen"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    # endregion

    # region basic links

    def test_addition_0(self):
        # parse compound
        gecodb_entry = "Rente_+n_Versicherung"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("rente", index=0, is_noun=True),
            Link("n", index=1, type="addition"),
            Stem("versicherung", index=2, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Rentenversicherung"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_addition_1(self):
        # parse compound
        gecodb_entry = "sehen_+s_Würdigkeit"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("sehen", index=0, is_noun=False),
            Link("s", index=1, type="addition"),
            Stem("würdigkeit", index=2, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Sehenswürdigkeit"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_addition_2(self):
        # parse compound
        gecodb_entry = "Bund_+es_Verfassung_+s_Gericht"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("bund", index=0, is_noun=True),
            Link("es", index=1, type="addition"),
            Stem("verfassung", index=2, is_noun=True),
            Link("s", index=3, type="addition"),\
            Stem("gericht", index=4, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Bundesverfassungsgericht"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_deletion_nom_0(self):
        # parse compound
        gecodb_entry = "Schule_-e_Jahr"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("schule", index=0, is_noun=True),
            Link("e", index=1, type="deletion"),
            Stem("jahr", index=2, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Schuljahr"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_deletion_nom_1(self):
        # parse compound
        gecodb_entry = "Erde_-e_Geschoss"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("erde", index=0, is_noun=True),
            Link("e", index=1, type="deletion"),
            Stem("geschoss", index=2, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Erdgeschoss"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_deletion_nom_2(self):
        # parse compound
        gecodb_entry = "Ende_-e_Stufe"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("ende", index=0, is_noun=True),
            Link("e", index=1, type="deletion"),
            Stem("stufe", index=2, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Endstufe"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_deletion_non_nom_0(self):
        # parse compound
        gecodb_entry = "suchen_#en_Maschine"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("suchen", index=0, is_noun=False),
            Link("en", index=1, type="deletion"),
            Stem("maschine", index=2, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Suchmaschine"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_deletion_non_nom_1(self):
        # parse compound
        gecodb_entry = "warten_#n_Zeit"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("warten", index=0, is_noun=False),
            Link("n", index=1, type="deletion"),
            Stem("zeit", index=2, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Wartezeit"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_deletion_non_nom_2(self):
        # parse compound
        gecodb_entry = "wohnen_#en_Haus"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("wohnen", index=0, is_noun=False),
            Link("en", index=1, type="deletion"),
            Stem("haus", index=2, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Wohnhaus"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    # endregion

    # region complex links

    def test_replacement_0(self):
        # parse compound
        gecodb_entry = "Datum_-um_+en_Satz"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("datum", index=0, is_noun=True),
            Link("um", index=1, type="deletion"),
            Link("en", index=2, type="addition"),
            Stem("satz", index=3, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Datensatz"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_replacement_1(self):
        # parse compound
        gecodb_entry = "Thema_-a_+en_Bereich"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("thema", index=0, is_noun=True),
            Link("a", index=1, type="deletion"),
            Link("en", index=2, type="addition"),
            Stem("bereich", index=3, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Themenbereich"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_replacement_2(self):
        # parse compound
        gecodb_entry = "Hilfe_-e_+s_Mittel"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("hilfe", index=0, is_noun=True),
            Link("e", index=1, type="deletion"),
            Link("s", index=2, type="addition"),
            Stem("mittel", index=3, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Hilfsmittel"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_replacement_0(self):
        # parse compound
        gecodb_entry = "Gast_+=e_Buch"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("gast", index=0, is_noun=True),
            Link("", index=1, type="umlaut"),
            Link("e", index=2, type="addition"),
            Stem("buch", index=3, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Gästebuch"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_replacement_1(self):
        # parse compound
        gecodb_entry = "Fachkraft_+=e_Mangel"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("fachkraft", index=0, is_noun=True),
            Link("", index=1, type="umlaut"),
            Link("e", index=2, type="addition"),
            Stem("mangel", index=3, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Fachkräftemangel"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_replacement_2(self):
        # parse compound
        gecodb_entry = "Gut_+=er_Wagen"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("gut", index=0, is_noun=True),
            Link("", index=1, type="umlaut"),
            Link("er", index=2, type="addition"),
            Stem("wagen", index=3, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Güterwagen"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_addition_with_expansion_0(self):
        # parse compound
        gecodb_entry = "Bau_(t)_+en_Schutz"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("bau", index=0, is_noun=True),
            Link("t", index=1, type="expansion"),
            Link("en", index=2, type="addition"),
            Stem("schutz", index=3, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Bautenschutz"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_addition_with_expansion_1(self):
        # parse compound
        gecodb_entry = "Mineral_(i)_+en_Sammlung"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("mineral", index=0, is_noun=True),
            Link("i", index=1, type="expansion"),
            Link("en", index=2, type="addition"),
            Stem("sammlung", index=3, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Mineraliensammlung"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_addition_with_expansion_2(self):
        # parse compound
        gecodb_entry = "Embryo_(n)_+en_Forschung"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("embryo", index=0, is_noun=True),
            Link("n", index=1, type="expansion"),
            Link("en", index=2, type="addition"),
            Stem("forschung", index=3, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Embryonenforschung"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    # endregion

    # region infixes

    def test_infix_0(self):
        # infixes are sometimes eliminated, sometimes not so we have to OR-check both variants
        # parse compound
        gecodb_entry = "ober~_Land_+es_Gericht"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            ...,
            Link("es", index=1, type="addition"),
            Stem("gericht", index=2, is_noun=True)
        ]
        actual_components = compound.components
        # detect true list
        elim = Stem("land", index=0, is_noun=True)
        not_elim = Stem("oberland", index=0, is_noun=True)
        self.assertTrue(
            actual_components[0] == elim or
            actual_components[0] == not_elim
        )
        eliminated = actual_components[0] == elim
        expected_components[0] = elim if eliminated else not_elim
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Landesgericht" if eliminated else "Oberlandesgericht"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_infix_1(self):
        # infixes are sometimes eliminated, sometimes not so we have to OR-check both variants
        # parse compound
        gecodb_entry = "nicht~_Regierung_+s_Organisation"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            ...,
            Link("s", index=1, type="addition"),
            Stem("organisation", index=2, is_noun=True)
        ]
        actual_components = compound.components
        # detect true list
        elim = Stem("regierung", index=0, is_noun=True)
        not_elim = Stem("nichtregierung", index=0, is_noun=True)
        self.assertTrue(
            actual_components[0] == elim or
            actual_components[0] == not_elim
        )
        eliminated = actual_components[0] == elim
        expected_components[0] = elim if eliminated else not_elim
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Regierungsorganisation" if eliminated else "Nichtregierungsorganisation"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_infix_2(self):
        # infixes are sometimes eliminated, sometimes not so we have to OR-check both variants
        # parse compound
        gecodb_entry = "Arbeit_+s_un~_Fähigkeit"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("arbeit", index=0, is_noun=True),
            Link("s", index=1, type="addition"),
            ...
        ]
        actual_components = compound.components
        # detect true list
        elim = Stem("fähigkeit", index=2, is_noun=True)
        not_elim = Stem("unfähigkeit", index=2, is_noun=True)
        self.assertTrue(
            actual_components[2] == elim or
            actual_components[2] == not_elim
        )
        eliminated = actual_components[2] == elim
        expected_components[2] = elim if eliminated else not_elim
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Arbeitsfähigkeit" if eliminated else "Arbeitsunfähigkeit"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    # endregion

    def test_complex_compound_0(self):
        # parse compound
        gecodb_entry = "Stute_+n_Milch_trinken_#en_Kur"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("stute", index=0, is_noun=True),
            Link("n", index=1, type="addition"),
            Stem("milch", index=2, is_noun=True),
            Link("", index=3, type="concatenation"),
            Stem("trinken", index=4, is_noun=False),
            Link("en", index=5, type="deletion"),
            Stem("kur", index=6, is_noun=True)

        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Stutenmilchtrinkkur"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_complex_compound_1(self):
        # parse compound
        gecodb_entry = "lang_Frist_fördern_#n_Programm"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("lang", index=0, is_noun=False),
            Link("", index=1, type="concatenation"),
            Stem("frist", index=2, is_noun=True),
            Link("", index=3, type="concatenation"),
            Stem("fördern", index=4, is_noun=False),
            Link("n", index=5, type="deletion"),
            Stem("programm", index=6, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Langfristförderprogramm"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_complex_compound_2(self):
        # parse compound
        gecodb_entry = "Verwaltung_+s_rechts_Pflege_rechts_pflegen_#n_Gesetz"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("verwaltung", index=0, is_noun=True),
            Link("s", index=1, type="addition"),
            Stem("rechts", index=2, is_noun=False),
            Link("", index=3, type="concatenation"),
            Stem("pflege", index=4, is_noun=True),
            Link("", index=5, type="concatenation"),
            Stem("rechts", index=6, is_noun=False),
            Link("", index=7, type="concatenation"),
            Stem("pflegen", index=8, is_noun=False),
            Link("n", index=9, type="deletion"),
            Stem("gesetz", index=10, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Verwaltungsrechtspflegerechtspflegegesetz"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_complex_compound_3(self):
        # parse compound
        gecodb_entry = "Datum_-um_+en_Zugriff_+en_Zug_Riff_+s_Recht" # incorrect, preserved
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("datum", index=0, is_noun=True),
            Link("um", index=1, type="deletion"),
            Link("en", index=2, type="addition"),
            Stem("zugriff", index=3, is_noun=True),
            Link("en", index=4, type="addition"),
            Stem("zug", index=5, is_noun=True),
            Link("", index=6, type="concatenation"),
            Stem("riff", index=7, is_noun=True),
            Link("s", index=8, type="addition"),
            Stem("recht", index=9, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Datenzugriffenzugriffsrecht"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_complex_compound_4(self):
        # parse compound
        gecodb_entry = "Kind_+er_Tag_+es_Pflege_Tag_(e)_+s_pflegen_#n_Person" # incorrect, preserved
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("kind", index=0, is_noun=True),
            Link("er", index=1, type="addition"),
            Stem("tag", index=2, is_noun=True),
            Link("es", index=3, type="addition"),
            Stem("pflege", index=4, is_noun=True),
            Link("", index=5, type="concatenation"),
            Stem("tag", index=6, is_noun=True),
            Link("e", index=7, type="expansion"),
            Link("s", index=8, type="addition"),
            Stem("pflegen", index=9, is_noun=False),
            Link("n", index=10, type="deletion"),
            Stem("person", index=11, is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Kindertagespflegetagespflegeperson"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)
    

if __name__ == '__main__':
    unittest.main()