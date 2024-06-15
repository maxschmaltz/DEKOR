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
            Stem("frage", span=(0, 5), is_noun=True),
            Link("", span=(5, 5), type="concatenation"),
            Stem("stellung", span=(5, 13), is_noun=True)
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
            Stem("viel", span=(0, 4), is_noun=False),
            Link("", span=(4, 4), type="concatenation"),
            Stem("zahl", span=(4, 8), is_noun=True)
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
            Stem("selbst", span=(0, 6), is_noun=False),
            Link("", span=(6, 6), type="concatenation"),
            Stem("wert", span=(6, 10), is_noun=True),
            Link("", span=(10, 10), type="concatenation"),
            Stem("gefühl", span=(10, 16), is_noun=True)
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
            Stem("online", span=(0, 6), is_noun=True),
            Link("-", span=(6, 7), type="hyphen"),
            Stem("shop", span=(7, 11), is_noun=True)
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
            Stem("e", span=(0, 1), is_noun=True),
            Link("-", span=(1, 2), type="hyphen"),
            Stem("mail", span=(2, 6), is_noun=True),
            Link("-", span=(6, 7), type="hyphen"),
            Stem("adresse", span=(7, 14), is_noun=True)
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
            Stem("max", span=(0, 3), is_noun=True),
            Link("-", span=(3, 4), type="hyphen"),
            Stem("planck", span=(4, 10), is_noun=True),
            Link("-", span=(10, 11), type="hyphen"),
            Stem("institut", span=(11, 19), is_noun=True)
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
            Stem("mangel", "mängel", span=(0, 6), is_noun=True),
            Link("", span=(6, 6), type="umlaut"),
            Stem("rüge", "rüge", span=(6, 10), is_noun=True)
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
            Stem("mutter", "mütter", span=(0, 6), is_noun=True),
            Link("", span=(6, 6), type="umlaut"),
            Stem("zentrum", span=(6, 13), is_noun=True)
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
            Stem("nagel", "nägel", span=(0, 5), is_noun=True),
            Link("", span=(5, 5), type="umlaut"),
            Stem("kauen", span=(5, 10), is_noun=False)
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
            Stem("rente", span=(0, 5), is_noun=True),
            Link("n", span=(5, 6), type="addition"),
            Stem("versicherung", span=(6, 18), is_noun=True)
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
            Stem("sehen", span=(0, 5), is_noun=False),
            Link("s", span=(5, 6), type="addition"),
            Stem("würdigkeit", "würdigkeit", span=(6, 16), is_noun=True)
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
            Stem("bund", span=(0, 4), is_noun=True),
            Link("es", span=(4, 6), type="addition"),
            Stem("verfassung", span=(6, 16), is_noun=True),
            Link("s", span=(16, 17), type="addition"),
            Stem("gericht", span=(17, 24), is_noun=True)
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
            Stem("schule", "schul", span=(0, 5), is_noun=True),
            Link("e", span=(5, 5), type="deletion_nom"),
            Stem("jahr", span=(5, 9), is_noun=True)
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
            Stem("erde", "erd", span=(0, 3), is_noun=True),
            Link("e", span=(3, 3), type="deletion_nom"),
            Stem("geschoss", span=(3, 11), is_noun=True)
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
            Stem("ende", "end", span=(0, 3), is_noun=True),
            Link("e", span=(3, 3), type="deletion_nom"),
            Stem("stufe", span=(3, 8), is_noun=True)
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
            Stem("suchen", "such", span=(0, 4), is_noun=False),
            Link("en", span=(4, 4), type="deletion_non_nom"),
            Stem("maschine", span=(4, 12), is_noun=True)
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
            Stem("warten", "warte", span=(0, 5), is_noun=False),
            Link("n", span=(5, 5), type="deletion_non_nom"),
            Stem("zeit", span=(5, 9), is_noun=True)
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
            Stem("wohnen", "wohn", span=(0, 4), is_noun=False),
            Link("en", span=(4, 4), type="deletion_non_nom"),
            Stem("haus", span=(4, 8), is_noun=True)
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
            Stem("datum", "dat", span=(0, 3), is_noun=True),
            Link("um", span=(3, 3), type="deletion_nom"),
            Link("en", span=(3, 5), type="addition"),
            Stem("satz", span=(5, 9), is_noun=True)
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
            Stem("thema", "them", span=(0, 4), is_noun=True),
            Link("a", span=(4, 4), type="deletion_nom"),
            Link("en", span=(4, 6), type="addition"),
            Stem("bereich", span=(6, 13), is_noun=True)
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
            Stem("hilfe", "hilf", span=(0, 4), is_noun=True),
            Link("e", span=(4, 4), type="deletion_nom"),
            Link("s", span=(4, 5), type="addition"),
            Stem("mittel", span=(5, 11), is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Hilfsmittel"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_addition_with_umlaut_0(self):
        # parse compound
        gecodb_entry = "Gast_+=e_Buch"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("gast", "gäst", span=(0, 4), is_noun=True),
            Link("", span=(4, 4), type="umlaut"),
            Link("e", span=(4, 5), type="addition"),
            Stem("buch", span=(5, 9), is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Gästebuch"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_addition_with_umlaut_1(self):
        # parse compound
        gecodb_entry = "Fachkraft_+=e_Mangel"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("fachkraft", "fachkräft", span=(0, 9), is_noun=True),
            Link("", span=(9, 9), type="umlaut"),
            Link("e", span=(9, 10), type="addition"),
            Stem("mangel", span=(10, 16), is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Fachkräftemangel"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_addition_with_umlaut_2(self):
        # parse compound
        gecodb_entry = "Gut_+=er_Wagen"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("gut", "güt", span=(0, 3), is_noun=True),
            Link("", span=(3, 3), type="umlaut"),
            Link("er", span=(3, 5), type="addition"),
            Stem("wagen", span=(5, 10), is_noun=True)
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
            Stem("bau", span=(0, 3), is_noun=True),
            Link("t", span=(3, 4), type="expansion"),
            Link("en", span=(4, 6), type="addition"),
            Stem("schutz", span=(6, 12), is_noun=True)
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
            Stem("mineral", span=(0, 7), is_noun=True),
            Link("i", span=(7, 8), type="expansion"),
            Link("en", span=(8, 10), type="addition"),
            Stem("sammlung", span=(10, 18), is_noun=True)
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
            Stem("embryo", span=(0, 6), is_noun=True),
            Link("n", span=(6, 7), type="expansion"),
            Link("en", span=(7, 9), type="addition"),
            Stem("forschung", span=(9, 18), is_noun=True)
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
        # detect producted list
        elim = Stem("land", span=(0, 4), is_noun=True)
        not_elim = Stem("oberland", span=(0, 8), is_noun=True)
        actual_components = compound.components
        self.assertTrue(
            actual_components[0] == elim or
            actual_components[0] == not_elim
        )
        eliminated = actual_components[0] == elim
        # compare components and their properties
        if eliminated:
            expected_components = [
                Stem("land", span=(0, 4), is_noun=True),
                Link("es", span=(4, 6), type="addition"),
                Stem("gericht", span=(6, 13), is_noun=True)
            ]
            expected_pretty = "Landesgericht"
        else:
            expected_components = [
                Stem("oberland", span=(0, 8), is_noun=True),
                Link("es", span=(8, 10), type="addition"),
                Stem("gericht", span=(10, 17), is_noun=True)
            ]
            expected_pretty = "Oberlandesgericht"
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_infix_1(self):
        # infixes are sometimes eliminated, sometimes not so we have to OR-check both variants
        # parse compound
        gecodb_entry = "nicht~_Regierung_+s_Organisation"
        compound = Compound(gecodb_entry)
        # detect producted list
        elim = Stem("regierung", span=(0, 9), is_noun=True)
        not_elim = Stem("nichtregierung", span=(0, 14), is_noun=True)
        actual_components = compound.components
        self.assertTrue(
            actual_components[0] == elim or
            actual_components[0] == not_elim
        )
        eliminated = actual_components[0] == elim
        # compare components and their properties
        if eliminated:
            expected_components = [
                Stem("regierung", span=(0, 9), is_noun=True),
                Link("s", span=(9, 10), type="addition"),
                Stem("organisation", span=(10, 22), is_noun=True)
            ]
            expected_pretty = "Regierungsorganisation"
        else:
            expected_components = [
                Stem("nichtregierung", span=(0, 14), is_noun=True),
                Link("s", span=(14, 15), type="addition"),
                Stem("organisation", span=(15, 27), is_noun=True)
            ]
            expected_pretty = "Nichtregierungsorganisation"
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    def test_infix_2(self):
        # infixes are sometimes eliminated, sometimes not so we have to OR-check both variants
        # parse compound
        gecodb_entry = "Arbeit_+s_un~_Fähigkeit"
        compound = Compound(gecodb_entry)
        # detect producted list
        elim = Stem("fähigkeit", "fähigkeit", span=(7, 16), is_noun=True)
        not_elim = Stem("unfähigkeit", "unfähigkeit", span=(7, 18), is_noun=True)
        actual_components = compound.components
        self.assertTrue(
            actual_components[2] == elim or
            actual_components[2] == not_elim
        )
        eliminated = actual_components[2] == elim
        # compare components and their properties
        if eliminated:
            expected_components = [
                Stem("arbeit", span=(0, 6), is_noun=True),
                Link("s", span=(6, 7), type="addition"),
                Stem("fähigkeit", "fähigkeit", span=(7, 16), is_noun=True)
            ]
            expected_pretty = "Arbeitsfähigkeit"
        else:
            expected_components = [
                Stem("arbeit", span=(0, 6), is_noun=True),
                Link("s", span=(6, 7), type="addition"),
                Stem("unfähigkeit", "unfähigkeit", span=(7, 18), is_noun=True)
            ]
            expected_pretty = "Arbeitsunfähigkeit"
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)

    # endregion

    def test_complex_compound_0(self):
        # parse compound
        gecodb_entry = "Stute_+n_Milch_trinken_#en_Kur"
        compound = Compound(gecodb_entry)
        # compare components and their properties
        expected_components = [
            Stem("stute", span=(0, 5), is_noun=True),
            Link("n", span=(5, 6), type="addition"),
            Stem("milch", span=(6, 11), is_noun=True),
            Link("", span=(11, 11), type="concatenation"),
            Stem("trinken", "trink", span=(11, 16), is_noun=False),
            Link("en", span=(16, 16), type="deletion_non_nom"),
            Stem("kur", span=(16, 19), is_noun=True)

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
            Stem("lang", span=(0, 4), is_noun=False),
            Link("", span=(4, 4), type="concatenation"),
            Stem("frist", span=(4, 9), is_noun=True),
            Link("", span=(9, 9), type="concatenation"),
            Stem("fördern", "förder", span=(9, 15), is_noun=False),
            Link("n", span=(15, 15), type="deletion_non_nom"),
            Stem("programm", span=(15, 23), is_noun=True)
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
            Stem("verwaltung", span=(0, 10), is_noun=True),
            Link("s", span=(10, 11), type="addition"),
            Stem("rechts", span=(11, 17), is_noun=False),
            Link("", span=(17, 17), type="concatenation"),
            Stem("pflege", span=(17, 23), is_noun=True),
            Link("", span=(23, 23), type="concatenation"),
            Stem("rechts", span=(23, 29), is_noun=False),
            Link("", span=(29, 29), type="concatenation"),
            Stem("pflegen", "pflege", span=(29, 35), is_noun=False),
            Link("n", span=(35, 35), type="deletion_non_nom"),
            Stem("gesetz", span=(35, 41), is_noun=True)
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
            Stem("datum", "dat", span=(0, 3), is_noun=True),
            Link("um", span=(3, 3), type="deletion_nom"),
            Link("en", span=(3, 5), type="addition"),
            Stem("zugriff", span=(5, 12), is_noun=True),
            Link("en", span=(12, 14), type="addition"),
            Stem("zug", span=(14, 17), is_noun=True),
            Link("", span=(17, 17), type="concatenation"),
            Stem("riff", span=(17, 21), is_noun=True),
            Link("s", span=(21, 22), type="addition"),
            Stem("recht", span=(22, 27), is_noun=True)
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
            Stem("kind", span=(0, 4), is_noun=True),
            Link("er", span=(4, 6), type="addition"),
            Stem("tag", span=(6, 9), is_noun=True),
            Link("es", span=(9, 11), type="addition"),
            Stem("pflege", span=(11, 17), is_noun=True),
            Link("", span=(17, 17), type="concatenation"),
            Stem("tag", span=(17, 20), is_noun=True),
            Link("e", span=(20, 21), type="expansion"),
            Link("s", span=(21, 22), type="addition"),
            Stem("pflegen", "pflege", span=(22, 28), is_noun=False),
            Link("n", span=(28, 28), type="deletion_non_nom"),
            Stem("person", span=(28, 34), is_noun=True)
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_pretty = "Kindertagespflegetagespflegeperson"
        actual_pretty = compound.lemma
        self.assertEqual(expected_pretty, actual_pretty)
    

if __name__ == '__main__':
    unittest.main()