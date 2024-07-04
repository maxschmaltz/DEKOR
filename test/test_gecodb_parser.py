import unittest

from dekor.gecodb_parser import Compound, Stem, Link


class TestGecoDBParser(unittest.TestCase):

    def test_addition_umlaut_0(self):
        # parse compound
        raw = "mangel_+=_rüge"
        compound = Compound(raw)
        # compare components and their properties
        expected_components = [
            Stem("mangel", "mängel", span=(0, 6)),
            Link("_+=_", realization="", span=(6, 6), type="addition_umlaut"),
            Stem("rüge", "rüge", span=(6, 10))
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_lemma = "mängelrüge"
        actual_lemma = compound.lemma
        self.assertEqual(expected_lemma, actual_lemma)

    def test_addition_umlaut_1(self):
        # parse compound
        raw = "gast_+=e_buch"
        compound = Compound(raw)
        # compare components and their properties
        expected_components = [
            Stem("gast", "gäst", span=(0, 4)),
            Link("_+=e_", realization="e", span=(4, 5), type="addition_umlaut"),
            Stem("buch", span=(5, 9))
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_lemma = "gästebuch"
        actual_lemma = compound.lemma
        self.assertEqual(expected_lemma, actual_lemma)

    def test_addition_umlaut_2(self):
        # parse compound
        raw = "gut_+=er_wagen"
        compound = Compound(raw)
        # compare components and their properties
        expected_components = [
            Stem("gut", "güt", span=(0, 3)),
            Link("_+=er_", realization="er", span=(3, 5), type="addition_umlaut"),
            Stem("wagen", span=(5, 10))
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_lemma = "güterwagen"
        actual_lemma = compound.lemma
        self.assertEqual(expected_lemma, actual_lemma)


    def test_addition_0(self):
        # parse compound
        raw = "rente_+n_versicherung"
        compound = Compound(raw)
        # compare components and their properties
        expected_components = [
            Stem("rente", span=(0, 5)),
            Link("_+n_", realization="n", span=(5, 6), type="addition"),
            Stem("versicherung", span=(6, 18))
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_lemma = "rentenversicherung"
        actual_lemma = compound.lemma
        self.assertEqual(expected_lemma, actual_lemma)

    def test_addition_1(self):
        # parse compound
        raw = "nerv_+en_system"
        compound = Compound(raw)
        # compare components and their properties
        expected_components = [
            Stem("nerv", span=(0, 4)),
            Link("_+n_", realization="en", span=(4, 6), type="addition"),
            Stem("system", span=(6, 12))
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_lemma = "nervensystem"
        actual_lemma = compound.lemma
        self.assertEqual(expected_lemma, actual_lemma)

    def test_addition_2(self):
        # parse compound
        raw = "bund_+es_verfassung_+s_gericht"
        compound = Compound(raw)
        # compare components and their properties
        expected_components = [
            Stem("bund", span=(0, 4)),
            Link("_+s_", realization="es", span=(4, 6), type="addition"),
            Stem("verfassung", span=(6, 16)),
            Link("_+s_", realization="s", span=(16, 17), type="addition"),
            Stem("gericht", span=(17, 24))
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_lemma = "bundesverfassungsgericht"
        actual_lemma = compound.lemma
        self.assertEqual(expected_lemma, actual_lemma)


    def test_deletion_0(self):
        # parse compound
        raw = "schule_-e_jahr"
        compound = Compound(raw)
        # compare components and their properties
        expected_components = [
            Stem("schule", "schul", span=(0, 5)),
            Link("_-e_", realization="", span=(5, 5), type="deletion"),
            Stem("jahr", span=(5, 9))
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_lemma = "schuljahr"
        actual_lemma = compound.lemma
        self.assertEqual(expected_lemma, actual_lemma)

    def test_deletion_1(self):
        # parse compound
        raw = "erde_-e_geschoss"
        compound = Compound(raw)
        # compare components and their properties
        expected_components = [
            Stem("erde", "erd", span=(0, 3)),
            Link("_-e_", realization="", span=(3, 3), type="deletion"),
            Stem("geschoss", span=(3, 11))
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_lemma = "erdgeschoss"
        actual_lemma = compound.lemma
        self.assertEqual(expected_lemma, actual_lemma)

    def test_deletion_2(self):
        # parse compound
        raw = "ende_-e_stufe"
        compound = Compound(raw)
        # compare components and their properties
        expected_components = [
            Stem("ende", "end", span=(0, 3)),
            Link("_-e_", realization="", span=(3, 3), type="deletion"),
            Stem("stufe", span=(3, 8))
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_lemma = "endstufe"
        actual_lemma = compound.lemma
        self.assertEqual(expected_lemma, actual_lemma)


    def test_concatenation_0(self):
        # parse compound
        raw = "frage_stellung"
        compound = Compound(raw)
        # compare components and their properties
        expected_components = [
            Stem("frage", span=(0, 5)),
            Link("_", span=(5, 5), type="concatenation"),
            Stem("stellung", span=(5, 13))
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_lemma = "fragestellung"
        actual_lemma = compound.lemma
        self.assertEqual(expected_lemma, actual_lemma)

    def test_concatenation_1(self):
        # parse compound
        raw = "ring_finger"
        compound = Compound(raw)
        # compare components and their properties
        expected_components = [
            Stem("ring", span=(0, 4)),
            Link("_", span=(4, 4), type="concatenation"),
            Stem("finger", span=(4, 10))
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_lemma = "ringfinger"
        actual_lemma = compound.lemma
        self.assertEqual(expected_lemma, actual_lemma)

    def test_concatenation_2(self):
        # parse compound
        raw = "tier_schutz_verein"
        compound = Compound(raw)
        # compare components and their properties
        expected_components = [
            Stem("tier", span=(0, 4)),
            Link("_", span=(4, 4), type="concatenation"),
            Stem("schutz", span=(4, 10)),
            Link("_", span=(10, 10), type="concatenation"),
            Stem("verein", span=(10, 16))
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_lemma = "tierschutzverein"
        actual_lemma = compound.lemma
        self.assertEqual(expected_lemma, actual_lemma)


    def test_complex_compound_0(self):
        # parse compound
        raw = "schaden_+s_ersatz_anspruch"
        compound = Compound(raw)
        # compare components and their properties
        expected_components = [
            Stem("schaden", span=(0, 7)),
            Link("_+s_", realization="s", span=(7, 8), type="addition"),
            Stem("ersatz", span=(8, 14)),
            Link("_", span=(14, 14), type="concatenation"),
            Stem("anspruch", span=(14, 22))
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_lemma = "schadensersatzanspruch"
        actual_lemma = compound.lemma
        self.assertEqual(expected_lemma, actual_lemma)

    def test_complex_compound_1(self):
        # parse compound
        raw = "haupt_schule_-e_abschluss"
        compound = Compound(raw)
        # compare components and their properties
        expected_components = [
            Stem("haupt", span=(0, 5)),
            Link("_", span=(5, 5), type="concatenation"),
            Stem("schule", realization="schul", span=(5, 10)),
            Link("_-e_", realization="", span=(10, 10), type="deletion"),
            Stem("abschluss", span=(10, 19))
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_lemma = "hauptschulabschluss"
        actual_lemma = compound.lemma
        self.assertEqual(expected_lemma, actual_lemma)

    def test_complex_compound_2(self):
        # parse compound
        raw = "bund_+es_arzt_+=e_kammer"
        compound = Compound(raw)
        # compare components and their properties
        expected_components = [
            Stem("bund", span=(0, 4)),
            Link("_+s_", realization="es", span=(4, 6), type="addition"),
            Stem("arzt", realization="ärzt", span=(6, 10)),
            Link("_+=e_", realization="e", span=(10, 11), type="addition_umlaut"),
            Stem("kammer", span=(11, 17))
        ]
        actual_components = compound.components
        self.assertListEqual(expected_components, actual_components)
        # compare pretty form
        expected_lemma = "bundesärztekammer"
        actual_lemma = compound.lemma
        self.assertEqual(expected_lemma, actual_lemma)


if __name__ == '__main__':
    unittest.main()