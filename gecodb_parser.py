import re
from dataclasses import dataclass


DE = '[a-zäöüß]+'
LINK_TYPES = {
    # more complex types come first so that types that use subpatterns
    # are not matched prematurely;
    # e.g., _(x)_+xx_ should come before _+xx_ because otherwise
    # _(x)_+xx_ type links will be defined as _+xx_ ones
    "addition_with_expansion": f'(_\({DE}\)_\+{DE}_)',    # Bau_(t)_+en_Schutz = Bautenschutz
    "addition_with_umlaut": f'(_\+={DE}_)',               # Gast_+=e_Buch = Gästebuch
    "replacement": f'(_\-{DE}_\+{DE}_)',                  # Hilfe_-e_+s_Mittel = Hilfsmittel
    "addition": f'(_\+{DE}_)',                            # Bund_+es_Land = Bundesland
    "deletion_nom": f'(_\-{DE}_)',                        # Schule_-e_Jahr = Schuljahr
    "deletion_non_nom": f'(_#{DE}_)',                     # suchen_#en_Maschine = Suchmaschine
    "umlaut": '(_\+=_)',                                  # Mutter_+=_Rente = Mütterrente
    "hyphen": '(_--_)',                                   # Online_--_Shop = Online-shop
    "concatenation": '(_)'                                # Zeit_Punkt = Zeitpunkt
}
LINK = '|'.join(LINK_TYPES.values())
UMLAUTS = {
    'au': 'äu',
    'a': 'ä',
    'o': 'ö',
    'u': 'ü'
}


def get_span(string, range):
    start, end = range
    span = string[start: end]
    return span


@dataclass
class Stem:

    component: str
    index: int
    is_noun: bool
    # further features

    def __repr__(self) -> str:
        return self.component


@dataclass
class Link:

    component: str
    index: int
    type: str
    # further features

    def __repr__(self) -> str:
        return self.component


@dataclass
class Compound:

    def __init__(self, gecodb_entry):
        self.raw = gecodb_entry
        self._analyze(gecodb_entry) # adds .stems, .links, .components

    def _get_stem_obj(self, component):
        stem = Stem(
            component=component.lower(),
            index=self.i,
            is_noun=component[0].isupper()
        )
        return stem

    def _get_link_objs(self, component):
        for link_type, pattern in LINK_TYPES.items():
            match = re.match(pattern.replace(DE, f'({DE})'), component) # add parenthesis to pattern to capture links
            if match:
                match link_type:
                    case "addition_with_expansion":
                        links = [
                            Link(
                                component=get_span(component, match.regs[-2]), # expansion part
                                index=self.i,
                                type="expansion"    # "addition"?
                            ),
                            Link(
                                component=get_span(component, match.regs[-1]), # addition part
                                index=self.i + 1,
                                type="addition"
                            )   
                        ]
                        self.i += 1
                        break
                    case "addition_with_umlaut":
                        links = [
                            Link(
                                component="",   # umlaut part
                                index=self.i,
                                type="umlaut"
                            ),
                            Link(
                                component=get_span(component, match.regs[-1]), # addition part
                                index=self.i + 1,
                                type="addition"
                            )   
                        ]
                        self.i += 1
                        break
                    case "replacement":
                        links = [
                            Link(
                                component=get_span(component, match.regs[-2]), # deletion part
                                index=self.i,
                                type="deletion"
                            ),
                            Link(
                                component=get_span(component, match.regs[-1]), # addition part
                                index=self.i + 1,
                                type="addition"
                            )   
                        ]
                        self.i += 1
                        break
                    case "addition":
                        links = [
                            Link(
                                component=get_span(component, match.regs[-1]),
                                index=self.i,
                                type="addition"
                            )
                        ]
                        break
                    case "deletion_nom" | "deletion_non_nom":
                        links = [
                            Link(
                                component=get_span(component, match.regs[-1]),
                                index=self.i,
                                type="deletion"
                            )
                        ]
                        break
                    case "hyphen":
                        links = [
                            Link(
                                component="-",
                                index=self.i,
                                type="hyphen"
                            )
                        ]
                        break
                    case _: # "umlaut", "concatenation"
                        links = [
                            Link(
                                component="",
                                index=self.i,
                                type=link_type
                            )
                        ]
                        break
        else: links = []
        return links  

    def _fuse_links(self, links):
        for link in links:
            match link.type:
                case "expansion" | "addition":
                    self.lemma += link.component
                case "deletion":
                    self.lemma = re.sub(f'{link.component}$', '', self.lemma, count=1)
                case "hyphen":
                    self.lemma += '-'
                case "umlaut":
                    last_suffix_with_umlaut = re.search('(au|a|o|u)[^aou]+$', self.lemma, flags=re.I)
                    if last_suffix_with_umlaut:
                        suffix_before_umlaut = get_span(self.lemma, last_suffix_with_umlaut.regs[0])
                        umlaut = get_span(self.lemma, last_suffix_with_umlaut.regs[1])
                        suffix_after_umlaut = re.sub(
                            umlaut,
                            UMLAUTS[umlaut],
                            suffix_before_umlaut
                        )
                        self.lemma = re.sub(f'{suffix_before_umlaut}$', suffix_after_umlaut, self.lemma)
                case _: # "concatenation"
                    pass

    def _analyze(self, gecodb_compound):
        self.i = 0  # global scope of index
        self.stems, self.links = [], []
        components = re.split(LINK, gecodb_compound, flags=re.I)
        self.lemma = ""
        for component in components:
            if not component: continue  # None from capturing group
            if not '_' in component:    # stem
                stem = self._get_stem_obj(component)
                self.stems.append(stem)
                self.lemma += component # accumulative fusion into the original lemma
            else:
                links = self._get_link_objs(component)
                self.links += links
                self._fuse_links(links) # accumulative fusion into the original lemma
            self.i += 1
        del self.i
        self.components = sorted(self.stems + self.links, key=lambda c: c.index)
        self.lemma = self.lemma.capitalize()

    def __repr__(self) -> str:
        return f'{self.lemma} <-- {self.raw}'

# Stute_+n_Milch_trinken_#en_Kur        6406505




# def read_data(gecodb_path, min_freq=None):
#     gecodb = pandas.read_csv(
#         gecodb_path,
#         sep='\t',
#         names=['comp', 'freq'],
#         encoding='utf-8'
#     )
#     if min_freq is not None:
#         gecodb[gecodb['freq'] >= min_freq]
#    return gecodb