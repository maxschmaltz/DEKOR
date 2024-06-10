import re
from dataclasses import dataclass
import random
import pandas as pd
from typing import Tuple


DE = '[a-zäöüß]+'
LINK_TYPES = {
    # more complex types come first so that types that use subpatterns
    # are not matched prematurely;
    # e.g., _(x)_+xx_ should come before _+xx_ because otherwise
    # _(x)_+xx_ type links will be defined as _+xx_ ones
    # this one is not a link, but will be added here for further processing
    "infix": f'({DE}~_)',                                    # Arbeit_+s_un~_Fähigkeit = Arbeits(un)fähigkeit
    # links
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
UMLAUTS_REVERSED = {v: k for k, v in UMLAUTS.items()}


def get_span(string, range):
    start, end = range
    span = string[start: end]
    return span


@dataclass
class Stem:

    component: str
    realization: str
    span: Tuple[int]
    is_noun: bool
    # further features

    def __repr__(self) -> str:
        return self.component


@dataclass
class Link:

    component: str
    span: Tuple[int]
    type: str
    # further features

    @classmethod
    def empty(cls):
        return cls("", (-1, -1), "none")

    def __repr__(self) -> str:
        return self.component


@dataclass
class Compound:

    def __init__(self, gecodb_entry):
        self.raw = gecodb_entry
        self._analyze(gecodb_entry) # adds .stems, .links, .components, .lemma

    def _get_stem_obj(self, component):
        is_noun = component[0].isupper()
        component = self.infix + component  # add cumulative infix if applicable
        stem = Stem(
            component=component.lower(),
            realization=component.lower(),
            span=(self.i, self.i + len(component)),
            is_noun=is_noun
        )
        self.infix = ""
        self.i += len(component)
        return stem

    def _get_link_objs(self, component):
        for link_type, pattern in LINK_TYPES.items():
            match = re.match(
                pattern.replace(DE, f'({DE})'), # add parenthesis to pattern to capture links
                component,
                flags=re.I
            )
            if match:
                match link_type:
                    case "addition_with_expansion":
                        exp_component = get_span(component, match.regs[-2])
                        add_component = get_span(component, match.regs[-1])
                        links = [
                            Link(
                                component=exp_component,    # expansion part
                                span=(self.i, self.i + len(exp_component)),
                                type="expansion"    # TODO: "addition"?
                            ),
                            Link(
                                component=add_component,    # addition part
                                span=(self.i + len(exp_component), self.i + len(exp_component) + len(add_component)),
                                type="addition"
                            )   
                        ]
                        self.i += (len(exp_component) + len(add_component))
                        break
                    case "addition_with_umlaut":
                        add_component = get_span(component, match.regs[-1])
                        links = [
                            Link(
                                component="",   # umlaut part
                                span=(self.i, self.i),  # length of 0
                                type="umlaut"
                            ),
                            Link(
                                component=add_component, # addition part
                                span=(self.i, self.i + len(add_component)),
                                type="addition"
                            )   
                        ]
                        self.i += len(add_component)
                        break
                    case "replacement":
                        del_component = get_span(component, match.regs[-2])
                        add_component = get_span(component, match.regs[-1])
                        self.i -= len(del_component)    # will be subtracted from previous stem in fusion
                        links = [
                            Link(
                                component=del_component, # deletion part
                                span=(self.i, self.i),  # length of 0
                                type="deletion"
                            ),
                            Link(
                                component=add_component, # addition part
                                span=(self.i, self.i + len(add_component)),
                                type="addition"
                            )   
                        ]
                        self.i += len(add_component)
                        break
                    case "addition":
                        add_component = get_span(component, match.regs[-1])
                        links = [
                            Link(
                                component=add_component,
                                span=(self.i, self.i + len(add_component)),
                                type="addition"
                            )
                        ]
                        self.i += len(add_component)
                        break
                    case r"deletion.+":
                        del_component = get_span(component, match.regs[-1])
                        self.i -= len(del_component)    # will be subtracted from previous stem in fusion
                        links = [
                            Link(
                                component=del_component,
                                span=(self.i, self.i),  # length of 0
                                # type="deletion"
                                type=link_type
                            )
                        ]
                        break
                    case "hyphen":
                        links = [
                            Link(
                                component="-",
                                span=(self.i, self.i + 1),  # length of 1
                                type="hyphen"
                            )
                        ]
                        self.i += 1
                        break
                    # will be processed in fusion
                    case "infix":
                        links = []
                        break
                    case _: # "umlaut", "concatenation"
                        links = [
                            Link(
                                component="",
                                span=(self.i, self.i),  # length of 0
                                type=link_type
                            )
                        ]
                        break
        else: links = []
        return links  
    
    def _fuse_links(self, links):
        if self.stems: last_stem = self.stems[-1]
        for link in links:
            match link.type:
                case "expansion" | "addition" | "infix":
                    self.lemma += link.component
                case r"deletion.+":
                    self.lemma = re.sub(f'{link.component}$', '', self.lemma, count=1, flags=re.I)
                    if last_stem: 
                        last_stem.realization = re.sub(f'{link.component}$', '', last_stem.component, count=1, flags=re.I)
                        last_stem.span = (last_stem.span[0], last_stem.span[1] - len(link.component) )
                case "hyphen":
                    self.lemma += '-'
                case "umlaut":
                    match = re.search('(au|a|o|u)[^aou]+$', self.lemma, flags=re.I)
                    if match:
                        suffix_before_umlaut = get_span(self.lemma, match.regs[0])
                        umlaut = get_span(self.lemma, match.regs[1])
                        suffix_after_umlaut = re.sub(
                            umlaut,
                            UMLAUTS[umlaut],
                            suffix_before_umlaut,
                            flags=re.I
                        )
                        self.lemma = re.sub(
                            f'{suffix_before_umlaut}$',
                            suffix_after_umlaut,
                            self.lemma,
                            flags=re.I
                        )
                        if last_stem:
                            last_stem.realization = re.sub(
                                f'{suffix_before_umlaut}$',
                                suffix_after_umlaut,
                                last_stem.component,
                                flags=re.I
                            )
                case _: # "concatenation"
                    pass

    def _analyze(self, gecodb_compound):
        self.i = 0  # global scope of index
        self.infix = "" # global scope of infix
        self.stems, self.links = [], []
        components = re.split(LINK, gecodb_compound, flags=re.I)
        self.lemma = ""
        # infix = ""
        for component in components:
            if not component: continue  # None from capturing group
            if "~" in component:
                match = re.match(
                    LINK_TYPES["infix"].replace(DE, f'({DE})'), # add parenthesis to pattern to capture infix
                    component,
                    flags=re.I
                )
                # Infixes are optional and do not convey much useful information:
                # ein~_Familie_+n_Haus is the same for us as Familie_+n_Haus;
                # however, if we completely ignore them, we might accidentally train the model
                # to not detect them as a normal part of a compound but rather as an anomaly
                # (because they'd have never seen those) so we might get inadequate analysis
                # of compounds with such infixes. To prevent that, we will
                # eliminate them in 60% cases and merge them into the stem otherwise  
                self.infix = "" if random.random() >= 0.4 else get_span(component, match.regs[-1])
                if not self.infix:
                    self.raw = self.raw.replace(component, "", 1)
            if not '_' in component:    # stem
                stem = self._get_stem_obj(component)
                self.lemma += stem.component # accumulative fusion into the original lemma
                self.stems.append(stem)
            else:
                links = self._get_link_objs(component)
                self._fuse_links(links) # accumulative fusion into the original lemma
                if not (len(links) == 1 and links[0].type == "infix" and not links[0].component): # if not eliminated infix
                    self.links += links
        del self.i; del self.infix
        self.components = sorted(self.stems + self.links, key=lambda c: c.span)
        self.lemma = self.lemma.capitalize()

    def __repr__(self) -> str:
        return f'{self.lemma} <-- {self.raw}'
    

def parse_gecodb(gecodb_path, min_freq=25):
    gecodb = pd.read_csv(
        gecodb_path,
        sep='\t',
        names=['raw', 'freq'],
        encoding='utf-8'
    )
    if min_freq: gecodb = gecodb[gecodb['freq'] >= min_freq]
    gecodb['compound'] = gecodb['raw'].apply(Compound)
    return gecodb