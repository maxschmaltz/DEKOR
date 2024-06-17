"""
Module for parsing the COW compounds dataset.
"""

import re
from dataclasses import dataclass, field
import random
import pandas as pd
from typing import Tuple, List, Optional, Union


DE = '[a-zäöüß]+'   # German alphabet
LINK_TYPES = {
    # more complex types come first so that types that use subpatterns
    # are not matched prematurely;
    # e.g., _(x)_+xx_ should come before _+xx_ because otherwise
    # _(x)_+xx_ type links will be defined as _+xx_ ones
    # this one is not a link, but will be added here for further processing
    "infix": f'(_?{DE}~_?)',                              # Arbeit_+s_un~_Fähigkeit = Arbeits(un)fähigkeit
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
LINK = '|'.join(LINK_TYPES.values())    # any link
UMLAUTS = {
    'au': 'äu',
    'a': 'ä',
    'o': 'ö',
    'u': 'ü'
}
UMLAUTS_REVERSED = {v: k for k, v in UMLAUTS.items()}


def get_span(string: str, range: Tuple[int]) -> str:    # get a span of a match
    start, end = range
    span = string[start: end]
    return span


@dataclass
class Stem:

    """
    A stem component of a `Compound`.

    Attributes
    ----------
    component : `str`
        abstract representation of the component

    realization : `str`, optional
        concrete realization of the component; if not given, equals to `component`

    span : `Tuple[int]`
        span of the component realization in the `Compound` lemma

    is_noun : `bool`
        whether the component is a nominal stem

    Example
    -------
    >>> compound = Compound("Schule_-e_Jahr")
    >>> compound.components[0].component
    "schule"
    >>> compound.components[0].realization
    "schul"
    >>> compound.components[0].span
    (0, 5)
    >>> compound.components[0].is_noun
    True
    """

    component: str = field(compare=True)
    realization: Optional[str] = field(compare=False, default=None)
    span: Tuple[int] = field(compare=False, kw_only=True)
    is_noun: bool = field(compare=False, kw_only=True)
    # further features

    def __post_init__(self) -> None:
        if not self.realization:
            self.realization = self.component

    def __repr__(self) -> str:
        return self.component


@dataclass
class Link:

    """
    A link component of a `Compound`.

    Attributes
    ----------
    component : `str`
        abstract representation of the component

    span : `Tuple[int]`
        span of the component in the `Compound` lemma

    type : `str`
        type of the link according to the COW dataset

    Example
    -------
    >>> compound = Compound("Schule_-e_Jahr")
    >>> compound.components[1].component
    "e"
    >>> compound.components[0].span
    (5, 5)
    >>> compound.components[0].type
    "deletion_nom"
    """

    component: str = field(compare=True)
    span: Tuple[int] = field(compare=False, kw_only=True)
    type: str = field(compare=True, kw_only=True)
    # further features

    @classmethod
    def empty(cls):

        """
        Creates an empty link.
        """

        return cls("", span=(-1, -1), type="none")

    def __repr__(self) -> str:
        return self.component


@dataclass
class Compound:

    """
    Class for representation of a COW compound entry.

    Parameters
    ----------
    raw : `str`
        COW dataset entry to analyze

    Attributes
    ----------
    lemma : `str`
        recovered lemma representation of the compound

    stems : `List[Stem]`
        list of the stems of the compound

    links : `List[Link]`
        list of the links of the compound
        
    components : `List[Union[Stem, Link]]`
        list of the stems and the links of the compound, sorted by span

    Example
    -------
    >>> compound = Compound("Schule_-e_Jahr")
    >>> compound.raw
    "Schule_-e_Jahr"
    >>> compound.lemma
    "Schuljahr"
    >>> compound.components
    ["schule", "e", "jahr"]
    """

    raw: str = field(compare=False)
    lemma: str = field(compare=True, init=False)
    stems: List[Stem] = field(compare=False, init=False)
    links: List[Link] = field(compare=False, init=False)
    components: List[Union[Stem, Link]] = field(compare=True, init=False)

    def __post_init__(self) -> None:
        self._analyze(self.raw) # defines .stems, .links, .components, .lemma

    def _get_stem_obj(self, component: str) -> Stem:
        is_noun = component[0].isupper()
        component = self.infix + component  # add cumulative infix if applicable
        stem = Stem(
            component=component.lower(),
            # realization as the component at first, will be modified later if needed
            span=(self.i, self.i + len(component)),
            is_noun=is_noun
        )
        self.infix = "" # reset infix
        self.i += len(component)
        return stem

    def _get_link_objs(self, component: str) -> List[Link]:
        # for simpler processing, some of the complex link types will be split into
        # the components, e.g. "replacement" is actually "deletion_nom" followed by "addition"
        for link_type, pattern in LINK_TYPES.items():
            match = re.match(
                pattern.replace(DE, f'({DE})'), # add parenthesis to pattern to capture links
                component,
                flags=re.I
            )
            if match:
                match link_type:
                    case "addition_with_expansion":
                        # using capturing groups in patterns to retrieve links
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
                        # skip links in lemma
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
                        self.i -= len(del_component)    # link will be subtracted from previous stem in fusion
                        links = [
                            Link(
                                component=del_component, # deletion part
                                span=(self.i, self.i),  # length of 0
                                type="deletion_nom"
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
                    case s if "deletion" in s:  # "deletion_nom" | "deletion_non_nom"
                        del_component = get_span(component, match.regs[-1])
                        self.i -= len(del_component)    # link will be subtracted from previous stem in fusion
                        links = [
                            Link(
                                component=del_component,
                                span=(self.i, self.i),  # length of 0
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
    
    def _fuse_links(self, links: List[Link]):
        # build the lemma by fusing currently processed part with incoming links;
        # that includes, for example, umlaut and deletion processing, concatenation and so on
        if self.stems: last_stem = self.stems[-1]
        for link in links:
            match link.type:
                case "expansion" | "addition" | "infix":
                    self.lemma += link.component
                case s if "deletion" in s:
                    self.lemma = re.sub(f'{link.component}$', '', self.lemma, count=1, flags=re.I)
                    if last_stem:
                        # adjust realization: clip the deleted part
                        last_stem.realization = re.sub(
                            f'{link.component}$',
                            '',
                            last_stem.component,
                            count=1,
                            flags=re.I
                        )
                        # adjust span accordingly
                        last_stem.span = (last_stem.span[0], last_stem.span[1] - len(link.component))
                case "hyphen":
                    self.lemma += '-'
                case "umlaut":
                    # search for the closest to the link "umlautable" vowel
                    match = re.search('(au|a|o|u)[^aou]+$', self.lemma, flags=re.I)
                    if match:
                        # the whole suffix containing the vowel
                        suffix_before_umlaut = get_span(self.lemma, match.regs[0])
                        # the vowel itself
                        umlaut = get_span(self.lemma, match.regs[1])
                        # perform umlaut in the suffix
                        suffix_after_umlaut = re.sub(
                            umlaut,
                            UMLAUTS[umlaut],
                            suffix_before_umlaut,
                            flags=re.I
                        )
                        # replace suffix before umlaut with converted suffix
                        self.lemma = re.sub(
                            f'{suffix_before_umlaut}$',
                            suffix_after_umlaut,
                            self.lemma,
                            count=1,
                            flags=re.I
                        )
                        # adjust realization: perform umlaut
                        if last_stem:
                            last_stem.realization = re.sub(
                                f'{suffix_before_umlaut}$',
                                suffix_after_umlaut,
                                last_stem.component,
                                count=1,
                                flags=re.I
                            )
                case _: # "concatenation"
                    pass

    def _analyze(self, gecodb_compound: str) -> None:
        self.i = 0  # global scope of index to be available from anywhere in the class
        self.infix = "" # global scope of infix to be available from anywhere in the class
        self.stems, self.links = [], []
        components = re.split(LINK, gecodb_compound, flags=re.I)    # split by links; capturing groups will store the links
        self.lemma = ""
        for component in components:
            if not component: continue  # `None` from capturing group occasionally occurs
            if "~" in component:
                match = re.match(
                    LINK_TYPES['infix'].replace(DE, f'({DE})'), # add parenthesis to pattern to capture infix
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
                    # remove from raw
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
        self.components = sorted(self.stems + self.links, key=lambda c: c.span) # sort by span => order of appearance
        self.lemma = self.lemma.capitalize()

    def __repr__(self) -> str:
        return f'{self.lemma} <-- {self.raw}'
    

def parse_gecodb(gecodb_path: str, min_count: Optional[int]=25) -> pd.DataFrame:

    """
    Parse the COW dataset.

    Parameters
    ----------
    gecodb_path : `str`
        path to the TSV COW dataset

    min_count : `int`, optional, defaults to 25
        minimal count of compounds to keep; all compounds occurring less will be dropped

    Returns
    -------
    `pandas.DataFrame`
        dataframe with three columns:
        * "raw": `str`: COW dataset entry
        * "count": `int`: number of occurrences
        * "compound": `Compound`: processed compounds
    """

    gecodb = pd.read_csv(
        gecodb_path,
        sep='\t',
        names=['raw', 'count'],
        encoding='utf-8'
    )
    if min_count: gecodb = gecodb[gecodb['count'] >= min_count]
    gecodb['compound'] = gecodb['raw'].apply(Compound)
    return gecodb