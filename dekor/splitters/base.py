"""
Base model for splitting German compounds based on the DECOW16 compound data.
"""

import re
from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Iterable, Iterator, Self, Tuple, List, Dict

from dekor.utils.gecodb_parser import Compound, Link, UMLAUTS
from dekor.utils.vocabs import StringVocab, UNK

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseSplitter(ABC):

    """
    Base class for splitters.
    """

    # caching preprocessed data is impossible because it gets shuffled on every iteration

    name: str
    record_none_links: bool # enforce
    eliminate_allomorphy: bool  # enforce
    vocab_links: StringVocab
    _elink = Link(  # for analyzing positions
        UNK,
        span=(-1, -1),
        type=UNK
    )

    def _metadata(self) -> dict:
        # for parameter tracking
        return {
            "record_none_links": self.record_none_links,
            "eliminate_allomorphy": self.eliminate_allomorphy
        }

    def _get_positions(
        self,
        compound: Compound,
        context_window: int
    ) -> Iterator[Tuple[Tuple[str], Compound]]:

        # Analyze a single compound; performed as a sliding window
        # with a sliding window inside over the compound lemma, where for each position it is stored,
        # which left and right context in n-grams there is and what is in between and
        # whether that "in between" is a link and, if yes, which one.
        # Example:
        #   "bundestag" with 2-grams
        #   ">b", "un", "" --> no link
        #   ">b", "nd", "u" --> no link
        #   ">b", "de", "un" --> no link
        #   ...
        #   "nd", "es", "" --> no link
        #   "nd", "st", "e" --> no link
        #   "nd", "ta", "es" --> link "_+s_"
        #   ...
        # Later, uniques context-link triples will be used in prediction.

        lemma = f'>{compound.lemma}<'    # BOS and EOS
        n = context_window  # shorthand
        l = len(lemma) - 1  # -1 because indexing starts at 0

        # as we know which links to expect, we will track them 
        next_link_idx = 0
        # Masks will be of a form (c_l, c_r, c_m, l), where
        #   * c_l is the left n-gram
        #   * c_r is the right n-gram
        #   * c_m is the middle n-gram
        #   * l is the link id (unknown id if none)
        # Then this masks can be used differently dependent on the splitter

        # Make sliding window; however, we want to start not directly with
        # n-grams, but first come from 1-grams to n-grams at the left of the compound
        # and then slide by n-grams; same with the end: not the last n-gram,
        # but n-gram to 1-gram. To be more clear: having 'Bundestag' and 3-grams, we don't want contexts
        # to be directly (("bun", "des"), ("und", "est"), ..., ("des", "tag")), 
        # but we rather want (("b", "und"), ("bu", "nde"), ("bun", "des"), ..., ("des", "tag"), ("est", "ag"), ("sta", "g")).
        # As we process compounds unidirectionally and move left to right,
        # we want subtract max n-gram length to achieve this effect; thus, with a window of length
        # max n-gram length, we will begin with 1-grams, ..., reach n-grams, ..., and end with ..., 1-grams
        for i in range(1 - n + 1, l - n):  # 1 from both sides because there can not be a link right after BOS
            masks = []
            # next expected link; we use empty link in case there are no links anymore to unify the workflow below
            next_link = compound.links[next_link_idx] if next_link_idx < len(compound.links) else self._elink
            s = max(0, i)   # start of left
            m = i + n   # end of left = start of mid
            # break cycle if case left forces the right to be the single EOS
            # which it makes no sense to record because link can not appear there or any further
            if m > l - 1: break # -1 for special symbols
            for r in range(4):  # max length of a link representation is 3 as in -ens-
                e = m + r   # end of mid = start of right
                f = m + r + n   # end of right
                # break cycle if case right context is going to be the single EOS
                if e > l - 1: break   # -1 for special symbols
                left = lemma[s: m]
                mid = lemma[m: e]
                right = lemma[e: f]
                # define if there is a link incoming at this index;
                # because of such implementation, mid will always be the same
                # with the link realization
                if (m - 1, e - 1) == next_link.span:    # -1 is correction because of special symbols
                    link = next_link
                    # increment
                    next_link_idx += 1
                else:
                    # the idea behind recording none links is that it will give us
                    # the proportion between a link and no link on a certain position;
                    # if not recorded, 5 occurrences of link A will be the same
                    # as 25 occurrences of link B in different positions (if no other
                    # links have been encountered), but with none links recorded,
                    # 995-5 will be less probable than 975-25
                    if self.record_none_links:
                        link = self._elink
                    else: continue
                masks.append(((left, right, mid), link))
            # This will be a generator because all of the splitters will have
            # an additional processing of masks so generating will prevent double iteration
            # over the same list; moreover, this is needed in prediction for some splitters
            # to be able to "meddle" in the middle of processing.
            # Also, we will yield masks in packs, because in further processing we will
            # need to define the most probable link for a single position
            # and not all the windows from this one position;
            # for example, in "bundestag" we want to predict not 4 windows separately as in
            # ("und", "", "est"), ("und", "e", "sta"), ("und", "es", "tag") ...
            # but rather whether there is a link after "und" given all windows.
            yield masks

    @abstractmethod
    def fit(self, compounds: Iterable[Compound], *args, **kwargs) -> Self:
        pass

    def _predict(
        self,
        lemma: str,
        logits: np.ndarray,
        link_candidates: Dict[int, Tuple[int, str]]
    ) -> Compound:

        # For a single lemma, this function gets a list of link candidates,
        # i.e. each (possibly link-) ngram at each lemma index, and
        # respective to the candidates probability distribution
        # over links, so which link (or none) this ngram is with which prob.
        # This info is used to determine the most probable link and its position
        # which will satisfy the grammar constrains.
        # We know for sure that we works with N+N compounds which means
        # we can heuristically restrict all improbable additional
        # links it will predict. Thus, for the compound, we will
        # gather probabilities of all non-none links with respect to their
        # positions and then choose the most probable one.
            
        # at this point, none links have done their job and gave
        # the proportion (if they were even recorded) so
        # we should zero them to easily detect the most probable
        # non-none link
        logits[:, self.vocab_links.unk_id] = 0

        # There is a set of heuristics that have to be filtered out in place
        # in order to get cleaner result; for example, there can not be an umlaut link
        # in case there is no umlaut before it. This can mostly be checked once a link with
        # its representation is parsed; however, it is highly inefficient to do that
        # will all links whose probability is more that 0. Instead, we will treat the
        # probabilities as a stack with probs ordered from highest to lowest;
        # thus, at each iteration, we will check if the current max probable link
        # passes the filter and if yes, we'll break the cycle, if no, zero this prob
        # and take the next highest probable one. Thus, at the end we will output
        # the most probable link of all that passed the filter.  
        while True:

            # it cannot exceed because even if all non-zero links will appear invalid,
            # there will be no probs anymore and the function will exit

            max_prob = logits.max()
            if not max_prob:
                return Compound(lemma)   # no link detected, so return with no links
            
            best_idx, best_link_id = np.where(logits == max_prob)
            best_idx, best_link_id = best_idx[0], best_link_id[0]   # unpack
            i, best_realization = link_candidates[best_idx]
            best_link = self.vocab_links.decode(best_link_id)
            component, realization, link_type = Compound.get_link_info(
                best_link,
                eliminate_allomorphy=self.eliminate_allomorphy
            )

            raw = lemma[:i] # left constituent

            # heuristically filter out predictions that cannot be correct
            # using `if` so that no further checks are performed once one has failed
            if (    # use if so that no further checks are performed once one has failed
                # deletion type with addition realization
                (
                    link_type == "deletion"
                    and len(best_realization) > 0
                ) or
                # concatenation type with addition realization
                (
                    link_type == "concatenation"
                    and len(best_realization) > 0
                ) or
                # impossible addition; there might be or not be an e-, depends on whether we eliminate allomorphy
                (
                    "addition" in link_type and
                    not re.match(f"^e?{realization}$", best_realization)
                ) or
                # umlaut link where there is no rightmost (!) umlaut before
                (
                    "umlaut" in link_type and
                    not re.search(f"({'|'.join(UMLAUTS.values())})(?!.+({'|'.join(UMLAUTS.keys())}))", raw)
                    # not re.search('|'.join(UMLAUTS_REVERSED.keys()), raw)
                )
            ):
                # zero this impossible link prob
                logits[best_idx, best_link_id] = 0
                continue

            # NOTE: reconstruct from components?
            # If allomorphy is eliminated, we can predict cases like
            # tag_+s_ticket correctly and know that the realization is -es-;
            # however, since we dont reconstruct the compound from components,
            # if we pass tag_+s_ticket to `Compound`, it will think that the
            # realization is -s- even though we know it is not the case.
            # That is why, if eliminated allomorphy encountered,
            # we must reconstruct the link as if allomorphy does not get eliminated,
            # and then `Compound` will still parse the link with elimination
            # but will receive the correct realization we predicted.
            if link_type == "addition_umlaut":
                raw = Compound.reverse_umlaut(raw)
            elif link_type == "deletion":
                to_delete = Compound.get_deletion(component)
                raw += to_delete
            if best_realization != realization:
                component = re.sub(realization, best_realization, component)

            raw = raw + component + lemma[i + len(best_realization):]
            pred = Compound(raw)
            return pred

    @abstractmethod
    def predict(self, lemmas: List[str], *args, **kwargs) -> List[Compound]:
        pass

    def __repr__(self) -> str:
        return str(self._metadata())