"""
N-gram model for splitting German compounds based on the DECOW16 compound data.
"""

import re
import random
import numpy as np
from tqdm import tqdm
from typing import Iterable, Optional, List, Tuple, Self

from dekor.splitters.base import BaseSplitter
from dekor.utils.gecodb_parser import (
    Compound,
    Link,
    UMLAUTS_REVERSED
)
from dekor.utils.vocabs import StringVocab


class NGramsSplitter(BaseSplitter):

    """
    N-grams-based compound splitter that relies on the COW dataset format.
    First, fits train COW entries, then predicts lemma splits in this format.

    Parameters
    ----------
    n : `int`, optional, defaults to `2`
        maximum n-grams length

    record_none_links : `bool`, optional, defaults to `False`
        whether to record contexts between which no links occur;
        hint: that could lead to a strong bias towards no link choice

    eliminate_allomorphy : `bool`, optional, defaults to `True`
        whether to eliminate allomorphy of the input link, e.g. _+es_ to _+s_

    verbose : `bool`, optional, defaults to `True`
        whether to show progress bar when fitting and predicting compounds
    """

    name = "ngrams"

    def __init__(
            self,
            n: Optional[int]=2,
            record_none_links: Optional[bool]=False,
            eliminate_allomorphy: Optional[bool]=True,
            verbose: Optional[bool]=True
        ) -> None:
        self.n = n
        self.record_none_links = record_none_links
        self.eliminate_allomorphy = eliminate_allomorphy
        self.verbose = verbose
        self.vocab_links = StringVocab()
        self.vocab_positions = StringVocab()
        self._elink = Link(
            self.vocab_links.unk,
            span=(-1, -1),
            type=self.vocab_links.unk
        )

    def _metadata(self) -> dict:
        return {
            "n": self.n,
            "record_none_links": self.record_none_links,
            "eliminate_allomorphy": self.eliminate_allomorphy
        }

    def _forward(self, compound: Compound) -> List[Tuple[int]]:

        # Analyze a single compound; performed as a sliding window
        # with a sliding window inside
        # over the compound lemma, where for each position it is stored,
        # which left and right context in n-grams there is and what is in between and
        # whether that "in between" is and, if, yes, which one.
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
        # Later, uniques context-link triples will be counted
        # and this counts information will be used in prediction.
        lemma = f'>{compound.lemma}<'    # BOS and EOS
        n = self.n
        l = len(lemma) - 1  # -1 because indexing starts at 0

        # as we know which links to expect, we will track them 
        next_link_idx = 0
        # masks will be of a form (c_l, c_r, c_m, l), where
        #   * c_l is the left n-gram
        #   * c_r is the right n-gram
        #   * c_m is the middle n-gram
        #   * l is the link id (unknown id if none)
        masks = []
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
                    if self.record_none_links: link = self._elink
                    else: continue
                # A mask has a form (c_l, c_r, c_m, l), where
                #   * c_l is the left n-gram
                #   * c_r is the right n-gram
                #   * c_m is the mid n-gram
                #   * l is the link id (unknown link id if none)
                # We want to get a mapping from <c_l, c_r, c_m> of contexts C = (c1, c2, ...)
                # to distribution of link ids L = (l1, l2, ...).
                # If we do that with matrices, they become too large to operate over because they become extremely sparse
                # and take up the whole memory (they are also 4D matrices which makes it worse);
                # that is why we need a more compact way to encode positions.
                # What we can do is to encode positions <c_l, c_r, c_m> separately as strings and thus reduce
                # the 4D matrix of shape (|C|, |C|, |C|, |L|) to a 2D matrix of shape (|P|, |L|),
                # where P is the set of encoded positions. Moreover, we will this significantly
                # reduce the sparseness of the matrix because there will be no unknowns contexts
                # so no zeros between n-grams, only zeros on L distribution when a link never occurs in a position.
                # For that, we join the contexts into a single position.
                link_id = self.vocab_links.add(link.component)
                position = '|'.join([left, right, mid])
                position_id = self.vocab_positions.add(position)
                masks.append((position_id, link_id))

        return masks

    def fit(self, compounds: Iterable[Compound]) -> Self:

        """
        Feed DECOW16 compounds to the model. That includes iterating through each compound
        with a sliding window and counting occurrences of links between n-gram contexts
        to try to fit to the target distribution.

        Parameters
        ----------
        compounds : `Iterable[Compound]`
            collection of `Compound` objects out of COW dataset to fit

        Returns
        -------
        `NGramsSplitter`
            fit model
        """

        all_masks = []
        progress_bar = tqdm(compounds, desc="Fitting") if self.verbose else compounds
        for compound in progress_bar:
            # collect masks from a single compound
            masks = self._forward(compound)
            all_masks += masks
        all_masks = np.array(all_masks, dtype=np.int32)

        # count unique position --> link id mapping entries
        _, unique_indices, counts = np.unique(all_masks, return_index=True, return_counts=True, axis=0)
        # `np.unique()` sorts masks so the original order is not preserved;
        # to recreate it, the method returns indices of masks in the original dataset
        # in order of their counts; for example, `unique_indices[5]` contains
        # the index of that mask in `masks_links` whose count is under `counts[5]`
        unique_masks_links = all_masks[unique_indices]
        positions_ids = unique_masks_links[:, 0]
        link_ids = unique_masks_links[:, 1]
        # heuristically force <unk> link from <unk> context
        positions_ids = np.append(positions_ids, self.vocab_positions.unk_id)
        link_ids = np.append(link_ids, self.vocab_links.unk_id)
        counts = np.append(counts, 1)

        n_pos, n_links = len(self.vocab_positions), len(self.vocab_links)
        counts_links = np.zeros((n_pos, n_links), dtype=np.float32)  #f or further division
        counts_links[positions_ids, link_ids] = counts

        # counts to frequencies; since all positions point at least to <unk>,
        # no zero division will be encountered
        counts_links /= counts_links.sum(axis=1, dtype=np.float32).reshape(
            (n_pos, 1)
        )
        # rewrite <unk> link from <unk> context close to 0 so that
        # if there is an actual data available, it is preferred
        counts_links[self.vocab_positions.unk_id, self.vocab_links.unk_id] = 1e-10
        self.freqs_links = counts_links

        return self
    
    def _predict(self, lemma: str) -> Compound:

        # predict a single lemma and return a DECOW16-format `Compound`
        raw = ""    # will iteratively restore DECOW16-format raw compound
        lemma = f'>{lemma.lower()}<'        # BOS and EOS
        n = self.n
        l = len(lemma) - 1  # -1 because indexing starts at 0
        c = 0   # correction to skip links (see below)

        # same sliding window
        for i in range(1 - n + 1, l - n):  # 1 from both sides because there can not be a link right after BOS
            s = max(0, i + c)   # start of left
            m = i + n + c   # end of left = start of mid
            # break cycle if case left forces the right to be the single EOS
            # which it makes no sense to record because link can not appear there or any further
            if m > l - 1: break # -1 for special symbols
            link_candidates = []
            realizations = []
            for r in range(4):  # max length of a link representation is 3 as in -ens-
                e = m + r   # end of mid = start of right
                f = m + r + n   # end of right
                # break cycle if case right context is going to be the single EOS
                if e > l - 1: break   # -1 for special symbols
                left = lemma[s: m]
                mid = lemma[m: e]
                right = lemma[e: f]
                # will return unknown id if unknown
                position = '|'.join([left, right, mid])
                position_id = self.vocab_positions.encode(position)
                candidates = self.freqs_links[position_id]
                # keep track of link candidate ids
                link_candidates.append(candidates)
                # keep track of link candidate strings
                realizations.append(mid)
            link_candidates = np.stack(link_candidates)

            # top up the raw
            raw += lemma[i - (1 - n) + c] # discard the diff in the loop declaration + add skip step

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
                # the zero link will at some point become the most probable one and will break the loop

                # you can consider the whole thing as observations and their values,
                # where realizations are the observations and link ids are values;
                # so `link_candidates` tell us: at row r (= at realization r)
                # there is a distribution D with the most probable id l;
                # that is why we want to concatenate those into pairs and select a pair;
                # not argmax because argmax returns left-most values
                best_link_ids = np.stack(
                    np.where(link_candidates == link_candidates.max()),
                    axis=1
                )
                best_realization_idx, best_link_id = random.choice(best_link_ids)
                best_realization = realizations[best_realization_idx]

                # if there is no link, then unknown id is returned
                if best_link_id != self.vocab_links.unk_id:

                    best_link = self.vocab_links.decode(best_link_id)
                    component, realization, link_type = Compound.get_link_info(
                        best_link,
                        eliminate_allomorphy=self.eliminate_allomorphy
                    )

                    # heuristically filter out predictions that cannot be correct
                    if (    # use if so that no further checks are performed once one has failed
                        # umlaut link where there is no umlaut before; test only last stem, i.e. part after the last "_"
                        ("umlaut" in link_type and not re.search('|'.join(UMLAUTS_REVERSED.keys()), raw.split("_")[-1]))
                        # there are no other cases because it is deterministic after which contexts which links can 
                        # and cannot be predicted so no cases where there cannot be the predicted link can be encountered
                    ):
                        # zero this impossible link prob
                        link_candidates[best_realization_idx, best_link_id] = 0
                        continue


                    if link_type == "addition_umlaut":
                        raw = Compound.reverse_umlaut(raw)
                    elif link_type == "deletion":
                        to_delete = Compound.get_deletion(component)
                        raw += to_delete

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
                    if best_realization != realization:
                        component = re.sub(realization, best_realization, component)
                    raw += component

                    # When we encounter a link, we know for sure that there can not
                    # be another link after it (at least in v4 implementation).
                    # That is why we want to skip the link after we found it.
                    # For example, if we have "bundestag" and the model decided that after "nd",
                    # there is an "es", there is no sense for us to add "es" and
                    # start further with "de"; we want to continue straight to "ta".
                    # However, we cannot just assign a higher `i` because it
                    # will reset to its anticipated value in the new iteration,
                    # so we have to maintain a correction to add to `i`
                    # in order to be sure we are skipping the link. 
                    c += len(best_realization)

                    break

                else: break

        raw += lemma[-2] # complete raw when the window has sled, -1 for EOS

        pred = Compound(raw, eliminate_allomorphy=self.eliminate_allomorphy)

        return pred
    
    def predict(self, lemmas: List[str]) -> List[Compound]:

        """
        Make prediction from lemmas to DECOW16-format `Compound`s

        Parameters
        ----------
        lemmas : `List[str]`
            lemmas to predict

        Returns
        -------
        `List[Compound]`
            preds in DECOW16 compound format
        """

        progress_bar = tqdm(lemmas, desc="Predicting") if self.verbose else lemmas
        # note: async run is not efficient as one iteration is too fast
        return [
            self._predict(lemma)
            for lemma in progress_bar
        ]