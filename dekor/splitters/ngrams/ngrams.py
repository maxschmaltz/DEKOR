"""
N-gram model for splitting German compounds based on the DECOW16 compound data.
"""

import re
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
        *,
        n: Optional[int]=3,
        record_none_links: bool,
        eliminate_allomorphy: bool,
        verbose: Optional[bool]=True
    ) -> None:
        self.n = n
        self.record_none_links = record_none_links
        self.eliminate_allomorphy = eliminate_allomorphy
        self.verbose = verbose
        self.vocab_positions = StringVocab()
        self.vocab_links = StringVocab()
        self._elink = Link(
            self.vocab_links.unk,
            span=(-1, -1),
            type=self.vocab_links.unk
        )

    def _metadata(self) -> dict:
        return {
            "n": self.n,
            **super()._metadata()
        }

    def _forward(self, compound: Compound) -> List[Tuple[int]]:

        masks = []
        for mask in self._get_positions(compound):
            for (left, right, mid), link in mask:
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

        # first things first, we need to analyze the lemma as a `Compound`;
        # thus, we will be able to reuse the `_get_positions()`
        compound = Compound(lemma)

        # We know for sure that we works with N+N compounds which means
        # we can heuristically restrict all improbable additional
        # links it will predict. Thus, for the compound, we will
        # gather probabilities of all non-none links with respect to their
        # positions and then choose the most probable one.
        link_probs = []
        # we will also map final positions in the prob matrix
        # to link indices to easily restore raw compound
        idx = 0 # mask index
        link_candidates = {}

        for i, mask in enumerate(self._get_positions(compound)):
            for (left, right, mid), _ in mask:  # link is to be predicted
                position = '|'.join([left, right, mid])
                position_id = self.vocab_positions.encode(position)
                probs = self.freqs_links[position_id]
                link_probs.append(probs)
                # start link index and link representation
                link_candidates[idx] = (i + 1, mid); idx += 1   # i + 1: correction for BOS
            
        link_probs = np.stack(link_probs)
        # at this point, none links have done their job and gave
        # the proportion (if they were even recorded) so
        # we should zero them to easily detect the most probable
        # non-none link
        link_probs[:, self.vocab_links.unk_id] = 0

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

            max_prob = link_probs.max()
            if not max_prob:
                return lemma   # no link detected, so return with no links
            
            best_idx, best_link_id = np.where(link_probs == max_prob)
            best_idx, best_link_id = best_idx[0], best_link_id[0]   # unpack
            i, best_realization = link_candidates[best_idx]
            best_link = self.vocab_links.decode(best_link_id)
            component, realization, link_type = Compound.get_link_info(
                best_link,
                eliminate_allomorphy=self.eliminate_allomorphy
            )

            # heuristically filter out predictions that cannot be correct
            # using `if` so that no further checks are performed once one has failed
            if (    
                # umlaut link where there is no umlaut before; `i` helps retrieve the past before the link
                ("umlaut" in link_type and not re.search('|'.join(UMLAUTS_REVERSED.keys()), lemma[:i]))
                # there are no other cases because it is deterministic after which contexts which links can 
                # and cannot be predicted so no cases where there cannot be the predicted link can be encountered
            ):
                # zero this impossible link prob
                link_probs[best_idx, best_link_id] = 0
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
            if best_realization != realization:
                component = re.sub(realization, best_realization, component)

            raw = lemma[:i] + component + lemma[i + len(best_realization):]
            pred = Compound(raw)
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
        preds = [
            self._predict(lemma)
            for lemma in progress_bar
        ]
        return preds