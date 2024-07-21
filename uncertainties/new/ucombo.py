from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from math import isnan, sqrt
from typing import Dict, List, Tuple, Union
import uuid


@dataclass(frozen=True)
class UAtom:
    """
    Custom class to keep track of "atoms" of uncertainty. Two UncertaintyAtoms are
    always uncorrelated.
    """
    std_dev: float
    uuid: uuid.UUID = field(default_factory=uuid.uuid4, init=False)

    def __post_init__(self):
        if self.std_dev < 0:
            raise ValueError(f'Uncertainty must be non-negative, not {self.std_dev}.')

    def __str__(self):
        """
        __str__ drops the uuid
        """
        return f'{self.__class__.__name__}({self.std_dev})'


@lru_cache(maxsize=None)
def get_expanded_combo(
        combo: UCombo,
) -> ExpandedUCombo:
    """
    Recursively expand a nested UncertaintyCombo into an ExpandedUncertaintyCombo whose
    terms all represent weighted UncertaintyAtoms.
    """
    expanded_dict: Dict[UAtom, float] = defaultdict(float)
    for term, term_weight in combo:
        if isinstance(term, UAtom):
            expanded_dict[term] += term_weight
        else:
            expanded_term = get_expanded_combo(term)
            for atom, atom_weight in expanded_term:
                expanded_dict[atom] += atom_weight * term_weight

    combo_list: List[Tuple[UAtom, float]] = []
    for atom, weight in expanded_dict.items():
        if atom.std_dev == 0 or (weight == 0 and not isnan(atom.std_dev)):
            continue
        combo_list.append((atom, float(weight)))
    combo_tuple: Tuple[Tuple[UAtom, float], ...] = tuple(combo_list)

    return ExpandedUCombo(combo_tuple)


@lru_cache(maxsize=None)
def get_std_dev(combo: ExpandedUCombo) -> float:
    """
    Get the standard deviation corresponding to an UncertaintyCombo. The UncertainyCombo
    is expanded and the weighted UncertaintyAtoms are added in quadrature.
    """
    list_of_squares = [
        (weight*atom.std_dev)**2 for atom, weight in combo
    ]
    std_dev = sqrt(sum(list_of_squares))
    return std_dev


"""
UCombos represents a (possibly nested) linear superposition of UAtoms. The UCombo is a
sequence of terms in a linear combination. Each term is represented by a 2-tuple. The 
second element of the 2-tuple is the weight of that term. The first element is either a
UAtom or another UCombo. In the latter case the original UCombo is nested.

By passing the weights through the linear combinations and collecting like terms, any 
UCombo can be expanded into a form where each term is an UAtom. This would be an 
ExpandedUCombo.

Nested UCombo are supported as a performance optimization. There is a cost to expanding 
linear combinations during uncertainty propagation calculations. Supporting nested 
UCombo allows expansion to be deferred through intermediate calculations until a 
standard deviation or correlation must be calculated at the end of an error propagation 
calculation.
"""
# TODO: How much does this optimization quantitatively improve performance?


@dataclass(frozen=True)
class UCombo:
    combo: Tuple[Tuple[Union[UAtom, UCombo], float], ...]

    def __iter__(self):
        return iter(self.combo)

    def expanded(self: UCombo) -> ExpandedUCombo:
        return get_expanded_combo(self)

    @property
    def std_dev(self: UCombo) -> float:
        return get_std_dev(self.expanded())

    def __str__(self):
        ret_str = ""
        first = True
        for term, weight in self.combo:
            if not first:
                ret_str += " + "
            else:
                first = False

            if isinstance(term, UAtom):
                ret_str += f"{weight}×{term}"
            else:
                ret_str += f"{weight}×({term})"
        return ret_str


@dataclass(frozen=True)
class ExpandedUCombo(UCombo):
    combo: Tuple[Tuple[UAtom, float], ...]

    @property
    def atom_weight_dict(self: ExpandedCombo) -> dict[UAtom, float]:
        return {atom: weight for atom, weight in self}
