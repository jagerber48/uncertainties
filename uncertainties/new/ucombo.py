from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from math import sqrt
from typing import Dict, List, Tuple, Union
import uuid


@dataclass(frozen=True)
class UAtom:
    """
    Custom class to keep track of "atoms" of uncertainty. Two UncertaintyAtoms are
    always uncorrelated.
    """
    uuid: uuid.UUID = field(default_factory=uuid.uuid4, init=False)


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
        if weight == 0:
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
    std_dev = sqrt(sum([weight**2 for _, weight in combo]))
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

    def __add__(self, other):
        if not isinstance(other, (UAtom, UCombo)):
            return NotImplemented
        return UCombo(((self, 1.0), (other, 1.0)))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, scalar):
        if not isinstance(scalar, float):
            return NotImplemented
        return UCombo(((self, scalar),))

    def __rmul__(self, scalar):
        return self.__mul__(scalar)


@dataclass(frozen=True)
class ExpandedUCombo(UCombo):
    combo: Tuple[Tuple[UAtom, float], ...]

    @property
    def atom_weight_dict(self: ExpandedUCombo) -> dict[UAtom, float]:
        return {atom: weight for atom, weight in self}
