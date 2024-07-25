from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from dataclasses import dataclass, field
from math import sqrt
from numbers import Real
from typing import Dict, Tuple, TypeVar, Union
import uuid


@dataclass(frozen=True)
class UAtom:
    uuid: uuid.UUID = field(init=False, default_factory=uuid.uuid4)

    def __str__(self):
        uuid_abbrev = f"{str(self.uuid)[0:2]}..{str(self.uuid)[-3:-1]}"
        return f"{self.__class__.__name__}({uuid_abbrev})"


UComboDict = Dict[UAtom, float]


@lru_cache(maxsize=None)
def get_ucombo_dict(ucombo: Self) -> UComboDict:
    expanded_dict: UComboDict = defaultdict(float)

    for term, term_weight in ucombo:
        if isinstance(term, UAtom):
            expanded_dict[term] += term_weight
        else:
            expanded_term = get_ucombo_dict(term)
            for atom, atom_weight in expanded_term.items():
                expanded_dict[atom] += term_weight * atom_weight

    pruned_expanded_dict = {
        atom: weight for atom, weight in expanded_dict.items() if weight != 0
    }
    return pruned_expanded_dict


@lru_cache(maxsize=None)
def get_std_dev(ucombo_dict: UComboDict) -> float:
    return sqrt(sum(weight**2 for weight in ucombo_dict.values()))


Self = TypeVar("Self", bound="UCombo")  # TODO: typing.Self introduced in Python 3.11


# TODO: Right now UCombo lacks __slots__. Python 3.10 allows slot=True input argument to
#   dataclass. Until then the easiest way to get __slots__ back would be to not use a
#   dataclass here.
@dataclass(frozen=True)
class UCombo:
    ucombo_tuple: Tuple[Tuple[Union[UAtom, UCombo], float], ...]

    @property
    def expanded_dict(self: Self) -> UComboDict:
        return get_ucombo_dict(self)

    @property
    def std_dev(self: Self) -> float:
        return get_std_dev(self.expanded_dict)

    def __add__(self: Self, other) -> Self:
        if not isinstance(other, UCombo):
            return NotImplemented
        return UCombo(((self, 1.0), (other, 1.0)))

    def __radd__(self: Self, other):
        return self.__add__(other)

    def __mul__(self: Self, scalar: Real):
        if not isinstance(scalar, Real):
            return NotImplemented
        return UCombo(
            (
                (self, float(scalar)),
            )
        )

    def __rmul__(self: Self, scalar: Real):
        return self.__mul__(scalar)

    def __iter__(self: Self):
        return iter(self.ucombo_tuple)

    def __str__(self: Self) -> str:
        ret_str = ""
        first = True
        for term, weight in self:
            if not first:
                ret_str += " + "
            else:
                first = False

            if isinstance(term, UAtom):
                ret_str += f"{weight}×{str(term)}"
            else:
                ret_str += f"{weight}×({str(term)})"
        return ret_str
