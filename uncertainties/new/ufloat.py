from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from math import sqrt, isnan
from numbers import Real
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


"""
UncertaintyCombo represents a (possibly nested) linear superposition of 
UncertaintyAtoms. The UncertaintyCombo is an n-tuple of terms in the linear 
superposition and each term is represented by a 2-tuple. The second element of the
2-tuple is the weight of that term. The first element is either an UncertaintyAtom or
another UncertaintyCombo. In the latter case the original UncertaintyCombo is nested.

By passing the weights through the linear combinations and collecting like terms, any 
UncertaintyCombo can be expanded into a form where each term is an UncertaintyAtom. This
would be an ExpandedUncertaintyCombo.

Nested UncertaintyCombos are supported as a performance optimization. There is a
cost to expanding linear combinations during uncertainty propagation calculations. 
Supporting nested UncertaintyCombos allows expansion to be deferred through intermediate
calculations until a standard deviation or correlation must be calculated at the end of
an error propagation calculation.
"""
# TODO: How much does this optimization quantitatively improve performance?


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
        combo_list.append((atom, weight))
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


@dataclass(frozen=True)
class UCombo:
    combo: Tuple[Tuple[Union[UAtom, "UCombo"], float], ...]

    def __iter__(self):
        return iter(self.combo)

    def expanded(self: "UCombo") -> "ExpandedUCombo":
        return get_expanded_combo(self)

    @property
    def std_dev(self: "UCombo") -> float:
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


class UFloat:
    """
    Core class. Stores a mean value (value, nominal_value, n) and an uncertainty stored
    as a (possibly unexpanded) linear combination of uncertainty atoms. Two UFloat's
    which share non-zero weight for a certain uncertainty atom are correlated.

    UFloats can be combined using arithmetic and more sophisticated mathematical
    operations. The uncertainty is propagtaed using the rules of linear uncertainty
    propagation.
    """
    def __init__(self, value: Real, uncertainty: Union[UCombo, Real]):
        """
        Using properties for value and uncertainty makes them essentially immutable.
        """
        self._value: float = float(value)

        if isinstance(uncertainty, Real):
            atom = UAtom(float(uncertainty))
            combo = UCombo(((atom, 1.0),))
            self._uncertainty: UCombo = combo
        else:
            self._uncertainty: UCombo = uncertainty

    @property
    def value(self: "UFloat") -> float:
        return self._value

    @property
    def uncertainty(self: "UFloat") -> UCombo:
        return self._uncertainty

    @property
    def std_dev(self: "UFloat") -> float:
        return self.uncertainty.std_dev

    def __str__(self) -> str:
        return f'{self.val} ± {self.std_dev}'

    def __repr__(self) -> str:
        """
        Very verbose __repr__ including the entire uncertainty linear combination repr.
        """
        return (
            f'{self.__class__.__name__}({repr(self.value)}, {repr(self.uncertainty)})'
        )

    def __bool__(self):
        return self != UFloat(0, 0)

    # Aliases
    @property
    def val(self: "UFloat") -> float:
        return self.value

    @property
    def nominal_value(self: "UFloat") -> float:
        return self.val

    @property
    def n(self: "UFloat") -> float:
        return self.val

    @property
    def s(self: "UFloat") -> float:
        return self.std_dev

    def __eq__(self: "UFloat", other: "UFloat") -> bool:
        if not isinstance(other, UFloat):
            return False
        val_eq = self.val == other.val

        self_expanded_linear_combo = self.uncertainty.expanded()
        other_expanded_linear_combo = other.uncertainty.expanded()
        uncertainty_eq = self_expanded_linear_combo == other_expanded_linear_combo
        return val_eq and uncertainty_eq

    def __hash__(self):
        return hash((hash(self.val), hash(self.uncertainty)))

    def __pos__(self: "UFloat") -> "UFloat": ...

    def __neg__(self: "UFloat") -> "UFloat": ...

    def __abs__(self: "UFloat") -> "UFloat": ...

    def __trunc__(self: "UFloat") -> "UFloat": ...

    def __add__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __radd__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __sub__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __rsub__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __mul__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __rmul__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __truediv__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __rtruediv__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __pow__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __rpow__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __mod__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __rmod__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...


def ufloat(val: Real, unc: Real) -> UFloat:
    return UFloat(val, unc)
