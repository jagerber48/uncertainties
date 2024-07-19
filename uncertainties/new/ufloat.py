from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from math import sqrt, isnan
from numbers import Real
from typing import Optional, Tuple, Union
import uuid


@dataclass(frozen=True)
class UncertaintyAtom:
    """
    Custom class to keep track of "atoms" of uncertainty. Two UncertaintyAtoms are
    always uncorrelated.
    """
    std_dev: float
    uuid: uuid.UUID = field(default_factory=uuid.uuid4, init=False)


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
UncertaintyCombo = Tuple[
    Tuple[
        Union[UncertaintyAtom, "UncertaintyCombo"],
        float
    ],
    ...
]
ExpandedUncertaintyCombo = Tuple[
    Tuple[
        UncertaintyAtom,
        float
    ],
    ...
]


@lru_cache(maxsize=None)
def get_expanded_combo(combo: UncertaintyCombo) -> ExpandedUncertaintyCombo:
    """
    Recursively expand a nested UncertaintyCombo into an ExpandedUncertaintyCombo whose
    terms all represent weighted UncertaintyAtoms.
    """
    expanded_dict = defaultdict(float)
    for combo, combo_weight in combo:
        if isinstance(combo, UncertaintyAtom):
            expanded_dict[combo] += combo_weight
        else:
            expanded_combo = get_expanded_combo(combo)
            for atom, atom_weight in expanded_combo:
                expanded_dict[atom] += atom_weight * combo_weight

    pruned_expanded_dict = {}
    for atom, weight in expanded_dict.items():
        if atom.std_dev == 0 or (weight == 0 and not isnan(atom.std_dev)):
            continue
        pruned_expanded_dict[atom] = weight

    return tuple((atom, weight) for atom, weight in pruned_expanded_dict.items())


@lru_cache(maxsize=None)
def get_std_dev(combo: UncertaintyCombo) -> float:
    """
    Get the standard deviation corresponding to an UncertaintyCombo. The UncertainyCombo
    is expanded and the weighted UncertaintyAtoms are added in quadrature.
    """
    expanded_combo = get_expanded_combo(combo)
    list_of_squares = [
        (weight*atom.std_dev)**2 for atom, weight in expanded_combo
    ]
    std_dev = sqrt(sum(list_of_squares))
    return std_dev


class UFloat:
    """
    Core class. Stores a mean value (value, nominal_value, n) and an uncertainty stored
    as a (possibly unexpanded) linear combination of uncertainty atoms. Two UFloat's
    which share non-zero weight for a certain uncertainty atom are correlated.

    UFloats can be combined using arithmetic and more sophisticated mathematical
    operations. The uncertainty is propagtaed using the rules of linear uncertainty
    propagation.
    """
    def __init__(
            self,
            /,
            value: Real,
            uncertainty: Union[UncertaintyCombo, Real] = (),
            tag: Optional[str] = None
    ):
        self._val = float(value)
        if isinstance(uncertainty, Real):
            if uncertainty < 0:
                raise ValueError(
                    f'Uncertainty must be non-negative, not {uncertainty}.'
                )
            atom = UncertaintyAtom(float(uncertainty))
            uncertainty_combo = ((atom, 1.0),)
            self.uncertainty_lin_combo = uncertainty_combo
        else:
            self.uncertainty_lin_combo = uncertainty
        self.tag = tag

    @property
    def val(self: "UFloat") -> float:
        return self._val

    @property
    def std_dev(self: "UFloat") -> float:
        return get_std_dev(self.uncertainty_lin_combo)

    @property
    def uncertainty(self: "UFloat") -> UncertaintyCombo:
        return self.uncertainty_lin_combo

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.val}, {self.std_dev})'

    def __bool__(self):
        return self != UFloat(0, 0)

    # Aliases
    @property
    def nominal_value(self: "UFloat") -> float:
        return self.val

    @property
    def n(self: "UFloat") -> float:
        return self.val

    @property
    def value(self: "UFloat") -> float:
        return self.val

    @property
    def s(self: "UFloat") -> float:
        return self.std_dev

    def __eq__(self: "UFloat", other: "UFloat") -> bool:
        if not isinstance(other, UFloat):
            return False
        val_eq = self.val == other.val

        self_expanded_linear_combo = get_expanded_combo(self.uncertainty_lin_combo)
        other_expanded_linear_combo = get_expanded_combo(other.uncertainty_lin_combo)
        uncertainty_eq = self_expanded_linear_combo == other_expanded_linear_combo
        return val_eq and uncertainty_eq

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


def ufloat(val: Real, unc: Real, tag=None) -> UFloat:
    return UFloat(val, unc, tag)
