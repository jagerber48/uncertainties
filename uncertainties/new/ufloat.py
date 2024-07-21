from __future__ import annotations

from numbers import Real
from typing import TypeVar, Union

from uncertainties.formatting import format_ufloat
from uncertainties.new.ucombo import UAtom, UCombo
from uncertainties.new.numeric_base import NumericBase


Self = TypeVar("Self", bound="UFloat")


class UFloat(NumericBase):
    """
    Core class. Stores a mean value (value, nominal_value, n) and an uncertainty stored
    as a (possibly unexpanded) linear combination of uncertainty atoms. Two UFloat's
    which share non-zero weight for a certain uncertainty atom are correlated.

    UFloats can be combined using arithmetic and more sophisticated mathematical
    operations. The uncertainty is propagtaed using the rules of linear uncertainty
    propagation.
    """

    __slots__ = ["_value", "_uncertainty"]

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
    def value(self: Self) -> float:
        return self._value

    @property
    def uncertainty(self: Self) -> UCombo:
        return self._uncertainty

    @property
    def std_dev(self: Self) -> float:
        return self.uncertainty.std_dev

    def __format__(self: Self, format_spec: str = "") -> str:
        return format_ufloat(self, format_spec)

    def __str__(self: Self) -> str:
        return format(self)
        # return f'{self.val} ± {self.std_dev}'

    def __repr__(self: Self) -> str:
        return str(self)
        # """
        # Very verbose __repr__ including the entire uncertainty linear combination repr.
        # """
        # return (
        #     f'{self.__class__.__name__}({repr(self.value)}, {repr(self.uncertainty)})'
        # )

    def __bool__(self: Self) -> bool:
        return self != UFloat(0, 0)

    # Aliases
    @property
    def val(self: Self) -> float:
        return self.value

    @property
    def nominal_value(self: Self) -> float:
        return self.val

    @property
    def n(self: Self) -> float:
        return self.val

    @property
    def s(self: Self) -> float:
        return self.std_dev

    def __eq__(self: Self, other: Self) -> bool:
        if not isinstance(other, UFloat):
            return False
        val_eq = self.val == other.val

        self_expanded_linear_combo = self.uncertainty.expanded()
        other_expanded_linear_combo = other.uncertainty.expanded()
        uncertainty_eq = self_expanded_linear_combo == other_expanded_linear_combo
        return val_eq and uncertainty_eq

    def __hash__(self: Self) -> int:
        return hash((hash(self.val), hash(self.uncertainty)))


def ufloat(val: Real, unc: Real) -> UFloat:
    return UFloat(val, unc)
