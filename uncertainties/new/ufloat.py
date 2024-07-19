from __future__ import annotations

from numbers import Real
from typing import Union

from uncertainties.new.ucombo import UAtom, UCombo


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
    def value(self: UFloat) -> float:
        return self._value

    @property
    def uncertainty(self: UFloat) -> UCombo:
        return self._uncertainty

    @property
    def std_dev(self: UFloat) -> float:
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
    def val(self: UFloat) -> float:
        return self.value

    @property
    def nominal_value(self: UFloat) -> float:
        return self.val

    @property
    def n(self: UFloat) -> float:
        return self.val

    @property
    def s(self: UFloat) -> float:
        return self.std_dev

    def __eq__(self: UFloat, other: UFloat) -> bool:
        if not isinstance(other, UFloat):
            return False
        val_eq = self.val == other.val

        self_expanded_linear_combo = self.uncertainty.expanded()
        other_expanded_linear_combo = other.uncertainty.expanded()
        uncertainty_eq = self_expanded_linear_combo == other_expanded_linear_combo
        return val_eq and uncertainty_eq

    def __hash__(self):
        return hash((hash(self.val), hash(self.uncertainty)))

    def __pos__(self: UFloat) -> UFloat: ...

    def __neg__(self: UFloat) -> UFloat: ...

    def __abs__(self: UFloat) -> UFloat: ...

    def __trunc__(self: UFloat) -> UFloat: ...

    def __add__(self: UFloat, other: Union[UFloat, Real]) -> UFloat: ...

    def __radd__(self: UFloat, other: Union[UFloat, Real]) -> UFloat: ...

    def __sub__(self: UFloat, other: Union[UFloat, Real]) -> UFloat: ...

    def __rsub__(self: UFloat, other: Union[UFloat, Real]) -> UFloat: ...

    def __mul__(self: UFloat, other: Union[UFloat, Real]) -> UFloat: ...

    def __rmul__(self: UFloat, other: Union[UFloat, Real]) -> UFloat: ...

    def __truediv__(self: UFloat, other: Union[UFloat, Real]) -> UFloat: ...

    def __rtruediv__(self: UFloat, other: Union[UFloat, Real]) -> UFloat: ...

    def __pow__(self: UFloat, other: Union[UFloat, Real]) -> UFloat: ...

    def __rpow__(self: UFloat, other: Union[UFloat, Real]) -> UFloat: ...

    def __mod__(self: UFloat, other: Union[UFloat, Real]) -> UFloat: ...

    def __rmod__(self: UFloat, other: Union[UFloat, Real]) -> UFloat: ...


def ufloat(val: Real, unc: Real) -> UFloat:
    return UFloat(val, unc)
