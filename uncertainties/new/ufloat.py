from __future__ import annotations

from math import isfinite, isnan, isinf
from numbers import Real
from typing import Optional, TypeVar, Union

from uncertainties.formatting import format_ufloat
from uncertainties.new.numeric_base import NumericBase
from uncertainties.new.ucombo import UAtom, UCombo
from uncertainties.parsing import str_to_number_with_uncert


Self = TypeVar("Self", bound="UFloat")


class UFloat(NumericBase):
    """
    Stores a mean value (value, nominal_value, n) and an uncertainty stored
    as a (possibly nested) linear combination of uncertainty atoms. Two UFloat instances
    which share non-zero weight for a certain uncertainty atom are correlated.

    UFloats can be combined using arithmetic and more sophisticated mathematical
    operations. The uncertainty is propagtaed using the rules of linear uncertainty
    propagation.
    """

    __slots__ = ["_value", "_uncertainty", "tag"]

    def __init__(
        self,
        value: Real,
        uncertainty: Union[UCombo, Real],
        tag: Optional[str] = None,
    ):
        self._value: float = float(value)

        if isinstance(uncertainty, Real):
            combo = UCombo(((UAtom(tag=tag), float(uncertainty)),))
            self._uncertainty: UCombo = combo
        else:
            self._uncertainty: UCombo = uncertainty

        self.tag = tag  # TODO: I do not think UFloat should have tag attribute.
        #   Maybe UAtom can, but I'm not sure why.

    @property
    def value(self: Self) -> float:
        return self._value

    @property
    def uncertainty(self: Self) -> UCombo:
        return self._uncertainty

    @property
    def std_dev(self: Self) -> float:
        return self.uncertainty.std_dev

    def std_score(self: Self, value: float) -> float:
        """
        Return (value - nominal_value), in units of the standard deviation.
        """
        try:
            return (value - self.value) / self.std_dev
        except ZeroDivisionError:
            return float("nan")

    @property
    def error_components(self: Self) -> dict[UAtom, float]:
        return self.uncertainty.expanded_dict

    def __eq__(self: Self, other: Self) -> bool:
        if not isinstance(other, UFloat):
            return False
        return self.n == other.n and self.u == other.u

    def __format__(self: Self, format_spec: str = "") -> str:
        return format_ufloat(self, format_spec)

    def __str__(self: Self) -> str:
        return format(self)

    def __repr__(self: Self) -> str:
        """
        Note that the repr includes the std_dev and not the uncertainty. This repr is
        incomplete since it does not reveal details about the uncertainty UCombo and
        correlations.
        """
        return f"{self.__class__.__name__}({repr(self.value)}, {repr(self.std_dev)})"

    def __bool__(self: Self) -> bool:
        return self != UFloat(0, 0)

    # Aliases
    @property
    def val(self: Self) -> float:
        return self.value

    @property
    def nominal_value(self: Self) -> float:
        return self.value

    @property
    def n(self: Self) -> float:
        return self.value

    @property
    def s(self: Self) -> float:
        return self.std_dev

    @property
    def u(self: Self) -> UCombo:
        return self.uncertainty

    def isfinite(self: Self) -> bool:
        return isfinite(self.value)

    def isinf(self: Self) -> bool:
        return isinf(self.value)

    def isnan(self: Self) -> bool:
        return isnan(self.value)

    def __hash__(self: Self) -> int:
        return hash((hash(self.val), hash(self.uncertainty)))


def ufloat(value: float, uncertainty: float, tag: Optional[str] = None) -> UFloat:
    return UFloat(value, uncertainty, tag)


def ufloat_fromstr(ufloat_str: str, tag: Optional[str] = None):
    # TODO: Do we really want to strip here?
    (nom, std) = str_to_number_with_uncert(ufloat_str.strip())
    return UFloat(nom, std, tag)


def nominal_value(x: Union[UFloat, Real]) -> float:
    if isinstance(x, UFloat):
        return x.value
    elif isinstance(x, Real):
        return float(x)
    else:
        raise TypeError(f"x must be a UFloat or Real, not {type(x)}")


def std_dev(x: Union[UFloat, Real]) -> float:
    if isinstance(x, UFloat):
        return x.std_dev
    elif isinstance(x, Real):
        return 0.0
    else:
        raise TypeError(f"x must be a UFloat or Real, not {type(x)}")
