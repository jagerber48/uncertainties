from __future__ import annotations

from math import isfinite, isnan, isinf
from numbers import Real
from typing import Optional, TypeVar, Union

from uncertainties.formatting import format_ufloat
from uncertainties.numeric_base import NumericBase
from uncertainties.ucombo import UAtom, UCombo
from uncertainties.parsing import str_to_number_with_uncert

try:
    import numpy as np
except ImportError:
    np = None
    allow_numpy = False
else:
    allow_numpy = True


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
            tag: Optional[str]=None,
    ):
        """
        Using properties for value and uncertainty makes them essentially immutable.
        """
        self._value: float = float(value)

        if isinstance(uncertainty, Real):
            combo = UCombo(
                (
                    (UAtom(tag=tag), float(uncertainty)),
                )
            )
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

    def std_score(self, value):
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
        return self.n == other.n and self.u.expanded_dict == other.u.expanded_dict

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
    """
    Create an uncertainties Variable from a string representation.
    Several representation formats are supported.

    Arguments:
    ----------
    ufloat_str: string
        string representation of a value with uncertainty
    tag:   string or `None`
        optional tag for tracing and organizing Variables ['None']

    Returns:
    --------
    uncertainties Variable.

    Notes:
    --------
    1. Invalid representations raise a ValueError.

    2. Using the form "nominal(std)" where "std" is an integer creates
       a Variable with "std" giving the least significant digit(s).
       That is, "1.25(3)" is the same as `ufloat(1.25, 0.03)`,
       while "1.25(3.)" is the same as `ufloat(1.25, 3.)`

    Examples:
    -----------

    >>> x = ufloat_fromsstr("12.58+/-0.23")  # = ufloat(12.58, 0.23)
    >>> x = ufloat_fromsstr("12.58 ± 0.23")  # = ufloat(12.58, 0.23)
    >>> x = ufloat_fromsstr("3.85e5 +/- 2.3e4")  # = ufloat(3.8e5, 2.3e4)
    >>> x = ufloat_fromsstr("(38.5 +/- 2.3)e4")  # = ufloat(3.8e5, 2.3e4)

    >>> x = ufloat_fromsstr("72.1(2.2)")  # = ufloat(72.1, 2.2)
    >>> x = ufloat_fromsstr("72.15(4)")  # = ufloat(72.15, 0.04)
    >>> x = ufloat_fromstr("680(41)e-3")  # = ufloat(0.68, 0.041)
    >>> x = ufloat_fromstr("23.2")  # = ufloat(23.2, 0.1)
    >>> x = ufloat_fromstr("23.29")  # = ufloat(23.29, 0.01)

    >>> x = ufloat_fromstr("680.3(nan)") # = ufloat(680.3, numpy.nan)
    """
    (nom, std) = str_to_number_with_uncert(ufloat_str.strip())
    return UFloat(nom, std, tag)


def nominal_value(x: Union[UFloat, Real]) -> float:
    """
    Return the nominal value of x if it is a quantity with
    uncertainty (i.e., an AffineScalarFunc object); otherwise, returns
    x unchanged.

    This utility function is useful for transforming a series of
    numbers, when only some of them generally carry an uncertainty.
    """
    if isinstance(x, UFloat):
        return x.value
    elif isinstance(x, Real):
        return float(x)
    else:
        raise TypeError(f"x must be a UFloat or Real, not {type(x)}")


def std_dev(x: Union[UFloat, Real]) -> float:
    """
    Return the standard deviation of x if it is a quantity with
    uncertainty (i.e., an AffineScalarFunc object); otherwise, returns
    the float 0.

    This utility function is useful for transforming a series of
    numbers, when only some of them generally carry an uncertainty.
    """
    if isinstance(x, UFloat):
        return x.std_dev
    elif isinstance(x, Real):
        return 0.0
    else:
        raise TypeError(f"x must be a UFloat or Real, not {type(x)}")
