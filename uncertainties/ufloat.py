from __future__ import annotations

from math import isfinite, isnan, isinf
from numbers import Real
from typing import Sequence, TypeVar, Union

from uncertainties.formatting import format_ufloat
from uncertainties.ucombo import UAtom, UCombo
from uncertainties.numeric_base import NumericBase

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

    __slots__ = ["_value", "_uncertainty"]

    def __init__(self, value: Real, uncertainty: Union[UCombo, Real]):
        """
        Using properties for value and uncertainty makes them essentially immutable.
        """
        self._value: float = float(value)

        if isinstance(uncertainty, Real):
            combo = UCombo(
                (
                    (UAtom(), float(uncertainty)),
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


def correlated_values(nominal_values, covariance_matrix):
    """
    Return an array of UFloat from a sequence of nominal values and a covariance matrix.
    """
    if not allow_numpy:
        raise ValueError(
            'numpy import failed. Unable to calculate UFloats from covariance matrix.'
        )

    n = covariance_matrix.shape[0]
    L = np.linalg.cholesky(covariance_matrix)

    ufloat_atoms = []
    for _ in range(n):
        ufloat_atoms.append(UFloat(0, 1))

    result = np.array(nominal_values) + L @ np.array(ufloat_atoms)
    return result


def covariance_matrix(ufloats: Sequence[UFloat]):
    """
    Return the covariance matrix of a sequence of UFloat.
    """
    # TODO: The only reason this function requires numpy is because it returns a numpy
    #   array. It could be made to return a nested list instead. But it seems ok to
    #   require numpy for users who want a covariance matrix.
    if not allow_numpy:
        raise ValueError(
            'numpy import failed. Unable to calculate covariance matrix.'
        )

    n = len(ufloats)
    cov = np.zeros((n, n))
    atom_weight_dicts = [
            ufloat.uncertainty.expanded_dict for ufloat in ufloats
    ]
    atom_sets = [
        set(atom_weight_dict.keys()) for atom_weight_dict in atom_weight_dicts
    ]
    for i in range(n):
        atom_weight_dict_i = atom_weight_dicts[i]
        for j in range(i, n):
            atom_intersection = atom_sets[i].intersection(atom_sets[j])
            if not atom_intersection:
                continue
            term = 0
            atom_weight_dict_j = atom_weight_dicts[j]
            for atom in atom_intersection:
                term += atom_weight_dict_i[atom] * atom_weight_dict_j[atom]
            cov[i, j] = term
            cov[j, i] = term
    return cov
