from __future__ import annotations

from typing import TypeVar, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from numbers import Real


Self = TypeVar("Self", bound="NumericBase")


class NumericBase:
    def __pos__(self: Self) -> Self: ...

    def __neg__(self: Self) -> Self: ...

    def __abs__(self: Self) -> Self: ...

    def __trunc__(self: Self) -> Self: ...

    def __add__(self: Self, other: Union[Self, Real]) -> Self: ...

    def __radd__(self: Self, other: Union[Self, Real]) -> Self: ...

    def __sub__(self: Self, other: Union[Self, Real]) -> Self: ...

    def __rsub__(self: Self, other: Union[Self, Real]) -> Self: ...

    def __mul__(self: Self, other: Union[Self, Real]) -> Self: ...

    def __rmul__(self: Self, other: Union[Self, Real]) -> Self: ...

    def __truediv__(self: Self, other: Union[Self, Real]) -> Self: ...

    def __rtruediv__(self: Self, other: Union[Self, Real]) -> Self: ...

    def __pow__(self: Self, other: Union[Self, Real]) -> Self: ...

    def __rpow__(self: Self, other: Union[Self, Real]) -> Self: ...

    def __mod__(self: Self, other: Union[Self, Real]) -> Self: ...

    def __rmod__(self: Self, other: Union[Self, Real]) -> Self: ...

    """
    Implementation of the methods below enables interoperability with the corresponding
    numpy ufuncs. See https://numpy.org/doc/stable/reference/ufuncs.html.
    """

    def exp(self: Self) -> Self: ...

    def log(self: Self) -> Self: ...

    def log2(self: Self) -> Self: ...

    def log10(self: Self) -> Self: ...

    def sqrt(self: Self) -> Self: ...

    def square(self: Self) -> Self: ...

    def sin(self: Self) -> Self: ...

    def cos(self: Self) -> Self: ...

    def tan(self: Self) -> Self: ...

    def arcsin(self: Self) -> Self: ...

    def arccos(self: Self) -> Self: ...

    def arctan(self: Self) -> Self: ...

    def arctan2(self: Self, other: Union[Real, Self]) -> Self: ...

    def hypot(self: Self, other: Union[Real, Self]) -> Self: ...

    def sinh(self: Self) -> Self: ...

    def cosh(self: Self) -> Self: ...

    def tanh(self: Self) -> Self: ...

    def arcsinh(self: Self) -> Self: ...

    def arccosh(self: Self) -> Self: ...

    def arctanh(self: Self) -> Self: ...

    def degrees(self: Self) -> Self: ...

    def radians(self: Self) -> Self: ...

    def deg2rad(self: Self) -> Self: ...

    def rad2deg(self: Self) -> Self: ...