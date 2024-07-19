import math
from numbers import Real
import sys
from typing import Union

from uncertainties.new.ufloat import UFloat
from uncertainties.new.func_conversion import ToUFuncPositional


float_funcs_dict = {
        '__abs__': ('abs(x)/x',),
        '__pos__': ('1',),
        '__neg__': ('-1',),
        '__trunc__': ('0',),
        '__add__': ('1', '1'),
        '__radd__': ('1', '1'),
        '__sub__': ('1', '-1'),
        '__rsub__': ('-1', '1'),  # Reversed order __rsub__(x, y) = y - x
        '__mul__': ('y', 'x'),
        '__rmul__': ('y', 'x'),
        '__truediv__': ('1/y', '-x/y**2'),
        '__rtruediv__': ('-x/y**2', '1/y'),  # reversed order __rtruediv__(x, y) = y/x
        '__floordiv__': ('0', '0'),
        '__rfloordiv__': ('0', '0'),
        '__pow__': (None, None),  # TODO: add these, see `uncertainties` source
        '__rpow__': (None, None),
        '__mod__': (None, None),
        '__rmod__': (None, None),
    }


def add_float_funcs_to_ufloat():
    """
    Monkey-patch common float operations from the float class over to the UFloat class
    using the ToUFuncPositional decorator.
    """
    # TODO: There is some additional complexity added by allowing analytic derivative
    #   functions instead of taking numerical derivatives for all functions. It would
    #   be interesting to benchmark the different approaches and see if the additional
    #   complexity is worth the performance.
    for func_name, deriv_funcs in float_funcs_dict.items():
        float_func = getattr(float, func_name)
        ufloat_ufunc = ToUFuncPositional(deriv_funcs)(float_func)
        setattr(UFloat, func_name, ufloat_ufunc)


UReal = Union[Real, UFloat]


def acos(value: UReal, /) -> UReal: ...


def acosh(value: UReal, /) -> UReal: ...


def asin(value: UReal, /) -> UReal: ...


def asinh(value: UReal, /) -> UReal: ...


def atan(value: UReal, /) -> UReal: ...


def atan2(y: UReal, x: UReal, /) -> UReal: ...


def atanh(value: UReal, /) -> UReal: ...


def cos(value: UReal, /) -> UReal: ...


def cosh(value: UReal, /) -> UReal: ...


def degrees(value: UReal, /) -> UReal: ...


def erf(value: UReal, /) -> UReal: ...


def erfc(value: UReal, /) -> UReal: ...


def exp(value: UReal, /) -> UReal: ...


def hypot(x: UReal, y: UReal, /) -> UReal: ...


def log(value: UReal, base=None, /) -> UReal: ...


def log10(value: UReal, /) -> UReal: ...


def radians(value: UReal, /) -> UReal: ...


def sin(value: UReal, /) -> UReal: ...


def sinh(value: UReal, /) -> UReal: ...


def sqrt(value: UReal, /) -> UReal: ...


def tan(value: UReal, /) -> UReal: ...


def tanh(value: UReal, /) -> UReal: ...


def log_der0(*args):
    """
    Derivative of math.log() with respect to its first argument.

    Works whether 1 or 2 arguments are given.
    """
    if len(args) == 1:
        return 1 / args[0]
    else:
        return 1 / args[0] / math.log(args[1])  # 2-argument form


math_funcs_dict = {
    # In alphabetical order, here:
    "acos": ("-1/math.sqrt(1-x**2)",),
    "acosh": ("1/math.sqrt(x**2-1)",),
    "asin": ("1/math.sqrt(1-x**2)",),
    "asinh": ("1/math.sqrt(1+x**2)",),
    "atan": ("1/(1+x**2)",),
    "atan2": ('y/(x**2+y**2)', "-x/(x**2+y**2)"),
    "atanh": ("1/(1-x**2)",),
    "cos": ("-math.sin(x)",),
    "cosh": ("math.sinh(x)",),
    "degrees": ("math.degrees(1)",),
    "erf": ("(2/math.sqrt(math.pi))*math.exp(-(x**2))",),
    "erfc": ("-(2/math.sqrt(math.pi))*math.exp(-(x**2))",),
    "exp": ("math.exp(x)",),
    "hypot": ("x/math.hypot(x, y)", "y/math.hypot(x, y)"),
    "log": (log_der0, "-math.log(x, y) / y / math.log(y)"),
    "log10": ("1/x/math.log(10)",),
    "radians": ("math.radians(1)",),
    "sin": ("math.cos(x)",),
    "sinh": ("math.cosh(x)",),
    "sqrt": ("0.5/math.sqrt(x)",),
    "tan": ("1 + math.tan(x)**2",),
    "tanh": ("1 - math.tanh(x)**2",),
}

this_module = sys.modules[__name__]


def add_math_funcs_to_umath():
    for func_name, deriv_funcs in math_funcs_dict.items():
        func = getattr(math, func_name)
        ufunc = ToUFuncPositional(deriv_funcs, eval_locals={"math": math})(func)
        setattr(this_module, func_name, ufunc)


ufuncs_umath_dict = {
    'exp': lambda x: exp(x),
    'log': lambda x: log(x),
    'log2': lambda x: log(x, 2),
    'log10': lambda x: log10(x),
    'sqrt': lambda x: sqrt(x),
    'square': lambda x: x**2,
    'sin': lambda x: sin(x),
    'cos': lambda x: cos(x),
    'tan': lambda x: tan(x),
    'arcsin': lambda x: asin(x),
    'arccos': lambda x: acos(x),
    'arctan': lambda x: atan(x),
    'arctan2': lambda y, x: atan2(y, x),
    'hypot': lambda x, y: hypot(x, y),
    'sinh': lambda self: sinh(self),
    'cosh': lambda self: cosh(self),
    'tanh': lambda self: tanh(self),
    'arcsinh': lambda self: asinh(self),
    'arccosh': lambda self: acosh(self),
    'arctanh': lambda self: atanh(self),
    'degrees': lambda self: degrees(self),
    'radians': lambda self: radians(self),
    'deg2rad': lambda self: radians(self),
    'rad2deg': lambda self: degrees(self),
}


def add_ufuncs_to_ufloat():
    for func_name, func in ufuncs_umath_dict.items():
        setattr(UFloat, func_name, func)
