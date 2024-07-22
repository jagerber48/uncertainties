import warnings

from uncertainties.new.ufloat import (
    UFloat,
    correlated_values,
    covariance_matrix,
)
from uncertainties.new.umath import (
    add_float_funcs_to_ufloat,
    add_math_funcs_to_umath,
    add_ufuncs_to_ufloat,
)
from uncertainties.new.func_conversion import ToUFunc, ToUFuncPositional


__all__ = [
    "UFloat",
    "correlated_values",
    "covariance_matrix",
    "ToUFunc",
    "ToUFuncPositional",
]


try:
    from uncertainties.new.uarray import UArray
    __all__.append("UArray")
except ImportError:
    warnings.warn('Failed to import numpy. UArray functionality is unavailable.')


add_float_funcs_to_ufloat()
add_math_funcs_to_umath()
add_ufuncs_to_ufloat()
