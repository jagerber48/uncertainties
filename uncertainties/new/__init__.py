import warnings

from uncertainties.new.ufloat import UFloat, ufloat
from uncertainties.new.umath import (
    add_float_funcs_to_ufloat,
    add_math_funcs_to_umath,
    add_ufuncs_to_ufloat,
)


__all__ = ["UFloat", "ufloat"]


try:
    from uncertainties.new.uarray import UArray
    __all__.append("UArray")
except ImportError:
    warnings.warn('Failed to import numpy. UArray functionality is unavailable.')


add_float_funcs_to_ufloat()
add_math_funcs_to_umath()
add_ufuncs_to_ufloat()
