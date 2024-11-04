from __future__ import annotations

from functools import wraps
from math import sqrt
from numbers import Real
import sys
from typing import Callable, Union

import numpy as np

from uncertainties.new.ufloat import UFloat
from uncertainties.new.ucombo import UCombo
from uncertainties.new.func_conversion import (
    inject_to_args_kwargs,
    get_args_kwargs_list,
)


class UArray(np.ndarray):
    def __new__(cls, input_array) -> UArray:
        obj = np.asarray(input_array).view(cls)
        return obj

    @property
    def value(self):
        return np.array(np.vectorize(lambda uval: uval.value)(self), dtype=float)

    @property
    def std_dev(self):
        return np.array(np.vectorize(lambda uval: uval.std_dev)(self), dtype=float)

    @property
    def uncertainty(self):
        return np.array(np.vectorize(lambda uval: uval.uncertainty)(self), dtype=object)

    @property
    def expanded_uncertainty(self):
        return np.array(
            np.vectorize(lambda uval: uval.expanded_uncertainty)(self)
            , dtype=object
        )

    @classmethod
    def from_arrays(cls, value_array, uncertainty_array) -> UArray:
        return cls(np.vectorize(UFloat)(value_array, uncertainty_array))

    def __str__(self: UArray) -> str:
        return f"{self.__class__.__name__}({super().__str__()})"

    def mean(self, axis=None, dtype=None, out=None, keepdims=None, *, where=None):
        """
        Include to fix an issue where np.mean is returning a singleton 0d array rather
        than scalar.
        """
        args = [axis, dtype, out]
        if keepdims is not None:
            args.append(keepdims)
        kwargs = {}
        if where is not None:
            kwargs['where'] = where
        result = super().mean(*args, **kwargs)

        if result.ndim == 0:
            return result.item()
        return result

    # Aliases
    @property
    def val(self):
        return self.value

    @property
    def nominal_value(self):
        return self.value

    @property
    def n(self):
        return self.value

    @property
    def s(self):
        return self.std_dev

    @property
    def u(self):
        return self.uncertainty


SQRT_EPS = sqrt(sys.float_info.epsilon)


def array_numerical_partial_derivative(
        f: Callable[..., Real],
        target_param: Union[str, int],
        array_multi_index: tuple = None,
        *args,
        **kwargs
) -> float:
    """
    Numerically calculate the partial derivative of a function f with respect to the
    target_param (string name or position number of the float parameter to f to be
    varied) holding all other arguments, *args and **kwargs, constant.
    """
    if isinstance(target_param, int):
        x = args[target_param]
    else:
        x = kwargs[target_param]

    if array_multi_index is None:
        dx = abs(x) * SQRT_EPS  # Numerical Recipes 3rd Edition, eq. 5.7.5
        x_lower = x - dx
        x_upper = x + dx
    else:
        dx = np.mean(np.abs(x)) * SQRT_EPS
        x_lower = np.copy(x)
        x_upper = np.copy(x)
        x_lower[array_multi_index] -= dx
        x_upper[array_multi_index] += dx

    lower_args, lower_kwargs = inject_to_args_kwargs(
        target_param,
        x_lower,
        *args,
        **kwargs,
    )
    upper_args, upper_kwargs = inject_to_args_kwargs(
        target_param,
        x_upper,
        *args,
        **kwargs,
    )

    lower_y = f(*lower_args, **lower_kwargs)
    upper_y = f(*upper_args, **upper_kwargs)

    derivative = (upper_y - lower_y) / (2 * dx)
    return derivative


def to_uarray_func(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        return_uarray = False

        float_args = []
        for arg in args:
            if isinstance(arg, UFloat):
                float_args.append(arg.nominal_value)
                return_uarray = True
            elif isinstance(arg, np.ndarray):
                if isinstance(arg.flat[0], UFloat):
                    float_args.append(UArray(arg).nominal_value)
                    return_uarray = True
                else:
                    float_args.append(arg)
            else:
                float_args.append(arg)

        float_kwargs = {}
        for key, arg in kwargs.items():
            if isinstance(arg, UFloat):
                float_kwargs[key] = arg.nominal_value
                return_uarray = True
            elif isinstance(arg, np.ndarray):
                if isinstance(arg.flat[0], UFloat):
                    float_kwargs[key] = UArray(arg).nominal_value
                    return_uarray = True
                else:
                    float_kwargs[key] = arg
            else:
                float_kwargs[key] = arg

        new_nominal_array = func(*float_args, **float_kwargs)
        if not return_uarray:
            return new_nominal_array

        args_kwargs_list = get_args_kwargs_list(*args, **kwargs)

        ucombo_array = np.ones_like(new_nominal_array) * UCombo(())

        for label, arg in args_kwargs_list:
            if isinstance(arg, UFloat):
                deriv_arr = array_numerical_partial_derivative(
                    func,
                    label,
                    None,
                    *float_args,
                    **float_kwargs
                )
                ucombo_array += deriv_arr * arg.uncertainty
            elif isinstance(arg, np.ndarray):
                if isinstance(arg.flat[0], UFloat):
                    it = np.nditer(arg, flags=["multi_index", "refs_ok"])
                    for sub_arg in it:
                        multi_index = it.multi_index
                        deriv_arr = array_numerical_partial_derivative(
                            func,
                            label,
                            multi_index,
                            *float_args,
                            **float_kwargs,
                        )
                        ucombo_array += deriv_arr * sub_arg.item().uncertainty

        return UArray.from_arrays(
            new_nominal_array,
            ucombo_array,
        )
    return wrapped
