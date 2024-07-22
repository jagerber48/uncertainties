from functools import wraps
from math import sqrt
import sys
from typing import Any, Callable, Dict, Optional, Tuple, Union

from uncertainties.new.ufloat import UFloat, UCombo


# TODO: Much of the code in this module does some manual looping through args and
#   kwargs. This could be simplified with the use of inspect.signature. However,
#   unfortunately, some built-in functions in the math library, such as math.log, do not
#   yet work with inspect.signature. This is the case of python 3.12.
#   Monitor https://github.com/python/cpython/pull/117671 for updates.


def inject_to_args_kwargs(param, injected_arg, *args, **kwargs):
    if isinstance(param, int):
        new_kwargs = kwargs
        new_args = []
        for idx, arg in enumerate(args):
            if idx == param:
                new_args.append(injected_arg)
            else:
                new_args.append(arg)
    elif isinstance(param, str):
        new_args = args
        new_kwargs = kwargs
        new_kwargs[param] = injected_arg
    else:
        raise TypeError(f'{param} must be an int or str, not {type(param)}.')
    return new_args, new_kwargs


SQRT_EPS = sqrt(sys.float_info.epsilon)


def numerical_partial_derivative(
        f: Callable[..., float],
        target_param: Union[str, int],
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
    dx = abs(x) * SQRT_EPS  # Numerical Recipes 3rd Edition, eq. 5.7.5

    lower_args, lower_kwargs = inject_to_args_kwargs(
        target_param,
        x-dx,
        *args,
        **kwargs,
    )
    upper_args, upper_kwargs = inject_to_args_kwargs(
        target_param,
        x+dx,
        *args,
        **kwargs,
    )

    lower_y = f(*lower_args, **lower_kwargs)
    upper_y = f(*upper_args, **upper_kwargs)

    derivative = (upper_y - lower_y) / (2 * dx)
    return derivative


def get_args_kwargs_list(*args, **kwargs):
    args_kwargs_list = []
    for idx, arg in enumerate(args):
        args_kwargs_list.append((idx, arg))
    for key, arg in kwargs.items():
        args_kwargs_list.append((key, arg))
    return args_kwargs_list


DerivFuncDict = Optional[Dict[Union[str, int], Callable[..., float]]]


class ToUFunc:
    """
    Decorator which converts a function which accepts real numbers and returns a real
    number into a function which accepts UFloats and returns a UFloat. The returned
    UFloat will have the same value as if the original function had been called using
    the values of the input UFloats. But, additionally, it will have an uncertainty
    corresponding to the square root of the sum of squares of the uncertainties of the
    input UFloats weighted by the partial derivatives of the original function with
    respect to the corresponding input parameters.

    :param deriv_func_dict: Dictionary mapping positional or keyword parameters to
      functions that return the partial derivatives of the decorated function with
      respect to the corresponding parameter. This function will be called if a UFloat
      is passed as an argument to the corresponding parameter. If a UFloat is passed
      into a parameter which is not specified in deriv_func_dict then the partial
      derivative will be evaluated numerically.
    """
    def __init__(
            self,
            deriv_func_dict: DerivFuncDict = None,
    ):

        if deriv_func_dict is None:
            deriv_func_dict = {}
        self.deriv_func_dict: DerivFuncDict = deriv_func_dict

    def __call__(self, f: Callable[..., float]):
        # sig = inspect.signature(f)

        @wraps(f)
        def wrapped(*args, **kwargs):
            return_u_val = False
            float_args = []
            for arg in args:
                if isinstance(arg, UFloat):
                    float_args.append(arg.val)
                    return_u_val = True
                else:
                    float_args.append(arg)
            float_kwargs = {}
            for key, arg in kwargs.items():
                if isinstance(arg, UFloat):
                    float_kwargs[key] = arg.val
                    return_u_val = True
                else:
                    float_kwargs[key] = arg

            new_val = f(*float_args, **float_kwargs)

            if not return_u_val:
                return new_val

            args_kwargs_list = get_args_kwargs_list(*args, **kwargs)

            new_combo_list = []
            for label, arg in args_kwargs_list:
                if isinstance(arg, UFloat):
                    if label in self.deriv_func_dict:
                        deriv_func = self.deriv_func_dict[label]
                        derivative = deriv_func(*float_args, **float_kwargs)
                    else:
                        derivative = numerical_partial_derivative(
                            f,
                            label,
                            *float_args,
                            **float_kwargs,
                        )

                    try:
                        """
                        In cases where other args are ndarray or UArray the calculation
                        of the derivative may return an array rather than a scalar 
                        float.
                        """
                        derivative = float(derivative)
                    except TypeError:
                        """
                        This is relevant in cases like
                        ufloat * uarr
                        Here we want to pass on UFloat.__mul__ and defer calculation to 
                        UArray.__rmul__. In such a case derivative may successfully be
                        calculated as an array but this array can't be easily handled in
                        the new UCombo generation Returning NotImplemented here defers 
                        to UArray.__rmul__ which allows the numpy machinery to take over
                        the vectorization.
                        """
                        return NotImplemented

                    new_combo_list.append(
                        (arg.uncertainty, derivative)
                    )

            new_uncertainty_combo = UCombo(tuple(new_combo_list))
            return UFloat(new_val, new_uncertainty_combo)

        return wrapped


def func_str_to_positional_func(func_str, nargs, eval_locals=None):
    if eval_locals is None:
        eval_locals = {}
    if nargs == 1:
        def pos_func(x):
            eval_locals['x'] = x
            return eval(func_str, None, eval_locals)
    elif nargs == 2:
        def pos_func(x, y):
            eval_locals['x'] = x
            eval_locals['y'] = y
            return eval(func_str, None, eval_locals)
    else:
        raise ValueError(f'Only nargs=1 or nargs=2 is supported, not {nargs=}.')
    return pos_func


PositionalDerivFunc = Union[Callable[..., float], str]


def deriv_func_dict_positional_helper(
        deriv_funcs: Tuple[Optional[PositionalDerivFunc]],
        eval_locals=None,
):
    nargs = len(deriv_funcs)
    deriv_func_dict = {}

    for arg_num, deriv_func in enumerate(deriv_funcs):
        if callable(deriv_func):
            pass
        elif isinstance(deriv_func, str):
            deriv_func = func_str_to_positional_func(deriv_func, nargs, eval_locals)
        else:
            raise ValueError(
                f'Invalid deriv_func: {deriv_func}. Must be callable or a string.'
            )
        deriv_func_dict[arg_num] = deriv_func
    return deriv_func_dict


class ToUFuncPositional(ToUFunc):
    """
    Helper decorator for ToUFunc for functions which accept one or two floats as
    positional input parameters and return a float.

    :param deriv_funcs: List of functions or strings specifying a custom partial
      derivative function for the first positional parameters of the wrapped function.
      Elements of the list can be callable functions with the same signature as the
      wrapped function. They can also be string representations of functions such as
      'x', 'y', '1/y', '-x/y**2' etc. Unary functions should use 'x' as the parameter
      and binary functions should use 'x' and 'y' as the two parameters respectively.
    """
    def __init__(
            self,
            deriv_funcs: Tuple[Optional[PositionalDerivFunc]],
            eval_locals: Optional[Dict[str, Any]] = None,
    ):
        deriv_func_dict = deriv_func_dict_positional_helper(deriv_funcs, eval_locals)
        super().__init__(deriv_func_dict)
