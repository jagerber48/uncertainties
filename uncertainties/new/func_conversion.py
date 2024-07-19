from functools import wraps
from math import sqrt
from numbers import Real
import sys
from typing import Any, Callable, Collection, Dict, Optional, Tuple, Union

from uncertainties.new.ufloat import UFloat

SQRT_EPS = sqrt(sys.float_info.epsilon)


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


ParamSpecifier = Union[str, int]
DerivFuncDict = Optional[Dict[ParamSpecifier, Optional[Callable[..., float]]]]


class ToUFunc:
    """
    Decorator which converts a function which accepts real numbers and returns a real
    number into a function which accepts UFloats and returns a UFloat. The returned
    UFloat will have the same value as if the original function had been called using
    the values of the input UFloats. But, additionally, it will have an uncertainty
    corresponding to the square root of the sum of squares of the uncertainties of the
    input UFloats weighted by the partial derivatives of the original function with
    respect to the corresponding input parameters.

    :param ufloat_params: Collection of strings or integers indicating the name or
      position index of the parameters which will be made to accept UFloat.
    :param deriv_func_dict: Dictionary mapping parameters specified in ufloat_params to
      functions that return the partial derivatives of the decorated function with
      respect to the corresponding parameter. The partial derivative functions should
      have the same signature as the decorated function. If any ufloat param is absent
      or is mapped to ``None`` then the partial derivatives will be evaluated
      numerically.
    """
    def __init__(
            self,
            ufloat_params: Collection[ParamSpecifier],
            deriv_func_dict: DerivFuncDict = None,
    ):
        self.ufloat_params = ufloat_params

        if deriv_func_dict is None:
            deriv_func_dict = {}
        for ufloat_param in ufloat_params:
            if ufloat_param not in deriv_func_dict:
                deriv_func_dict[ufloat_param] = None
        self.deriv_func_dict: DerivFuncDict = deriv_func_dict

    def __call__(self, f: Callable[..., float]):
        # sig = inspect.signature(f)

        @wraps(f)
        def wrapped(*args, **kwargs):
            # TODO: The construction below could be simplied using inspect.signature.
            #   However, the math.log, and other math functions do not yet
            #   (as of python 3.12) work with inspect.signature. Therefore, we need to
            #   manually loop of args and kwargs.
            #   Monitor https://github.com/python/cpython/pull/117671
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

            new_uncertainty_lin_combo = []
            for u_float_param in self.ufloat_params:
                if isinstance(u_float_param, int):
                    try:
                        arg = args[u_float_param]
                    except IndexError:
                        continue
                else:
                    try:
                        arg = kwargs[u_float_param]
                    except KeyError:
                        continue
                if isinstance(arg, UFloat):
                    deriv_func = self.deriv_func_dict[u_float_param]
                    if deriv_func is None:
                        derivative = numerical_partial_derivative(
                            f,
                            u_float_param,
                            *float_args,
                            **float_kwargs,
                        )
                    else:
                        derivative = deriv_func(*float_args, **float_kwargs)

                    new_uncertainty_lin_combo.append(
                        (arg.uncertainty_lin_combo, derivative)
                    )
                elif not isinstance(arg, Real):
                    return NotImplemented

            new_uncertainty_lin_combo = tuple(new_uncertainty_lin_combo)
            return UFloat(new_val, new_uncertainty_lin_combo)

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
        if deriv_func is None:
            pass
        elif callable(deriv_func):
            pass
        elif isinstance(deriv_func, str):
            deriv_func = func_str_to_positional_func(deriv_func, nargs, eval_locals)
        else:
            raise ValueError(
                f'Invalid deriv_func: {deriv_func}. Must be None, callable, or a '
                f'string.'
            )
        deriv_func_dict[arg_num] = deriv_func
    return deriv_func_dict


class ToUFuncPositional(ToUFunc):
    """
    Helper decorator for ToUFunc for functions which accept one or two floats as
    positional input parameters and return a float.

    :param deriv_funcs: List of functions or strings specifying a custom partial
      derivative function for each parameter of the wrapped function. There must be an
      element in the list for every parameter of the wrapped function. Elements of the
      list can be callable functions with the same number of positional arguments
      as the wrapped function. They can also be string representations of functions such
      as 'x', 'y', '1/y', '-x/y**2' etc. Unary functions should use 'x' as the parameter
      and binary functions should use 'x' and 'y' as the two parameters respectively.
      An entry of None will cause the partial derivative to be calculated numerically.
    """
    def __init__(
            self,
            deriv_funcs: Tuple[Optional[PositionalDerivFunc]],
            eval_locals: Optional[Dict[str, Any]] = None,
    ):
        ufloat_params = tuple(range(len(deriv_funcs)))
        deriv_func_dict = deriv_func_dict_positional_helper(deriv_funcs, eval_locals)
        super().__init__(ufloat_params, deriv_func_dict)
