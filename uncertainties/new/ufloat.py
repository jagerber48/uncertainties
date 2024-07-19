from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from math import sqrt, isnan
from numbers import Real
import sys
from typing import (
    Any, Callable, Collection, Dict, Optional, Tuple, Union, TYPE_CHECKING,
)
import uuid

if TYPE_CHECKING:
    from inspect import Signature


@dataclass(frozen=True)
class UncertaintyAtom:
    """
    Custom class to keep track of "atoms" of uncertainty. Two UncertaintyAtoms are
    always uncorrelated.
    """
    std_dev: float
    uuid: uuid.UUID = field(default_factory=uuid.uuid4, init=False)


"""
UncertaintyCombo represents a (possibly nested) linear superposition of 
UncertaintyAtoms. The UncertaintyCombo is an n-tuple of terms in the linear 
superposition and each term is represented by a 2-tuple. The second element of the
2-tuple is the weight of that term. The first element is either an UncertaintyAtom or
another UncertaintyCombo. In the latter case the original UncertaintyCombo is nested.

By passing the weights through the linear combinations and collecting like terms, any 
UncertaintyCombo can be expanded into a form where each term is an UncertaintyAtom. This
would be an ExpandedUncertaintyCombo.

Nested UncertaintyCombos are supported as a performance optimization. There is a
cost to expanding linear combinations during uncertainty propagation calculations. 
Supporting nested UncertaintyCombos allows expansion to be deferred through intermediate
calculations until a standard deviation or correlation must be calculated at the end of
an error propagation calculation.
"""
# TODO: How much does this optimization quantitatively improve performance?
UncertaintyCombo = Tuple[
    Tuple[
        Union[UncertaintyAtom, "UncertaintyCombo"],
        float
    ],
    ...
]
ExpandedUncertaintyCombo = Tuple[
    Tuple[
        UncertaintyAtom,
        float
    ],
    ...
]


@lru_cache(maxsize=None)
def get_expanded_combo(combo: UncertaintyCombo) -> ExpandedUncertaintyCombo:
    """
    Recursively expand a nested UncertaintyCombo into an ExpandedUncertaintyCombo whose
    terms all represent weighted UncertaintyAtoms.
    """
    expanded_dict = defaultdict(float)
    for combo, combo_weight in combo:
        if isinstance(combo, UncertaintyAtom):
            expanded_dict[combo] += combo_weight
        else:
            expanded_combo = get_expanded_combo(combo)
            for atom, atom_weight in expanded_combo:
                expanded_dict[atom] += atom_weight * combo_weight

    pruned_expanded_dict = {}
    for atom, weight in expanded_dict.items():
        if atom.std_dev == 0 or (weight == 0 and not isnan(atom.std_dev)):
            continue
        pruned_expanded_dict[atom] = weight

    return tuple((atom, weight) for atom, weight in pruned_expanded_dict.items())


@lru_cache(maxsize=None)
def get_std_dev(combo: UncertaintyCombo) -> float:
    """
    Get the standard deviation corresponding to an UncertaintyCombo. The UncertainyCombo
    is expanded and the weighted UncertaintyAtoms are added in quadrature.
    """
    expanded_combo = get_expanded_combo(combo)
    list_of_squares = [
        (weight*atom.std_dev)**2 for atom, weight in expanded_combo
    ]
    std_dev = sqrt(sum(list_of_squares))
    return std_dev


class UFloat:
    """
    Core class. Stores a mean value (value, nominal_value, n) and an uncertainty stored
    as a (possibly unexpanded) linear combination of uncertainty atoms. Two UFloat's
    which share non-zero weight for a certain uncertainty atom are correlated.

    UFloats can be combined using arithmetic and more sophisticated mathematical
    operations. The uncertainty is propagtaed using the rules of linear uncertainty
    propagation.
    """
    def __init__(
            self,
            /,
            value: Real,
            uncertainty: Union[UncertaintyCombo, Real] = (),
            tag: Optional[str] = None
    ):
        self._val = float(value)
        if isinstance(uncertainty, Real):
            if uncertainty < 0:
                raise ValueError(
                    f'Uncertainty must be non-negative, not {uncertainty}.'
                )
            atom = UncertaintyAtom(float(uncertainty))
            uncertainty_combo = ((atom, 1.0),)
            self.uncertainty_lin_combo = uncertainty_combo
        else:
            self.uncertainty_lin_combo = uncertainty
        self.tag = tag

    @property
    def val(self: "UFloat") -> float:
        return self._val

    @property
    def std_dev(self: "UFloat") -> float:
        return get_std_dev(self.uncertainty_lin_combo)

    @property
    def uncertainty(self: "UFloat") -> UncertaintyCombo:
        return self.uncertainty_lin_combo

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.val}, {self.std_dev})'

    def __bool__(self):
        return self != UFloat(0, 0)

    # Aliases
    @property
    def nominal_value(self: "UFloat") -> float:
        return self.val

    @property
    def n(self: "UFloat") -> float:
        return self.val

    @property
    def value(self: "UFloat") -> float:
        return self.val

    @property
    def s(self: "UFloat") -> float:
        return self.std_dev

    def __eq__(self: "UFloat", other: "UFloat") -> bool:
        if not isinstance(other, UFloat):
            return False
        val_eq = self.val == other.val

        self_expanded_linear_combo = get_expanded_combo(self.uncertainty_lin_combo)
        other_expanded_linear_combo = get_expanded_combo(other.uncertainty_lin_combo)
        uncertainty_eq = self_expanded_linear_combo == other_expanded_linear_combo
        return val_eq and uncertainty_eq

    def __pos__(self: "UFloat") -> "UFloat": ...

    def __neg__(self: "UFloat") -> "UFloat": ...

    def __abs__(self: "UFloat") -> "UFloat": ...

    def __trunc__(self: "UFloat") -> "UFloat": ...

    def __add__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __radd__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __sub__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __rsub__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __mul__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __rmul__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __truediv__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __rtruediv__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __pow__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __rpow__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __mod__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...

    def __rmod__(self: "UFloat", other: Union["UFloat", Real]) -> "UFloat": ...


SQRT_EPS = sqrt(sys.float_info.epsilon)


def get_param_name(sig: Signature, param: Union[int, str]):
    if isinstance(param, int):
        param_name = list(sig.parameters.keys())[param]
    else:
        param_name = param
    return param_name


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

    # TODO: The construction below could be simplied using inspect.signature. However,
    #   the math.log, and other math functions do not yet (as of python 3.12) work with
    #   inspect.signature. Therefore, we need to manually loop of args and kwargs.
    #   Monitor https://github.com/python/cpython/pull/117671
    lower_args = []
    upper_args = []
    for idx, arg in enumerate(args):
        if idx == target_param:
            lower_args.append(x - dx)
            upper_args.append(x + dx)
        else:
            lower_args.append(arg)
            upper_args.append(arg)

    lower_kwargs = {}
    upper_kwargs = {}
    for key, arg in kwargs.items():
        if key == target_param:
            lower_kwargs[key] = x - dx
            upper_kwargs[key] = x + dx
        else:
            lower_kwargs[key] = arg
            upper_kwargs[key] = arg

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


def ufloat(val: Real, unc: Real, tag=None) -> UFloat:
    return UFloat(val, unc, tag)
