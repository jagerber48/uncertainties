import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from uncertainties.new import ufloat, umath, covariance_matrix
from uncertainties.new.func_conversion import numerical_partial_derivative

from helpers import (
    power_special_cases,
    power_all_cases,
    power_wrt_ref,
    compare_derivatives,
    numbers_close,
    get_single_uatom_and_weight,
)
###############################################################################
# Unit tests


single_input_data_path = Path(Path(__file__).parent, "data", "single_inputs.json")
with open(single_input_data_path, "r") as f:
    single_input_dict = json.load(f)

double_input_data_path = Path(Path(__file__).parent, "data", "double_inputs.json")
with open(double_input_data_path, "r") as f:
    double_input_dict = json.load(f)

real_single_input_funcs = (
    "asinh",
    "atan",
    "cos",
    "cosh",
    "degrees",
    "erf",
    "erfc",
    "exp",
    "radians",
    "sin",
    "sinh",
    "tan",
    "tanh",
)
positive_single_input_funcs = (
    "log",
    "log10",
    "sqrt",
)
minus_one_to_plus_one_single_input_funcs = (
    "acos",
    "asin",
    "atanh",
)
greater_than_one_single_input_funcs = ("acosh",)

real_single_input_cases = list(
    itertools.product(real_single_input_funcs, single_input_dict["real"])
)
positive_single_input_cases = list(
    itertools.product(positive_single_input_funcs, single_input_dict["positive"])
)
minus_one_to_plus_one_single_input_cases = list(
    itertools.product(
        minus_one_to_plus_one_single_input_funcs,
        single_input_dict["minus_one_to_plus_one"],
    )
)
greater_than_one_single_input_cases = list(
    itertools.product(
        greater_than_one_single_input_funcs,
        single_input_dict["greater_than_one"],
    )
)
single_input_cases = (
    real_single_input_cases
    + positive_single_input_cases
    + minus_one_to_plus_one_single_input_cases
    + greater_than_one_single_input_cases
)


@pytest.mark.parametrize("func, value", single_input_cases)
def test_single_input_func_derivatives(func, value):
    uval = ufloat(value, 1.0)

    math_func = getattr(math, func)
    umath_func = getattr(umath, func)

    _, umath_deriv = get_single_uatom_and_weight(umath_func(uval))
    numerical_deriv = numerical_partial_derivative(
        math_func,
        0,
        value,
    )

    assert numbers_close(
        umath_deriv,
        numerical_deriv,
        fractional=True,
        tolerance=1e-4,
    )


real_double_input_funcs = ("atan2",)

positive_double_input_funcs = (
    "hypot",
    "log",
)

real_double_input_cases = list(
    itertools.product(real_double_input_funcs, *zip(*double_input_dict["real"]))
)
print(real_double_input_cases)
positive_double_input_cases = list(
    itertools.product(positive_double_input_funcs, *zip(*double_input_dict["positive"]))
)

double_input_cases = real_double_input_cases + positive_double_input_cases


@pytest.mark.parametrize("func, value_0, value_1", double_input_cases)
def test_double_input_func_derivatives(func, value_0, value_1):
    uval_0 = ufloat(value_0, 1.0)
    uval_1 = ufloat(value_1, 1.0)

    uatom_0, _ = get_single_uatom_and_weight(uval_0)
    uatom_1, _ = get_single_uatom_and_weight(uval_1)

    math_func = getattr(math, func)
    umath_func = getattr(umath, func)

    func_uval = umath_func(uval_0, uval_1)

    umath_deriv_0 = func_uval.error_components[uatom_0]
    umath_deriv_1 = func_uval.error_components[uatom_1]
    numerical_deriv_0 = numerical_partial_derivative(
        math_func,
        0,
        value_0,
        value_1,
    )
    numerical_deriv_1 = numerical_partial_derivative(
        math_func,
        1,
        value_0,
        value_1,
    )

    assert numbers_close(
        umath_deriv_0,
        numerical_deriv_0,
        fractional=True,
        tolerance=1e-4,
    )
    assert numbers_close(
        umath_deriv_1,
        numerical_deriv_1,
        fractional=True,
        tolerance=1e-4,
    )


@pytest.mark.xfail(
    reason="Can't recover this test, no more derivative attribute to use for "
    "compare_derivatives."
)
def test_fixed_derivatives_math_funcs():
    """
    Comparison between function derivatives and numerical derivatives.

    This comparison is useful for derivatives that are analytical.
    """

    for name in umath_core.many_scalars_to_scalar_funcs:
        func = getattr(umath_core, name)
        # Numerical derivatives of func: the nominal value of func() results
        # is used as the underlying function:
        numerical_derivatives = uncert_core.NumericalDerivatives(
            lambda *args: func(*args)
        )
        compare_derivatives(func, numerical_derivatives)

    # Functions that are not in umath_core.many_scalars_to_scalar_funcs:

    ##
    # modf(): returns a tuple:
    def frac_part_modf(x):
        return umath_core.modf(x)[0]

    def int_part_modf(x):
        return umath_core.modf(x)[1]

    compare_derivatives(
        frac_part_modf, uncert_core.NumericalDerivatives(lambda x: frac_part_modf(x))
    )
    compare_derivatives(
        int_part_modf, uncert_core.NumericalDerivatives(lambda x: int_part_modf(x))
    )

    ##
    # frexp(): returns a tuple:
    def mantissa_frexp(x):
        return umath_core.frexp(x)[0]

    def exponent_frexp(x):
        return umath_core.frexp(x)[1]

    compare_derivatives(
        mantissa_frexp, uncert_core.NumericalDerivatives(lambda x: mantissa_frexp(x))
    )
    compare_derivatives(
        exponent_frexp, uncert_core.NumericalDerivatives(lambda x: exponent_frexp(x))
    )


def test_compound_expression():
    """
    Test equality between different formulas.
    """

    x = ufloat(3, 0.1)

    # Prone to numerical errors (but not much more than floats):
    assert umath.tan(x) == umath.sin(x) / umath.cos(x)


def test_numerical_example():
    "Test specific numerical examples"

    x = ufloat(3.14, 0.01)
    result = umath.sin(x)
    # In order to prevent big errors such as a wrong, constant value
    # for all analytical and numerical derivatives, which would make
    # test_fixed_derivatives_math_funcs() succeed despite incorrect
    # calculations:
    assert (
        "%.6f +/- %.6f" % (result.nominal_value, result.std_dev)
        == "0.001593 +/- 0.010000"
    )

    # Regular calculations should still work:
    assert "%.11f" % umath.sin(3) == "0.14112000806"


def test_monte_carlo_comparison():
    """
    Full comparison to a Monte-Carlo calculation.

    Both the nominal values and the covariances are compared between
    the direct calculation performed in this module and a Monte-Carlo
    simulation.
    """

    try:
        import numpy
        import numpy.random
    except ImportError:
        import warnings

        warnings.warn("Test not performed because NumPy is not available")
        return

    # Works on numpy.arrays of Variable objects (whereas umath_core.sin()
    # does not):

    # Example expression (with correlations, and multiple variables combined
    # in a non-linear way):
    def function(x, y):
        """
        Function that takes two NumPy arrays of the same size.
        """
        # The uncertainty due to x is about equal to the uncertainty
        # due to y:
        return 10 * x**2 - x * np.sin(y**3)

    x = ufloat(0.2, 0.01)
    y = ufloat(10, 0.001)
    function_result_this_module = function(x, y)
    nominal_value_this_module = function_result_this_module.nominal_value

    # Covariances "f*f", "f*x", "f*y":
    covariances_this_module = numpy.array(
        covariance_matrix((x, y, function_result_this_module))
    )

    def monte_carlo_calc(n_samples):
        """
        Calculate function(x, y) on n_samples samples and returns the
        median, and the covariances between (x, y, function(x, y)).
        """
        # Result of a Monte-Carlo simulation:
        x_samples = numpy.random.normal(x.nominal_value, x.std_dev, n_samples)
        y_samples = numpy.random.normal(y.nominal_value, y.std_dev, n_samples)

        function_samples = function(x_samples, y_samples)

        cov_mat = numpy.cov([x_samples, y_samples], function_samples)

        return numpy.median(function_samples), cov_mat

    nominal_value_samples, covariances_samples = monte_carlo_calc(1000000)

    assert np.allclose(
        covariances_this_module, covariances_samples, atol=0.01, rtol=0.01
    ), (
        "The covariance matrices do not coincide between"
        " the Monte-Carlo simulation and the direct calculation:\n"
        "* Monte-Carlo:\n%s\n* Direct calculation:\n%s"
        % (covariances_samples, covariances_this_module)
    )

    assert numbers_close(
        nominal_value_this_module,
        nominal_value_samples,
        # The scale of the comparison depends on the standard
        # deviation: the nominal values can differ by a fraction of
        # the standard deviation:
        math.sqrt(covariances_samples[2, 2]) / abs(nominal_value_samples) * 0.5,
    ), (
        "The nominal value (%f) does not coincide with that of"
        " the Monte-Carlo simulation (%f), for a standard deviation of %f."
        % (
            nominal_value_this_module,
            nominal_value_samples,
            math.sqrt(covariances_samples[2, 2]),
        )
    )


def test_math_module():
    "Operations with the math module"

    x = ufloat(-1.5, 0.1)

    # The exponent must not be differentiated, when calculating the
    # following (the partial derivative with respect to the exponent
    # is not defined):
    assert (x**2).nominal_value == 2.25

    # Regular operations are chosen to be unchanged:
    assert isinstance(umath_core.sin(3), float)

    # factorial() must not be "damaged" by the umath_core module, so as
    # to help make it a drop-in replacement for math (even though
    # factorial() does not work on numbers with uncertainties
    # because it is restricted to integers, as for
    # math.factorial()):
    assert umath_core.factorial(4) == 24

    # fsum is special because it does not take a fixed number of
    # variables:
    assert umath_core.fsum([x, x]).nominal_value == -3

    # Functions that give locally constant results are tested: they
    # should give the same result as their float equivalent:
    for name in umath_core.locally_cst_funcs:
        try:
            func = getattr(umath_core, name)
        except AttributeError:
            continue  # Not in the math module, so not in umath_core either

        assert func(x) == func(x.nominal_value)
        # The type should be left untouched. For example, isnan()
        # should always give a boolean:
        assert isinstance(func(x), type(func(x.nominal_value)))

    # The same exceptions should be generated when numbers with uncertainties
    # are used:

    # The type of the expected exception is first determined, because
    # it varies between versions of Python (OverflowError in Python
    # 2.6+, ValueError in Python 2.5,...):
    try:
        math.log(0)
    except Exception as err_math:
        # Python 3 does not make exceptions local variables: they are
        # restricted to their except block:
        err_math_args = err_math.args
        exception_class = err_math.__class__

    try:
        umath_core.log(0)
    except exception_class as err_ufloat:
        assert err_math_args == err_ufloat.args
    else:
        raise Exception("%s exception expected" % exception_class.__name__)
    try:
        umath_core.log(ufloat(0, 0))
    except exception_class as err_ufloat:
        assert err_math_args == err_ufloat.args
    else:
        raise Exception("%s exception expected" % exception_class.__name__)
    try:
        umath_core.log(ufloat(0, 1))
    except exception_class as err_ufloat:
        assert err_math_args == err_ufloat.args
    else:
        raise Exception("%s exception expected" % exception_class.__name__)


def test_hypot():
    """
    Special cases where derivatives cannot be calculated:
    """
    x = ufloat(0, 1)
    y = ufloat(0, 2)
    # Derivatives that cannot be calculated simply return NaN, with no
    # exception being raised, normally:
    result = umath_core.hypot(x, y)
    assert isnan(result.derivatives[x])
    assert isnan(result.derivatives[y])


def test_power_all_cases():
    """
    Test special cases of umath_core.pow().
    """
    power_all_cases(umath_core.pow)


# test_power_special_cases() is similar to
# test_uncertainties.py:test_power_special_cases(), but with small
# differences: the built-in pow() and math.pow() are slightly
# different:
def test_power_special_cases():
    """
    Checks special cases of umath_core.pow().
    """

    power_special_cases(umath_core.pow)

    # We want the same behavior for numbers with uncertainties and for
    # math.pow() at their nominal values.

    positive = ufloat(0.3, 0.01)
    negative = ufloat(-0.3, 0.01)

    # The type of the expected exception is first determined, because
    # it varies between versions of Python (OverflowError in Python
    # 2.6+, ValueError in Python 2.5,...):
    try:
        math.pow(0, negative.nominal_value)
    except Exception as err_math:
        # Python 3 does not make exceptions local variables: they are
        # restricted to their except block:
        err_math_args = err_math.args  # noqa
        exception_class = err_math.__class__  # noqa

    # http://stackoverflow.com/questions/10282674/difference-between-the-built-in-pow-and-math-pow-for-floats-in-python

    try:
        umath_core.pow(ufloat(0, 0.1), negative)
    except exception_class:  # "as err", for Python 2.6+
        pass
    else:
        raise Exception("%s exception expected" % exception_class.__name__)

    try:
        result = umath_core.pow(negative, positive)  # noqa
    except exception_class:  # Assumed: same exception as for pow(0, negative)
        # The reason why it should also fail in Python 3 is that the
        # result of Python 3 is a complex number, which uncertainties
        # does not handle (no uncertainties on complex numbers). In
        # Python 2, this should always fail, since Python 2 does not
        # know how to calculate it.
        pass
    else:
        raise Exception("%s exception expected" % exception_class.__name__)


def test_power_wrt_ref():
    """
    Checks special cases of the umath_core.pow() power operator.
    """
    power_wrt_ref(umath_core.pow, math.pow)
