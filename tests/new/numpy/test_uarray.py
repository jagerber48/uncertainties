import numpy as np

import pytest

from uncertainties.new import UArray, UFloat

from tests.helpers import ufloats_close


ufuncs_cases = [
    (np.sin, np.cos),
    (np.exp, np.exp),
]


@pytest.mark.parametrize("ufunc, deriv_func", ufuncs_cases)
def test_ufuncs(ufunc, deriv_func):
    x = UFloat(1, 0.1)
    actual = ufunc(x)
    expected = UFloat(ufunc(x.n), deriv_func(x.n)*x.uncertainty)
    assert ufloats_close(actual, expected)


def test_uarray_from_ufloats():
    x = UFloat(1, 0.1)
    y = UFloat(2, 0.2)
    z = UFloat(3, 0.3)
    uarr_1 = UArray([x, y, z])
    assert uarr_1[0] == x
    assert uarr_1[1] == y
    assert uarr_1[2] == z

    uarr_2 = UArray.from_arrays(
        [x.n, y.n, z.n],
        [x.u, y.u, z.u]
    )

    assert np.all(uarr_1 == uarr_2)


def test_uarray_scalar_ops():
    val_arr = np.array([1, 2, 3])
    unc_arr = np.array([0.1, 0.2, 0.3])
    scalar = 42.314

    uarr = UArray.from_arrays(val_arr, unc_arr)

    # Check __mul__
    assert np.all((uarr*scalar).n == val_arr*scalar)
    assert np.all((uarr*scalar).s == unc_arr*scalar)

    # Check __rmul__
    assert np.all((scalar*uarr).n == scalar*val_arr)
    assert np.all((scalar*uarr).s == scalar*unc_arr)


def test_uarray_ndarray_ops():
    val_arr = np.array([1, 2, 3])
    unc_arr = np.array([0.1, 0.2, 0.3])
    nd_arr = np.array([10, 20, 30])

    uarr = UArray.from_arrays(val_arr, unc_arr)

    # Check __add__
    assert np.all((uarr+nd_arr).n == val_arr+nd_arr)
    assert np.all((uarr+nd_arr).s == unc_arr)

    # Check __radd__
    assert np.all((nd_arr+uarr).n == nd_arr+val_arr)
    assert np.all((nd_arr+uarr).s == unc_arr)


def test_uarray_uarray_ops():
    val_arr_1 = np.array([1, 2, 3])
    unc_arr_1 = np.array([0.1, 0.2, 0.3])
    uarr_1 = UArray.from_arrays(val_arr_1, unc_arr_1)

    val_arr_2 = np.array([10, 20, 30])
    unc_arr_2 = np.array([1, 2, 3])
    uarr_2 = UArray.from_arrays(val_arr_2, unc_arr_2)

    assert np.all((uarr_1+uarr_2).n == val_arr_1+val_arr_2)
    assert np.all((uarr_1+uarr_2).s == np.sqrt(unc_arr_1**2 + unc_arr_2**2))

    assert np.all((uarr_1-uarr_1).n == np.zeros_like(uarr_1))
    assert np.all((uarr_1-uarr_1).s == np.zeros_like(uarr_1))
