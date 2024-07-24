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


def test_binary_ops():
    uarr = UArray.from_arrays([1, 2, 3], [0.1, 0.2, 0.3])

    assert np.all((uarr + uarr).n == np.array([2, 4, 6]))
    assert np.all((uarr + uarr).expanded_uncertainty == (2 * uarr).expanded_uncertainty)

    narr = np.array([10, 20, 30])

