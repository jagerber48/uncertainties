import numpy as np

from uncertainties.new import correlated_values, covariance_matrix, UArray


mean_vals = [1, 2, 3]
cov = np.array([
    [1, 0.2, 0.3],
    [0.2, 2, 0.2],
    [0.3, 0.2, 4],
])


def test_covariance():
    ufloats = correlated_values(mean_vals, cov)
    np.testing.assert_array_equal(UArray(ufloats).nominal_value, np.array(mean_vals))
    np.testing.assert_array_almost_equal(covariance_matrix(ufloats), cov)
