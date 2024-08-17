from typing import Sequence

from uncertainties.new.ufloat import UFloat


try:
    import numpy as np
except ImportError:
    np = None
    allow_numpy = False
else:
    allow_numpy = True


def correlated_values(nominal_values, covariance_matrix):
    """
    Return an array of UFloat from a sequence of nominal values and a covariance matrix.
    """
    if not allow_numpy:
        raise ValueError(
            "numpy import failed. Unable to calculate UFloats from covariance matrix."
        )

    n = covariance_matrix.shape[0]
    ufloat_atoms = np.array([UFloat(0, 1) for _ in range(n)])

    try:
        """
        Covariance matrices for linearly independent random variables are
        symmetric and positive-definite so they can be decomposed sa
        C = L * L.T

        with L a lower triangular matrix.
        Let R be a vector of independent random variables with zero mean and
        unity variance. Then consider
        Y = L * R
        and
        Cov(Y) = E[Y * Y.T] = E[L * R * R.T * L.T] = L * E[R * R.t] * L.T
               = L * Cov(R) * L.T = L * I * L.T = L * L.T = C
        where Cov(R) = I because the random variables in V are independent with
        unity variance. So Y defined as above has covariance C.
        """
        L = np.linalg.cholesky(covariance_matrix)
        Y = L @ ufloat_atoms
    except np.linalg.LinAlgError:
        """"
        If two random variables are linearly dependent, e.g. x and y=2*x, then
        their covariance matrix will be degenerate. In this case, a Cholesky
        decomposition is not possible, but an eigenvalue decomposition is. Even
        in this case, covariance matrices are symmetric, so they can be
        decomposed as

        C = U V U^T

        with U orthogonal and V diagonal with non-negative (though possibly
        zero-valued) entries. Let S = sqrt(V) and
        Y = U * S * R
        Then
        Cov(Y) = E[Y * Y.T] = E[U * S * R * R.T * S.T * U.T]
            = U * S * E[R * R.T] * S.T * U.T
            = U * S * I * S.T * U.T
            = U * S * S.T * U.T = U * V * U.T
            = C
        So Y defined as above has covariance C.
        """
        eig_vals, eig_vecs = np.linalg.eigh(covariance_matrix)
        """
        Eigenvalues may be close to zero but still negative. We clip these
        to zero.
        """
        eig_vals = np.clip(eig_vals, a_min=0, a_max=None)
        std_devs = np.diag(np.sqrt(np.clip(eig_vals, a_min=0, a_max=None)))
        Y = np.transpose(eig_vecs @ std_devs @ ufloat_atoms)

    result = np.array(nominal_values) + Y
    return result


def correlated_values_norm(nominal_values, std_devs, correlation_matrix):
    if allow_numpy:
        outer_std_devs = np.outer(std_devs, std_devs)
        cov_mat = correlation_matrix * outer_std_devs
    else:
        n = len(correlation_matrix)
        cov_mat = [[float("nan")] * n] * n
        for i in range(n):
            for j in range(n):
                cov_mat[i][i] = cov_mat[i][j] * np.sqrt(cov_mat[i][i] * cov_mat[j][j])
    return correlated_values(nominal_values, cov_mat)


def covariance_matrix(ufloats: Sequence[UFloat]):
    """
    Return the covariance matrix of a sequence of UFloat.
    """
    n = len(ufloats)
    if allow_numpy:
        cov = np.zeros((n, n))
    else:
        cov = [[0.0 for _ in range(n)] for _ in range(n)]
    atom_weight_dicts = [ufloat.uncertainty.expanded_dict for ufloat in ufloats]
    atom_sets = [set(atom_weight_dict.keys()) for atom_weight_dict in atom_weight_dicts]
    for i in range(n):
        atom_weight_dict_i = atom_weight_dicts[i]
        for j in range(i, n):
            atom_intersection = atom_sets[i].intersection(atom_sets[j])
            if not atom_intersection:
                continue
            atom_weight_dict_j = atom_weight_dicts[j]
            cov_ij = sum(
                atom_weight_dict_i[atom] * atom_weight_dict_j[atom]
                for atom in atom_intersection
            )
            cov[i][j] = cov_ij
            cov[j][i] = cov_ij
    if allow_numpy:
        cov = np.array(cov)
    return cov


def correlation_matrix(ufloats: Sequence[UFloat]):
    cov_mat = covariance_matrix(ufloats)
    if allow_numpy:
        std_devs = np.sqrt(np.diag(cov_mat))
        outer_std_devs = np.outer(std_devs, std_devs)
        corr_mat = cov_mat / outer_std_devs
        corr_mat[cov_mat == 0] = 0
    else:
        n = len(cov_mat)
        corr_mat = [[float("nan") for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                corr_mat[i][i] = cov_mat[i][j] / np.sqrt(cov_mat[i][i] * cov_mat[j][j])
    return corr_mat
