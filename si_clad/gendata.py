import numpy as np
import scipy.stats as stats
def generate(n, d, delta, true_outliers=None):
    """
    Generate synthetic data from a matrix normal distribution, with optional injected outliers.
    
    Parameters
    ----------
    n : int
        Number of rows (samples) in the generated matrix.
    d : int
        Number of columns (features) in the generated matrix.
    delta : float
        Magnitude of the mean shift added to outlier rows.
    true_outliers : array-like of shape (n_outliers,), optional
        Indices of the rows to be treated as true outliers. If None, randomly selects n//3 samples
        as outliers.

    Returns
    -------
    X : ndarray of shape (n, d)
        Generated data matrix sampled from a matrix normal distribution with optional outlier shifts.
    Sigma : ndarray of shape (n*d, n*d)
        Covariance matrix of the vectorized form of X (i.e., Kronecker product of row and column covariances).
    true_outliers : ndarray
        Array of indices corresponding to the rows that were designated as outliers.
    """
    M = np.zeros((n, d))
    U = np.identity(n)
    V = np.identity(d)
    if true_outliers is None:
        true_outlier_size = n//3
        true_outliers = np.random.choice(np.array(range(n)), size=true_outlier_size, replace=False)
    M[true_outliers] += delta 
    X = M + stats.matrix_normal.rvs(mean=np.zeros((n, d)), rowcov=U, colcov=V)
    Sigma = np.kron(V,U)
    return X, Sigma, true_outliers