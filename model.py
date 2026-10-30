"""
Gaussian Inference from Scratch in NumPy: Conditioning, Imputation, Sensor Fusion and Mixtures

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - mahalanobis_sq
import numpy as np

def mahalanobis_sq(Y, mu, Sigma):
    """Return (delta2, Z): squared Mahalanobis distances (N,) and
    rotated coordinates Z = (Y - mu) @ U (N, D).
    """

    # Convert a single point into a 2D array with shape (1, D).
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    # Eigendecompose the covariance matrix.
    # Eigenvalues are returned in ascending order.
    # Eigenvectors are stored as columns of U.
    eigenvalues, U = np.linalg.eigh(Sigma)

    # Centre the observations by subtracting the mean.
    Y_centered = Y - mu

    # Rotate the centred observations into the eigenvector basis.
    Z = Y_centered @ U

    # Compute squared Mahalanobis distances by scaling
    # each rotated coordinate by its corresponding eigenvalue.
    delta2 = np.sum((Z ** 2) / eigenvalues, axis=1)

    return delta2, Z

# Step 2 - mvn_logpdf
def mvn_logpdf(Y, mu, Sigma):
    """Return log N(y_n | mu, Sigma) for every row of Y,
    shape (N,), via a Cholesky factor of Sigma.
    """

    # Ensure Y is a 2D array, including when a single point is provided.
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    # Get the number of dimensions.
    D = Y.shape[1]

    # Compute the lower-triangular Cholesky factor: Sigma = L @ L.T.
    L = np.linalg.cholesky(Sigma)

    # Centre the observations by subtracting the mean.
    residuals = Y - mu

    # Solve L @ A = residuals.T to obtain whitened residuals.
    A = np.linalg.solve(L, residuals.T)

    # Compute squared Mahalanobis distances.
    delta2 = np.sum(A ** 2, axis=0)

    # Compute the log determinant using the Cholesky diagonal.
    log_det = 2.0 * np.sum(np.log(np.diag(L)))

    # Compute the multivariate Gaussian log density.
    log_density = -0.5 * (
        D * np.log(2.0 * np.pi)
        + log_det
        + delta2
    )

    return log_density

# Step 3 - mvn_sample
def mvn_sample(mu, Sigma, n, rng):
    """Return n samples from N(mu, Sigma), shape (n, D):
    mu + E @ L.T with E = rng.standard_normal((n, D)).
    """

    # Convert inputs to NumPy arrays.
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    # Get the dimensionality of the Gaussian distribution.
    D = mu.shape[0]

    # Compute the lower-triangular Cholesky factor:
    # Sigma = L @ L.T
    L = np.linalg.cholesky(Sigma)

    # Draw all standard normal noise in a single call.
    E = rng.standard_normal((n, D))

    # Transform standard normal samples into samples
    # from the multivariate Gaussian distribution.
    samples = mu + E @ L.T

    return samples

# Step 4 - affine_gaussian
def affine_gaussian(mu, Sigma, A, b):
    """Return (A mu + b, A Sigma A^T) for x ~ N(mu, Sigma);
    A may be a single row.
    """

    # Convert inputs to NumPy arrays.
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    # Handle a one-dimensional A as a single row (M = 1).
    if A.ndim == 1:
        A = A.reshape(1, -1)

    # Compute the transformed mean.
    new_mu = A @ mu + b

    # Compute the transformed covariance.
    new_Sigma = A @ Sigma @ A.T

    # Symmetrize the covariance to eliminate floating-point asymmetry.
    new_Sigma = 0.5 * (new_Sigma + new_Sigma.T)

    return new_mu, new_Sigma

# Step 5 - condition_gaussian
def condition_gaussian(mu, Sigma, idx_obs, y_obs):
    """Return (idx_hid, mu_cond, Sigma_cond) for
    p(y_hid | y_obs) using the moment form of eq 3.28.
    """

    # Convert inputs to NumPy arrays.
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    idx_obs = np.asarray(idx_obs, dtype=int).reshape(-1)
    y_obs = np.asarray(y_obs, dtype=float).reshape(-1)

    # Get the total number of dimensions.
    D = len(mu)

    # Identify hidden coordinates in ascending order.
    idx_hid = np.setdiff1d(np.arange(D), idx_obs)

    # Handle the case where no coordinates are observed.
    if len(idx_obs) == 0:
        return idx_hid, mu.copy(), Sigma.copy()

    # Partition the mean vector.
    mu_hid = mu[idx_hid]
    mu_obs = mu[idx_obs]

    # Partition the covariance matrix.
    Sigma_11 = Sigma[np.ix_(idx_hid, idx_hid)]
    Sigma_12 = Sigma[np.ix_(idx_hid, idx_obs)]
    Sigma_21 = Sigma[np.ix_(idx_obs, idx_hid)]
    Sigma_22 = Sigma[np.ix_(idx_obs, idx_obs)]

    # Calculate the difference between observed values and their mean.
    residual = y_obs - mu_obs

    # Compute the conditional mean:
    # mu_cond = mu_1 + Sigma_12 @ inv(Sigma_22) @ (y_obs - mu_2)
    # Use np.linalg.solve instead of explicitly computing the inverse.
    mu_cond = mu_hid + Sigma_12 @ np.linalg.solve(
        Sigma_22, residual
    )

    # Compute the conditional covariance using the Schur complement:
    # Sigma_cond = Sigma_11 - Sigma_12 @ inv(Sigma_22) @ Sigma_21
    Sigma_cond = Sigma_11 - Sigma_12 @ np.linalg.solve(
        Sigma_22, Sigma_21
    )

    # Symmetrize the conditional covariance to avoid round-off asymmetry.
    Sigma_cond = 0.5 * (Sigma_cond + Sigma_cond.T)

    return idx_hid, mu_cond, Sigma_cond

# Step 6 - condition_gaussian_precision
def condition_gaussian_precision(mu, Sigma, idx_obs, y_obs):
    """Return (idx_hid, mu_cond, Sigma_cond) for
    p(y_hid | y_obs) using the precision matrix Lambda = Sigma^-1.
    """

    # Convert inputs to NumPy arrays.
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    idx_obs = np.asarray(idx_obs, dtype=int).reshape(-1)
    y_obs = np.asarray(y_obs, dtype=float).reshape(-1)

    # Get the total number of dimensions.
    D = len(mu)

    # Identify hidden coordinates in ascending order.
    idx_hid = np.setdiff1d(np.arange(D), idx_obs)

    # Compute the precision matrix.
    # Lambda = Sigma^-1
    Lam = np.linalg.inv(Sigma)

    # Extract the required precision matrix blocks.
    Lam_11 = Lam[np.ix_(idx_hid, idx_hid)]
    Lam_12 = Lam[np.ix_(idx_hid, idx_obs)]

    # Extract the means of the hidden and observed coordinates.
    mu_hid = mu[idx_hid]
    mu_obs = mu[idx_obs]

    # Compute the observed residuals.
    residual = y_obs - mu_obs

    # Compute the conditional covariance:
    # Sigma_cond = inv(Lambda_11)
    Sigma_cond = np.linalg.solve(
        Lam_11,
        np.eye(len(idx_hid))
    )

    # Compute the conditional mean:
    # mu_cond = mu_hid - Sigma_cond @ Lam_12 @ residual
    mu_cond = mu_hid - Sigma_cond @ Lam_12 @ residual

    # Symmetrize the conditional covariance.
    Sigma_cond = 0.5 * (Sigma_cond + Sigma_cond.T)

    return idx_hid, mu_cond, Sigma_cond

# Step 7 - impute_known
def impute_known(X_nan, mu, Sigma):
    """Return (X_mean, X_var): posterior means and variances of the NaN
    entries under N(mu, Sigma); observed entries keep their value
    and get variance 0.
    """

    # Convert inputs to NumPy arrays.
    X_nan = np.asarray(X_nan, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    # Create separate output arrays without modifying the input.
    X_mean = X_nan.copy()
    X_var = np.zeros_like(X_nan, dtype=float)

    # Process each observation independently.
    for n in range(X_nan.shape[0]):

        # Identify missing and observed coordinates.
        missing = np.isnan(X_nan[n])
        idx_hid = np.flatnonzero(missing)
        idx_obs = np.flatnonzero(~missing)

        # If nothing is missing, retain the original row.
        if len(idx_hid) == 0:
            continue

        # Get the observed values.
        y_obs = X_nan[n, idx_obs]

        # Compute the conditional Gaussian distribution
        # using the previously implemented function.
        idx_cond, mu_cond, Sigma_cond = condition_gaussian(
            mu, Sigma, idx_obs, y_obs
        )

        # Replace missing entries with their posterior means.
        X_mean[n, idx_cond] = mu_cond

        # Store posterior variances (diagonal of the conditional covariance).
        X_var[n, idx_cond] = np.diag(Sigma_cond)

    return X_mean, X_var

# Step 8 - multiple_impute
def multiple_impute(X_nan, mu, Sigma, m, rng):
    """Return m posterior draws of the completed matrix, shape (m, N, D);
    hidden blocks are sampled jointly.
    """

    # Convert inputs to NumPy arrays.
    X_nan = np.asarray(X_nan, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    # Get the number of observations and dimensions.
    N, D = X_nan.shape

    # Initialize m copies of the original data matrix.
    # Observed entries remain unchanged in every draw.
    completed = np.broadcast_to(X_nan, (m, N, D)).copy()

    # Process each observation independently.
    for n in range(N):

        # Identify hidden and observed coordinates.
        idx_hid = np.flatnonzero(np.isnan(X_nan[n]))
        idx_obs = np.flatnonzero(~np.isnan(X_nan[n]))

        # Skip rows with no missing entries.
        if len(idx_hid) == 0:
            continue

        # Get the observed values.
        y_obs = X_nan[n, idx_obs]

        # Compute the conditional Gaussian distribution.
        cond_idx, mu_cond, Sigma_cond = condition_gaussian(
            mu, Sigma, idx_obs, y_obs
        )

        # Add a small diagonal jitter for numerical stability.
        H = len(cond_idx)
        Sigma_stable = Sigma_cond + 1e-10 * np.eye(H)

        # Compute the Cholesky factor of the conditional covariance.
        L = np.linalg.cholesky(Sigma_stable)

        # Draw all m standard normal samples in one call.
        E = rng.standard_normal((m, H))

        # Generate joint samples from the conditional Gaussian.
        samples = mu_cond + E @ L.T

        # Fill the missing coordinates in all m completed datasets.
        completed[:, n, cond_idx] = samples

    return completed

