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

# Step 9 - imputation_scores
def imputation_scores(X_true, X_nan, X_mean, X_var):
    """Return {"rmse", "nlpd", "coverage95"} over the hidden (NaN)
    entries of X_nan, as Python floats.
    """

    # Convert inputs to NumPy arrays.
    X_true = np.asarray(X_true, dtype=float)
    X_nan = np.asarray(X_nan, dtype=float)
    X_mean = np.asarray(X_mean, dtype=float)
    X_var = np.asarray(X_var, dtype=float)

    # Identify only the entries that were originally missing.
    mask = np.isnan(X_nan)

    # Extract prediction errors and corresponding posterior variances.
    errors = X_mean[mask] - X_true[mask]
    variances = X_var[mask]

    # 1. Root Mean Squared Error (RMSE).
    rmse = np.sqrt(np.mean(errors ** 2))

    # 2. Negative Log Predictive Density (NLPD).
    # Compute the Gaussian negative log density for each hidden entry.
    nlpd = np.mean(
        0.5 * (
            np.log(2.0 * np.pi * variances)
            + (errors ** 2) / variances
        )
    )

    # 3. 95% predictive interval coverage.
    # Check whether the true value lies within mean +/- 1.96 * std.
    coverage95 = np.mean(
        np.abs(errors) <= 1.96 * np.sqrt(variances)
    )

    # Return all metrics as standard Python floats.
    return {
        "rmse": float(rmse),
        "nlpd": float(nlpd),
        "coverage95": float(coverage95)
    }

# Step 10 - fit_mvn
def fit_mvn(X, ridge=0.0):
    """Return the MLE (mu, Sigma) of a Gaussian fit to the rows
    of X, with ridge * I added to Sigma.
    """

    # Convert the input into a NumPy array.
    X = np.asarray(X, dtype=float)

    # Get the number of observations and dimensions.
    N, D = X.shape

    # Compute the column-wise sample mean.
    mu = np.mean(X, axis=0)

    # Centre the observations around the mean.
    X_centered = X - mu

    # Compute the maximum likelihood covariance estimate.
    # MLE uses divisor N rather than N - 1.
    Sigma = (X_centered.T @ X_centered) / N

    # Add ridge regularization to the covariance matrix.
    Sigma += ridge * np.eye(D)

    # Symmetrize the covariance matrix to avoid round-off asymmetry.
    Sigma = 0.5 * (Sigma + Sigma.T)

    return mu, Sigma

# Step 11 - compare_imputers
def compare_imputers(X_train, X_true, X_nan, ridge=1e-3):
    """Return {"column": scores, "mvn": scores} for the column-mean
    baseline and the conditional MVN imputer.
    """

    # Convert inputs to NumPy arrays.
    X_train = np.asarray(X_train, dtype=float)
    X_true = np.asarray(X_true, dtype=float)
    X_nan = np.asarray(X_nan, dtype=float)

    # Fit a multivariate Gaussian to the complete training data.
    mu, Sigma = fit_mvn(X_train, ridge=ridge)

    # Identify missing entries.
    mask = np.isnan(X_nan)

    # --------------------------------------------------
    # 1. Column-mean baseline imputer
    # --------------------------------------------------

    # Fill missing entries using the corresponding column mean.
    # Preserve all observed entries.
    X_mean_column = np.where(mask, mu, X_nan)

    # Assign the marginal variance of each column to missing entries.
    # Observed entries receive zero variance.
    X_var_column = np.where(mask, np.diag(Sigma), 0.0)

    # Evaluate the column-mean imputer.
    column_scores = imputation_scores(
        X_true,
        X_nan,
        X_mean_column,
        X_var_column
    )

    # --------------------------------------------------
    # 2. Conditional MVN imputer
    # --------------------------------------------------

    # Impute missing entries using conditional Gaussian means
    # and their corresponding posterior variances.
    X_mean_mvn, X_var_mvn = impute_known(
        X_nan,
        mu,
        Sigma
    )

    # Evaluate the conditional MVN imputer.
    mvn_scores = imputation_scores(
        X_true,
        X_nan,
        X_mean_mvn,
        X_var_mvn
    )

    # Return the scores for both imputers.
    return {
        "column": column_scores,
        "mvn": mvn_scores
    }

# Step 12 - lgs_joint
def lgs_joint(mu_z, Sigma_z, W, b, Sigma_y):
    """Return (mu, Sigma) of the joint Gaussian over (z, y)
    for p(z) = N(mu_z, Sigma_z),
    p(y|z) = N(Wz + b, Sigma_y).
    """

    # Convert inputs to NumPy arrays.
    mu_z = np.asarray(mu_z, dtype=float)
    Sigma_z = np.asarray(Sigma_z, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    Sigma_y = np.asarray(Sigma_y, dtype=float)

    # Compute the mean of y:
    # E[y] = W @ mu_z + b
    mu_y = W @ mu_z + b

    # Compute the covariance between z and y:
    # Cov(z, y) = Sigma_z @ W.T
    Sigma_zy = Sigma_z @ W.T

    # Compute the covariance of y:
    # Cov(y) = W @ Sigma_z @ W.T + Sigma_y
    Sigma_yy = W @ Sigma_z @ W.T + Sigma_y

    # Construct the joint mean of the stacked vector (z, y).
    mu = np.concatenate((mu_z, mu_y))

    # Construct the joint covariance matrix:
    # [ Cov(z, z)   Cov(z, y) ]
    # [ Cov(y, z)   Cov(y, y) ]
    Sigma = np.block([
        [Sigma_z,    Sigma_zy],
        [Sigma_zy.T, Sigma_yy]
    ])

    # Symmetrize the joint covariance matrix.
    Sigma = 0.5 * (Sigma + Sigma.T)

    return mu, Sigma

# Step 13 - lgs_posterior
def lgs_posterior(mu_z, Sigma_z, W, b, Sigma_y, y):
    """Return (mu_post, Sigma_post) of p(z | y)
    by Bayes rule for Gaussians (eq 3.37).
    """

    # Convert inputs to NumPy arrays.
    mu_z = np.asarray(mu_z, dtype=float)
    Sigma_z = np.asarray(Sigma_z, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    Sigma_y = np.asarray(Sigma_y, dtype=float)
    y = np.asarray(y, dtype=float)

    # Get the dimensionality of the latent variable.
    L = len(mu_z)

    # Compute the prior precision matrix.
    Lambda_z = np.linalg.solve(Sigma_z, np.eye(L))

    # Compute the observation noise precision matrix.
    D = len(y)
    Lambda_y = np.linalg.solve(Sigma_y, np.eye(D))

    # Compute the posterior precision matrix:
    # Lambda_post = inv(Sigma_z) + W.T @ inv(Sigma_y) @ W
    Lambda_post = Lambda_z + W.T @ Lambda_y @ W

    # Compute the posterior covariance by solving the
    # linear system instead of explicitly inverting Lambda_post.
    Sigma_post = np.linalg.solve(
        Lambda_post,
        np.eye(L)
    )

    # Compute the information-weighted posterior mean:
    # eta_post = inv(Sigma_y) @ (y - b) transformed by W.T,
    # plus the prior information Lambda_z @ mu_z.
    eta_post = (
        W.T @ Lambda_y @ (y - b)
        + Lambda_z @ mu_z
    )

    # Solve for the posterior mean.
    mu_post = np.linalg.solve(Lambda_post, eta_post)

    # Symmetrize the posterior covariance.
    Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)

    return mu_post, Sigma_post

# Step 14 - lgs_log_evidence
def lgs_log_evidence(mu_z, Sigma_z, W, b, Sigma_y, y):
    """Return log p(y) = log N(y | W mu_z + b,
    Sigma_y + W Sigma_z W.T) as a Python float.
    """

    # Convert inputs to NumPy arrays.
    mu_z = np.asarray(mu_z, dtype=float)
    Sigma_z = np.asarray(Sigma_z, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    Sigma_y = np.asarray(Sigma_y, dtype=float)
    y = np.asarray(y, dtype=float)

    # Compute the marginal mean of the observation:
    # E[y] = W @ mu_z + b
    mu_y = W @ mu_z + b

    # Compute the marginal covariance of the observation:
    # Cov(y) = Sigma_y + W @ Sigma_z @ W.T
    Sigma_marginal = Sigma_y + W @ Sigma_z @ W.T

    # Symmetrize the marginal covariance matrix.
    Sigma_marginal = 0.5 * (
        Sigma_marginal + Sigma_marginal.T
    )

    # Evaluate the marginal Gaussian log density
    # using the previously implemented mvn_logpdf function.
    log_evidence = mvn_logpdf(
        y,
        mu_y,
        Sigma_marginal
    )

    # Return a standard Python float.
    return float(log_evidence[0])

# Step 15 - infer_scalar
def infer_scalar(ys, sigma2, mu0, tau0_sq):
    """Return (mu_N, tau2_N), the posterior of a scalar z after
    measurements ys ~ N(z, sigma2), prior N(mu0, tau0_sq).
    """

    # Convert measurements into a one-dimensional NumPy array.
    ys = np.asarray(ys, dtype=float).reshape(-1)

    # Number of measurements.
    N = len(ys)

    # Construct the prior distribution for the scalar latent variable.
    mu_z = np.array([mu0], dtype=float)
    Sigma_z = np.array([[tau0_sq]], dtype=float)

    # Construct the linear Gaussian observation model:
    # y = W @ z + b + noise
    W = np.ones((N, 1))
    b = np.zeros(N)

    # Observation noise covariance: sigma2 * I_N.
    Sigma_y = sigma2 * np.eye(N)

    # Compute the posterior using the previously implemented function.
    mu_post, Sigma_post = lgs_posterior(
        mu_z,
        Sigma_z,
        W,
        b,
        Sigma_y,
        ys
    )

    # Return posterior mean and variance as Python floats.
    return float(mu_post[0]), float(Sigma_post[0, 0])

# Step 16 - infer_vector
def infer_vector(Y, Sigma_y, mu_z, Sigma_z):
    """Return (mu_post, Sigma_post) of z after the rows of Y
    were measured as y_n ~ N(z, Sigma_y), prior N(mu_z, Sigma_z).
    """

    # Convert inputs to NumPy arrays.
    Y = np.asarray(Y, dtype=float)
    Sigma_y = np.asarray(Sigma_y, dtype=float)
    mu_z = np.asarray(mu_z, dtype=float)
    Sigma_z = np.asarray(Sigma_z, dtype=float)

    # Get the number of measurements and dimensions.
    N, L = Y.shape

    # Compute the prior and observation precision matrices.
    Lambda_z = np.linalg.solve(Sigma_z, np.eye(L))
    Lambda_y = np.linalg.solve(Sigma_y, np.eye(L))

    # Compute the posterior precision:
    # Lambda_post = inv(Sigma_z) + N * inv(Sigma_y)
    Lambda_post = Lambda_z + N * Lambda_y

    # Compute the posterior covariance.
    Sigma_post = np.linalg.solve(
        Lambda_post,
        np.eye(L)
    )

    # Compute the sufficient statistic: sum of all observations.
    y_sum = np.sum(Y, axis=0)

    # Compute the information-weighted posterior mean:
    # eta_post = inv(Sigma_y) @ sum(Y) + inv(Sigma_z) @ mu_z
    eta_post = Lambda_y @ y_sum + Lambda_z @ mu_z

    # Solve for the posterior mean.
    mu_post = np.linalg.solve(Lambda_post, eta_post)

    # Symmetrize the posterior covariance.
    Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)

    return mu_post, Sigma_post

# Step 17 - stack_sensors
def stack_sensors(ys, Sigmas):
    """Return (y, W, Sigma_y): concatenated readings,
    W = [I; I; ...] and the block-diagonal noise covariance.
    """

    # Convert sensor readings and covariance matrices to NumPy arrays.
    ys = [np.asarray(y, dtype=float).reshape(-1) for y in ys]
    Sigmas = [np.asarray(Sigma, dtype=float) for Sigma in Sigmas]

    # Determine the number of sensors and the dimension of each reading.
    M = len(ys)
    L = ys[0].shape[0]

    # Concatenate all sensor readings into a single vector.
    y = np.concatenate(ys)

    # Stack M identity matrices vertically.
    # Each sensor measures the same unknown vector z.
    I = np.eye(L)
    W = np.vstack([I for _ in range(M)])

    # Construct the block-diagonal observation noise covariance.
    Sigma_y = np.zeros((M * L, M * L), dtype=float)

    for i in range(M):
        start = i * L
        end = start + L

        # Place each sensor's covariance in its corresponding block.
        Sigma_y[start:end, start:end] = Sigmas[i]

    return y, W, Sigma_y

# Step 18 - fuse_sensors
def fuse_sensors(ys, Sigmas, mu_z, Sigma_z):
    """Return (mu_post, Sigma_post) of z given one reading per sensor,
    y_m ~ N(z, Sigmas[m]), prior N(mu_z, Sigma_z).
    """

    # Convert prior parameters to NumPy arrays.
    mu_z = np.asarray(mu_z, dtype=float)
    Sigma_z = np.asarray(Sigma_z, dtype=float)

    # Stack all sensor readings into a single observation model.
    y, W, Sigma_y = stack_sensors(ys, Sigmas)

    # The observation model has zero bias.
    b = np.zeros(y.shape[0], dtype=float)

    # Compute the posterior distribution using Bayes rule
    # for linear Gaussian systems.
    mu_post, Sigma_post = lgs_posterior(
        mu_z,
        Sigma_z,
        W,
        b,
        Sigma_y,
        y
    )

    # Symmetrize the posterior covariance for numerical stability.
    Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)

    return mu_post, Sigma_post

# Step 19 - diagnose_sensors
def diagnose_sensors(readings, Sigmas, mu_z, Sigma_z, scales):
    """Return (best, table): table maps (None, 1.0) and every
    (m, s) hypothesis to its total log evidence.
    """

    # Convert readings and covariance matrices to NumPy arrays.
    readings = [np.asarray(r, dtype=float) for r in readings]
    Sigmas = [np.asarray(S, dtype=float) for S in Sigmas]

    M = len(readings)
    T, L = readings[0].shape

    # Start with the hypothesis that all sensors are nominal.
    hypotheses = [(None, 1.0)]

    # Add hypotheses in which exactly one sensor is degraded.
    for m in range(M):
        for s in scales:
            hypotheses.append((m, float(s)))

    # Store the total log evidence for every hypothesis.
    table = {}

    # Evaluate each sensor degradation hypothesis.
    for m, s in hypotheses:

        # Construct the covariance matrices under this hypothesis.
        # Only sensor m receives the candidate scaling factor.
        covariances = [
            S * s if j == m else S.copy()
            for j, S in enumerate(Sigmas)
        ]

        total_log_evidence = 0.0

        # Evaluate the observations at every time step.
        for t in range(T):

            # Extract the reading from each sensor at time t.
            ys_t = [readings[j][t] for j in range(M)]

            # Stack the sensor readings and their noise covariances.
            y, W, Sigma_y = stack_sensors(ys_t, covariances)

            # The observation model has zero bias.
            b = np.zeros(M * L)

            # Calculate the marginal log evidence for this time step.
            log_evidence = lgs_log_evidence(
                mu_z,
                Sigma_z,
                W,
                b,
                Sigma_y,
                y
            )

            # Accumulate the evidence across all time steps.
            total_log_evidence += log_evidence

        # Store the total evidence for the current hypothesis.
        table[(m, s)] = float(total_log_evidence)

    # Select the hypothesis with the highest total log evidence.
    best = max(table, key=table.get)

    return best, table

# Step 20 - logsumexp_rows
def logsumexp_rows(A):
    """Return log(sum_k exp(A[n, k])) for every row n,
    shape (N,), computed stably.
    """

    # Convert input to a NumPy array.
    A = np.asarray(A, dtype=float)

    # Find the maximum value in each row.
    # keepdims=True preserves the shape for broadcasting.
    row_max = np.max(A, axis=1, keepdims=True)

    # Subtract the row maximum before exponentiation
    # to prevent numerical overflow and underflow.
    shifted = A - row_max

    # Compute log(sum(exp(shifted))) and add back the maximum.
    result = np.log(np.sum(np.exp(shifted), axis=1)) + row_max[:, 0]

    return result

# Step 21 - gmm_responsibilities
def gmm_responsibilities(Y, pis, mus, Sigmas):
    """Return (R, loglik): responsibilities r_nk of shape (N, K)
    and the total log likelihood (float).
    """

    # Convert inputs to NumPy arrays.
    Y = np.asarray(Y, dtype=float)
    pis = np.asarray(pis, dtype=float)
    mus = np.asarray(mus, dtype=float)
    Sigmas = np.asarray(Sigmas, dtype=float)

    # Get the number of observations and mixture components.
    N, D = Y.shape
    K = len(pis)

    # Initialize the matrix of log joint probabilities.
    log_joint = np.empty((N, K), dtype=float)

    # Compute log joint probabilities for every observation
    # and every Gaussian mixture component.
    for k in range(K):
        log_joint[:, k] = (
            np.log(pis[k])
            + mvn_logpdf(Y, mus[k], Sigmas[k])
        )

    # Compute the log marginal probability of each observation
    # using the numerically stable log-sum-exp function.
    log_normalizer = logsumexp_rows(log_joint)

    # Normalize the log joint probabilities to obtain responsibilities.
    # Each row of R sums to 1.
    R = np.exp(log_joint - log_normalizer[:, None])

    # Compute the total log likelihood across all observations.
    loglik = float(np.sum(log_normalizer))

    return R, loglik

# Step 22 - gmm_m_step
def gmm_m_step(Y, R, reg=1e-6):
    """Return (pis, mus, Sigmas) maximising the expected
    complete-data log likelihood for responsibilities R.
    """

    # Convert inputs to NumPy arrays.
    Y = np.asarray(Y, dtype=float)
    R = np.asarray(R, dtype=float)

    # Get the number of observations, dimensions and components.
    N, D = Y.shape
    K = R.shape[1]

    # Compute the effective number of observations assigned
    # to each Gaussian component.
    Nk = np.sum(R, axis=0)

    # Update the mixture weights.
    pis = Nk / N

    # Update the component means using weighted averages.
    mus = (R.T @ Y) / Nk[:, None]

    # Compute deviations from the corresponding component means.
    diff = Y[:, None, :] - mus[None, :, :]

    # Update covariance matrices using weighted outer products.
    Sigmas = np.einsum(
        'nk,nkd,nke->kde',
        R,
        diff,
        diff
    ) / Nk[:, None, None]

    # Add diagonal regularization to each covariance matrix.
    Sigmas += reg * np.eye(D)[None, :, :]

    # Symmetrize covariance matrices to eliminate
    # floating-point asymmetry.
    Sigmas = 0.5 * (
        Sigmas + np.swapaxes(Sigmas, 1, 2)
    )

    return pis, mus, Sigmas

# Step 23 - fit_gmm
def fit_gmm(Y, K, iters, seed, reg=1e-6):
    """Fit a K-component GMM by EM. Return (pis, mus, Sigmas, trace)
    with trace = log likelihood before each M step.
    """

    # Convert input into a NumPy array.
    Y = np.asarray(Y, dtype=float)

    # Get the number of observations and dimensions.
    N, D = Y.shape

    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # 1. Initialize mixture parameters
    # --------------------------------------------------

    # Initialize component means using K distinct random data rows.
    mus = Y[rng.choice(N, K, replace=False)].copy()

    # Initialize all mixture weights uniformly.
    pis = np.full(K, 1.0 / K)

    # Compute the maximum likelihood covariance of the entire dataset.
    data_mean = np.mean(Y, axis=0)
    centered = Y - data_mean

    Sigma0 = (centered.T @ centered) / N

    # Add regularization to the initial covariance.
    Sigma0 += reg * np.eye(D)

    # Symmetrize the covariance matrix.
    Sigma0 = 0.5 * (Sigma0 + Sigma0.T)

    # Initialize every component with an independent copy
    # of the same data covariance matrix.
    Sigmas = np.repeat(Sigma0[None, :, :], K, axis=0)

    # Store the log likelihood before each M step.
    trace = []

    # --------------------------------------------------
    # 2. Expectation-Maximization iterations
    # --------------------------------------------------

    for _ in range(iters):

        # E-step: calculate responsibilities and log likelihood.
        R, loglik = gmm_responsibilities(
            Y, pis, mus, Sigmas
        )

        # Record the log likelihood before the M step.
        trace.append(float(loglik))

        # M-step: update mixture weights, means and covariances.
        pis, mus, Sigmas = gmm_m_step(
            Y, R, reg=reg
        )

    return pis, mus, Sigmas, trace

# Step 24 - hard_assign
def hard_assign(Y, pis, mus, Sigmas):
    """Return the most probable component index for every row of Y
    (eq 3.100), shape (N,).
    """

    # Convert inputs to NumPy arrays.
    Y = np.asarray(Y, dtype=float)
    pis = np.asarray(pis, dtype=float)
    mus = np.asarray(mus, dtype=float)
    Sigmas = np.asarray(Sigmas, dtype=float)

    # Get the number of observations and mixture components.
    N = Y.shape[0]
    K = len(pis)

    # Store the log joint probabilities for every observation
    # and every mixture component.
    log_joint = np.empty((N, K), dtype=float)

    # Compute log N(y_n | mu_k, Sigma_k) + log(pi_k).
    for k in range(K):
        log_joint[:, k] = (
            mvn_logpdf(Y, mus[k], Sigmas[k])
            + np.log(pis[k])
        )

    # Select the component with the highest log joint probability.
    # np.argmax selects the lowest index in case of ties.
    assignments = np.argmax(log_joint, axis=1)

    return assignments

