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
import numpy as np

def mahalanobis_sq(Y, mu, Sigma):
    """Return (delta2, Z): squared Mahalanobis distances (N,) and rotated coordinates Z = (Y - mu) @ U (N, D)."""

    # Ensure Y is a 2D array, treating a single point as N = 1.
    Y = np.atleast_2d(Y)

    # Eigendecompose the covariance matrix.
    # Eigenvalues are returned in ascending order.
    eigenvalues, U = np.linalg.eigh(Sigma)

    # Center the observations by subtracting the mean.
    Y_centered = Y - mu

    # Rotate the centered observations into the eigenvector basis.
    Z = Y_centered @ U

    # Compute squared Mahalanobis distances.
    # Each rotated coordinate is scaled by its corresponding eigenvalue.
    delta2 = np.sum((Z ** 2) / eigenvalues, axis=1)

    return delta2, Z

