"""
Gaussian Inference from Scratch in NumPy: Conditioning, Imputation, Sensor Fusion and Mixtures

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - mahalanobis_sq
import numpy as np

def mahalanobis_sq(Y, mu, Sigma):
    """
    Return (delta2, Z):
    - delta2: squared Mahalanobis distances of shape (N,)
    - Z: rotated coordinates of shape (N, D)
    """

    # Convert a single point of shape (D,) into an array of shape (1, D).
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    # Eigendecompose the covariance matrix.
    # Eigenvalues are returned in ascending order.
    eigenvalues, U = np.linalg.eigh(Sigma)

    # Center the observations by subtracting the mean.
    centered_Y = Y - mu

    # Rotate the centered observations into the eigenvector basis.
    # Each column of Z corresponds to an eigenvalue.
    Z = centered_Y @ U

    # Compute the squared Mahalanobis distance:
    # delta2[n] = sum_j (Z[n, j]^2 / eigenvalues[j])
    delta2 = np.sum((Z ** 2) / eigenvalues, axis=1)

    return delta2, Z

