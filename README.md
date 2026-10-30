# Gaussian Inference from Scratch in NumPy: Conditioning, Imputation, Sensor Fusion and Mixtures

Chapter 3 of Murphy's Probabilistic Machine Learning puts a box around one result, the formula for conditioning a multivariate Gaussian, and then uses it again and again. Build that machinery in NumPy and put it to work on real problems. Evaluate the MVN density through a Cholesky factor and read the Mahalanobis distance off the eigendecomposition, sample with the affine rule, then condition in moment and precision form and use the result to fill in missing wine measurements with honest error bars, scored by RMSE, negative log predictive density and interval coverage. Invert the arrow of a linear Gaussian system to get Bayes rule for Gaussians and its evidence, reproduce the book's shrinkage and sensor-fusion figures, and use the evidence to find which of three tracking sensors has gone bad. Finally move to mixtures: fit a Gaussian mixture by EM with log-sum-exp responsibilities, watch hard clustering collapse to k-means, impute with a mixture of conditionals that beats the single Gaussian, and fit a Bernoulli mixture to binarized digits that discovers digit prototypes, compresses held-out images in bits per pixel and completes the bottom half of a digit from its top half (Murphy, PML Book 1, sections 3.2, 3.3 and 3.5).

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** mahalanobis_sq
- [x] **2.** mvn_logpdf
- [x] **3.** mvn_sample
- [x] **4.** affine_gaussian
- [x] **5.** condition_gaussian
- [x] **6.** condition_gaussian_precision
- [x] **7.** impute_known
- [x] **8.** multiple_impute
- [x] **9.** imputation_scores
- [x] **10.** fit_mvn
- [x] **11.** compare_imputers
- [x] **12.** lgs_joint
- [x] **13.** lgs_posterior
- [x] **14.** lgs_log_evidence
- [x] **15.** infer_scalar
- [x] **16.** infer_vector
- [x] **17.** stack_sensors
- [x] **18.** fuse_sensors
- [x] **19.** diagnose_sensors
- [x] **20.** logsumexp_rows
- [x] **21.** gmm_responsibilities
- [x] **22.** gmm_m_step
- [x] **23.** fit_gmm
- [x] **24.** hard_assign
- [x] **25.** gmm_impute
- [x] **26.** bmm_log_component

---

Built on Deep-ML.
