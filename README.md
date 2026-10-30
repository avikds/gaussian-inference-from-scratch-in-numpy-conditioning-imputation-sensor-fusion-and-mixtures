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
- [x] **27.** fit_bmm
- [x] **28.** bits_per_pixel
- [x] **29.** bmm_complete

## Results

```
Part 1: 100000 MVN draws; share beyond the 95% chi-squared(2) contour: 0.0499
        log N(0 | 0, I_50 * 1e-3) via Cholesky: 126.75

Part 2: eq 3.32 check, rho = 0.8 and y2 = 1: E = 0.8  V = 0.36
        Figure 3.7 (N = 10, D = 8, half hidden, true parameters):
          prior-mean imputation    RMSE 0.647 | NLPD 1.050 nats | 95% coverage 1.000
          conditional imputation   RMSE 0.567 | NLPD 0.891 nats | 95% coverage 1.000
          2000 multiple imputations: max |mean of draws - posterior mean| = 0.039
        Wine (119 training wines, 217 hidden entries of 767):
          column mean/variance     RMSE 1.057 | NLPD 1.494 nats | 95% coverage 0.959
          MVN conditional (3.28)   RMSE 0.812 | NLPD 1.205 nats | 95% coverage 0.903

Part 3: Figure 3.8, y = 3: prior N(0,1) -> (1.5, 0.5) | prior N(0,5) -> (2.5, 0.8333)
        Figure 3.10 (a) equal            fused estimate [0.5, -0.5]
        Figure 3.10 (b) sensor 2 better  fused estimate [0.833, -0.167]
        Figure 3.10 (c) anisotropic      fused estimate [0.846, -0.846]
        Faulty-sensor hunt over 30 logs of 10 readings (one sensor degraded 9x):
          evidence picked the degraded sensor in 30/30 logs
          fused-position RMSE: nominal model 0.2213 | diagnosed model 0.1575 | true noise 0.1564

Part 4: iris GMM, 5 EM restarts, final log likelihoods [-299.7, -303.5, -313.8, -299.8, -296.9]
        best restart: hard clusters agree with the species on 0.887 of the flowers
        Iris imputation (100 training flowers, 51 hidden entries):
          column mean/variance     RMSE 0.841 | NLPD 1.278 nats | 95% coverage 0.980
          single Gaussian          RMSE 0.451 | NLPD 0.405 nats | 95% coverage 0.961
          3-component GMM          RMSE 0.417 | NLPD 0.246 nats | 95% coverage 1.000

Part 5: binarized 8x8 digits, 1400 training and 397 held-out images
        K =  1: bits/pixel train 0.565 | held-out 0.569 | bottom-half completion accuracy 0.773, 0.592 bits/pixel
        K = 20: bits/pixel train 0.422 | held-out 0.426 | bottom-half completion accuracy 0.851, 0.457 bits/pixel
        largest mixing weights: [0.135, 0.096, 0.092, 0.087, 0.086]
        prototype of the heaviest component (# = on-probability > 0.5):
          ..###...
          ..####..
          ..#.##..
          ...###..
          ....##..
          .....#..
          ..#..#..
          ..####..
```
