"""
Gaussian Inference from Scratch in NumPy: Conditioning, Imputation, Sensor Fusion and Mixtures scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Gaussian Inference from Scratch in NumPy (Murphy, PML Book 1, chapter 3).
Conditioning and imputation (3.2), linear Gaussian systems and sensor fusion (3.3), Gaussian and Bernoulli mixtures (3.5)."""
import itertools
import numpy as np
from sklearn.datasets import load_digits, load_iris, load_wine
from solution import *  # noqa: F401,F403  (all step functions)


def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def split_and_hide(X, n_train, frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X_train, X_test = X[perm[:n_train]], X[perm[n_train:]]
    X_nan = np.where(rng.random(X_test.shape) < frac, np.nan, X_test)
    return X_train, X_test, X_nan


def fmt(s):
    return f"RMSE {s['rmse']:.3f} | NLPD {s['nlpd']:.3f} nats | 95% coverage {s['coverage95']:.3f}"


def main():
    # --- Part 1: the multivariate Gaussian ---
    S = np.array([[2.0, 0.6], [0.6, 1.0]])
    Y = mvn_sample(np.array([1.0, -1.0]), S, 100000, np.random.default_rng(0))
    d2, _ = mahalanobis_sq(Y, np.array([1.0, -1.0]), S)
    print("Part 1: 100000 MVN draws; share beyond the 95% chi-squared(2) contour:", round(float(np.mean(d2 > 5.991)), 4))
    print("        log N(0 | 0, I_50 * 1e-3) via Cholesky:", round(float(mvn_logpdf(np.zeros(50), np.zeros(50), 1e-3 * np.eye(50))[0]), 2))

    # --- Part 2: conditioning and imputation ---
    h, m, C = condition_gaussian(np.zeros(2), np.array([[1.0, 0.8], [0.8, 1.0]]), [1], [1.0])
    print("\nPart 2: eq 3.32 check, rho = 0.8 and y2 = 1: E =", round(float(m[0]), 4), " V =", round(float(C[0, 0]), 4))
    rng = np.random.default_rng(1)
    A = rng.standard_normal((8, 8))
    S8 = A @ A.T / 8 + 0.1 * np.eye(8)
    mu8 = rng.standard_normal(8)
    Y8 = mvn_sample(mu8, S8, 10, rng)
    X8 = np.where(rng.random(Y8.shape) < 0.5, np.nan, Y8)
    M8, V8 = impute_known(X8, mu8, S8)
    prior = np.where(np.isnan(X8), mu8, X8)
    print("        Figure 3.7 (N = 10, D = 8, half hidden, true parameters):")
    print("          prior-mean imputation   ", fmt(imputation_scores(Y8, X8, prior, np.where(np.isnan(X8), np.diag(S8), 0.0))))
    print("          conditional imputation  ", fmt(imputation_scores(Y8, X8, M8, V8)))
    draws = multiple_impute(X8, mu8, S8, 2000, np.random.default_rng(2))
    hid = np.isnan(X8)
    print("          2000 multiple imputations: max |mean of draws - posterior mean| =",
          round(float(np.abs(draws.mean(axis=0)[hid] - M8[hid]).max()), 3))
    wine = standardize(load_wine().data)
    X_train, X_test, X_nan = split_and_hide(wine, 119, 0.3, 0)
    r = compare_imputers(X_train, X_test, X_nan)
    print(f"        Wine (119 training wines, {int(np.isnan(X_nan).sum())} hidden entries of {X_nan.size}):")
    print("          column mean/variance    ", fmt(r["column"]))
    print("          MVN conditional (3.28)  ", fmt(r["mvn"]))

    # --- Part 3: linear Gaussian systems ---
    print("\nPart 3: Figure 3.8, y = 3: prior N(0,1) ->", tuple(round(v, 4) for v in infer_scalar([3.0], 1.0, 0.0, 1.0)),
          "| prior N(0,5) ->", tuple(round(v, 4) for v in infer_scalar([3.0], 1.0, 0.0, 5.0)))
    ys = [np.array([0.0, -1.0]), np.array([1.0, 0.0])]
    vague = 1e8 * np.eye(2)
    cases = {"(a) equal": [0.01 * np.eye(2)] * 2, "(b) sensor 2 better": [0.05 * np.eye(2), 0.01 * np.eye(2)],
             "(c) anisotropic": [0.01 * np.array([[10.0, 1.0], [1.0, 1.0]]), 0.01 * np.array([[1.0, 1.0], [1.0, 10.0]])]}
    for name, Sig in cases.items():
        print(f"        Figure 3.10 {name:20s} fused estimate", np.round(fuse_sensors(ys, Sig, np.zeros(2), vague)[0], 3).tolist())
    S3 = [0.01 * np.array([[10.0, 1.0], [1.0, 1.0]]), 0.01 * np.array([[1.0, 1.0], [1.0, 10.0]]), 0.04 * np.eye(2)]
    hits, err_nom, err_diag, err_true = 0, [], [], []
    logs = 30
    for seed in range(logs):
        rng = np.random.default_rng(100 + seed)
        bad = seed % 3
        Z = mvn_sample(np.zeros(2), np.eye(2), 10, rng)
        reads = [Z + mvn_sample(np.zeros(2), Sm * (9.0 if m == bad else 1.0), 10, rng) for m, Sm in enumerate(S3)]
        best, _ = diagnose_sensors(reads, S3, np.zeros(2), np.eye(2), (4.0, 9.0, 16.0))
        hits += int(best[0] == bad)
        for errs, Sig in ((err_nom, S3), (err_diag, [Sm * (best[1] if m == best[0] else 1.0) for m, Sm in enumerate(S3)]),
                          (err_true, [Sm * (9.0 if m == bad else 1.0) for m, Sm in enumerate(S3)])):
            F = np.array([fuse_sensors([rd[t] for rd in reads], Sig, np.zeros(2), np.eye(2))[0] for t in range(10)])
            errs.append(np.mean(np.sum((F - Z) ** 2, axis=1)))
    print(f"        Faulty-sensor hunt over {logs} logs of 10 readings (one sensor degraded 9x):")
    print(f"          evidence picked the degraded sensor in {hits}/{logs} logs")
    print("          fused-position RMSE: nominal model", round(float(np.sqrt(np.mean(err_nom))), 4),
          "| diagnosed model", round(float(np.sqrt(np.mean(err_diag))), 4), "| true noise", round(float(np.sqrt(np.mean(err_true))), 4))

    # --- Part 4: Gaussian mixtures ---
    iris = load_iris()
    Xi = standardize(iris.data)
    fits = [fit_gmm(Xi, 3, 100, s) for s in range(5)]
    best = max(fits, key=lambda f: f[3][-1])
    z = hard_assign(Xi, *best[:3])
    agree = max(float(np.mean(np.array(p)[z] == iris.target)) for p in itertools.permutations(range(3)))
    print("\nPart 4: iris GMM, 5 EM restarts, final log likelihoods", [round(f[3][-1], 1) for f in fits])
    print("        best restart: hard clusters agree with the species on", round(agree, 3), "of the flowers")
    X_train, X_test, X_nan = split_and_hide(Xi, 100, 0.3, 0)
    r = compare_imputers(X_train, X_test, X_nan)
    pis, mus, Sigmas, _ = fit_gmm(X_train, 3, 100, 0, reg=1e-2)
    gM, gV = gmm_impute(X_nan, pis, mus, Sigmas)
    print(f"        Iris imputation (100 training flowers, {int(np.isnan(X_nan).sum())} hidden entries):")
    print("          column mean/variance    ", fmt(r["column"]))
    print("          single Gaussian         ", fmt(r["mvn"]))
    print("          3-component GMM         ", fmt(imputation_scores(X_test, X_nan, gM, gV)))

    # --- Part 5: Bernoulli mixtures on binarized digits ---
    Xd = (load_digits().data >= 8).astype(float)
    perm = np.random.default_rng(0).permutation(len(Xd))
    D_train, D_test = Xd[perm[:1400]], Xd[perm[1400:]]
    print(f"\nPart 5: binarized 8x8 digits, {len(D_train)} training and {len(D_test)} held-out images")
    for K in (1, 20):
        pis, mus, trace = fit_bmm(D_train, K, 50, 0)
        hidx, P = bmm_complete(D_test, np.arange(32), pis, mus)
        T = D_test[:, hidx]
        acc = float(np.mean((P > 0.5) == (T > 0.5)))
        xent = float(-np.mean(T * np.log2(P) + (1 - T) * np.log2(1 - P)))
        print(f"        K = {K:2d}: bits/pixel train {bits_per_pixel(D_train, pis, mus):.3f} | held-out {bits_per_pixel(D_test, pis, mus):.3f}"
              f" | bottom-half completion accuracy {acc:.3f}, {xent:.3f} bits/pixel")
        if K == 20:
            print("        largest mixing weights:", np.round(np.sort(pis)[::-1][:5], 3).tolist())
            proto = mus[np.argmax(pis)].reshape(8, 8)
            print("        prototype of the heaviest component (# = on-probability > 0.5):")
            for row in proto:
                print("          " + "".join("#" if p > 0.5 else "." for p in row))


if __name__ == "__main__":
    main()

