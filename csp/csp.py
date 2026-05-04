import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

'''
1. for every epoch, calculate covariance matrix.
2. for every class, mean covariance.
3. Generalized eigenvalue
4. sort the eigenvalues
5. Projection
6. Log variance
7. repeat to the all epochs.
'''


class MyCsp(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=6):
        self.n_components = n_components
        self.W = None
        self.eigvals = None
        self.classes = None

    def fit(self, X, y):
        _, W, eigvals = compute_csp(X,y,n_components=self.n_components)

        self.W = W
        self.eigvals = eigvals
        self.classes = np.unique(y)

        return self

    def transform(self, X):
        if self.W is None:
            raise RuntimeError("MyCsp must be fitted before transform.")

        X_csp = []

        for epoch in X:
            # Projection
            Z = self.W.T @ epoch

            # Variance
            var = np.var(Z, axis=1)

            # Normalization
            var = var / np.sum(var)

            # Log variance
            log_var = np.log(var + np.finfo(float).eps)

            X_csp.append(log_var)

        return np.array(X_csp)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def matrix_inverse(A):
    n = A.shape[0]
    # A and identity matrix [A | I]
    augmented_m = np.hstack([A.copy().astype(float), np.eye(n)])

    for i in range(n):
        if augmented_m[i, i] == 0:
            for j in range(i + 1, n):
                if augmented_m[j, i] != 0:
                    augmented_m[[i, j]] = augmented_m[[j, i]]
                    break

        augmented_m[i] = augmented_m[i] / augmented_m[i, i]

        for j in range(n):
            if j != i:
                augmented_m[j] = augmented_m[j] - augmented_m[j, i] * augmented_m[i]

    return augmented_m[:, n:]


def compute_csp(X, y, n_components=6):
    epochs, channels, times = X.shape

    classes = np.unique(y)

    if len(classes) != 2:
        raise ValueError("CSP supports exactly 2 classes.")

    if n_components % 2 != 0:
        raise ValueError("n_components must be even.")

    c1, c2 = classes[0], classes[1]

    def mean_cov(epochs):
        covs = []
        for epoch in epochs:
            # epoch @ epoch.T- (14,481) @ (481,14) = (14,14)
            C = epoch @ epoch.T / epoch.shape[1]
            C = C / np.trace(C)
            covs.append(C)
        return np.mean(covs, axis=0)

    C1 = mean_cov(X[y == c1])
    C2 = mean_cov(X[y == c2])

    # print(f"C1 shape: {C1.shape}")
    # print(f"C2 shape: {C2.shape}")

    # Rayleigh Quotient:
    # J(w) = (w.T @ C1 @ w) / (w.T @ C2 @ w)

    # C = C1 + C2
    # C2 = C - C1
    # J(w) = (w.T @ C1 @ w) / (w.T @ (C - C1) @ w)

    # J(w) = (w.T @ C1 @ w) / (w.T @ C @ w)

    # Constraint:
    # w.T @ C @ w = 1

    # Lagrangian:
    # L(w) = w.T @ C1 @ w - lambda * (w.T @ C @ w - 1)

    # Derivative:
    # d/dw(w.T @ A @ w) = 2 @ A @ w

    # 2 @ C1 @ w - 2 * lambda @ C @ w = 0

    # C1 @ w = lambda @ C @ w

    # Since C = C1 + C2:
    # C1 @ w = lambda @ (C1 + C2) @ w

    C = C1 + C2

    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(C) @ C1)

    eigvals = eigvals.real
    eigvecs = eigvecs.real

    ix = np.argsort(eigvals)[::-1]

    eigvals = eigvals[ix]
    eigvecs = eigvecs[:, ix]

    half = n_components // 2

    selected_ix = np.r_[0:half, -half:0]

    W = eigvecs[:, selected_ix]

    X_csp = []

    for epoch in X:
        # Projection
        Z = W.T @ epoch

        # Variance
        var = np.var(Z, axis=1)

        # Normalization
        var = var / np.sum(var)

        # Log variance
        log_var = np.log(var + np.finfo(float).eps)

        X_csp.append(log_var)

    X_csp = np.array(X_csp)

    return X_csp, W, eigvals

'''
def run_tests():
    np.random.seed(42)
    X = np.random.randn(45, 14, 481)
    y = np.array([2] * 24 + [3] * 21)

    csp = MyCsp(n_components=6)
    X_csp = csp.fit_transform(X, y)

    assert X_csp.shape == (45, 6), f"unexpected shape: {X_csp.shape}"
    assert csp.W.shape == (14, 6), f"unexpected shape: {csp.W.shape}"
    assert csp.eigvals.shape == (14,), f"unexpected shape: {csp.eigvals.shape}"

    print("X_csp shape:", X_csp.shape)
    print("W shape:", csp.W.shape)
    print("First epoch features:", X_csp[0])
'''
