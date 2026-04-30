import numpy as np
from sklearn.linear_model import Lasso


class ALasso:
    """Adaptive Lasso implemented as a thin wrapper around scikit-learn's
    Lasso by column-scaling (feature-wise weights).

    This provides the adaptive weighting strategy while delegating the actual
    L1 optimization to scikit-learn. It requires an external initial
    coefficient estimate (`init_coef`).
    """

    def __init__(self, alpha=1.0, gamma=1.0, max_iter=1000, tol=1e-4, epsilon=1e-6,
                 init_coef=None, solver='cd', admm_rho=None, ista_step_size=None):
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = epsilon
        self.init_coef = init_coef
        self.solver = solver
        self.admm_rho = admm_rho
        self.ista_step_size = ista_step_size

        self.coef_ = None
        self.intercept_ = None
        self.weights_ = None
        self.init_coef_ = None
        self.n_iter_ = 0

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        if self.init_coef is None:
            raise ValueError("ALasso requires externally provided init_coef.")
        init_coef = np.asarray(self.init_coef, dtype=float).reshape(-1)
        if init_coef.size != n_features:
            raise ValueError(f"init_coef length must be {n_features}, got {init_coef.size}")
        if not np.all(np.isfinite(init_coef)):
            raise ValueError("init_coef must contain only finite values")

        self.init_coef_ = init_coef.copy()
        self.weights_ = 1.0 / (np.abs(self.init_coef_) + self.epsilon) ** self.gamma

        inv_weights = 1.0 / self.weights_
        X_weighted = X * inv_weights[np.newaxis, :]

        lasso = Lasso(alpha=self.alpha, fit_intercept=True, max_iter=self.max_iter, tol=self.tol)
        lasso.fit(X_weighted, y)

        coef_scaled = np.asarray(lasso.coef_, dtype=float)
        self.coef_ = coef_scaled * inv_weights
        self.intercept_ = float(getattr(lasso, 'intercept_', 0.0))
        self.n_iter_ = int(getattr(lasso, 'n_iter_', 0))
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_
