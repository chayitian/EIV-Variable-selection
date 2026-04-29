import numpy as np
from sklearn.preprocessing import StandardScaler


class Ridge:
    """
    岭回归（Ridge Regression）

    目标函数：
    (1 / (2n)) * ||y - Xβ||^2 + (alpha / 2) * ||β||^2

    参数
    ----------
    alpha : float
        L2 正则化强度
    fit_intercept : bool
        是否拟合截距项
    normalize : bool
        是否对特征进行标准化
    solver : str
        线性系统求解器，可选 'solve'、'cholesky'、'svd' 或 'gd'，默认 'solve'
    gd_lr : float or None
        solver='gd' 时的学习率；None 表示自动按 Lipschitz 常数取步长
    gd_max_iter : int
        solver='gd' 时最大迭代次数
    gd_tol : float
        solver='gd' 时收敛阈值
    """

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, solver='solve',
                 gd_lr=None, gd_max_iter=5000, gd_tol=1e-8):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.solver = solver
        self.gd_lr = gd_lr
        self.gd_max_iter = gd_max_iter
        self.gd_tol = gd_tol

        self.coef_ = None
        self.intercept_ = None
        self.scaler_X_ = None
        self.scaler_y_ = None

    def _solve_linear_system(self, Sigma_X, rho):
        """
        求解 (Sigma_X + alpha * I) * beta = rho
        """
        n_features = Sigma_X.shape[0]
        ridge_system = Sigma_X + self.alpha * np.eye(n_features)

        if self.solver == 'solve':
            return np.linalg.solve(ridge_system, rho)
        if self.solver == 'cholesky':
            L = np.linalg.cholesky(ridge_system)
            tmp = np.linalg.solve(L, rho)
            return np.linalg.solve(L.T, tmp)
        if self.solver == 'svd':
            U, s, Vt = np.linalg.svd(ridge_system, full_matrices=False)
            s_inv = np.where(s > 1e-12, 1.0 / s, 0.0)
            return Vt.T @ (s_inv * (U.T @ rho))
        if self.solver == 'gd':
            if self.gd_max_iter <= 0:
                raise ValueError("gd_max_iter must be positive")
            if self.gd_tol <= 0:
                raise ValueError("gd_tol must be positive")

            if self.gd_lr is None:
                lipschitz = max(np.max(np.linalg.eigvalsh(ridge_system)), 1e-12)
                lr = 1.0 / lipschitz
            else:
                if self.gd_lr <= 0:
                    raise ValueError("gd_lr must be positive when provided")
                lr = self.gd_lr

            beta = np.zeros_like(rho)
            for _ in range(self.gd_max_iter):
                grad = ridge_system @ beta - rho
                beta_new = beta - lr * grad
                if np.max(np.abs(beta_new - beta)) < self.gd_tol:
                    return beta_new
                beta = beta_new
            return beta
        raise ValueError("solver must be 'solve', 'cholesky', 'svd' or 'gd'")

    def fit(self, X, y):
        """
        拟合Ridge

        参数
        ----------
        X : np.ndarray
            协变量矩阵 (n_samples, n_features)
        y : np.ndarray
            响应变量向量 (n_samples,)
        """
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.solver not in ('solve', 'cholesky', 'svd', 'gd'):
            raise ValueError("solver must be 'solve', 'cholesky', 'svd' or 'gd'")

        n_samples, n_features = X.shape

        if self.normalize:
            self.scaler_X_ = StandardScaler()
            self.scaler_y_ = StandardScaler(with_std=False)

            X_scaled = self.scaler_X_.fit_transform(X)
            y_centered = self.scaler_y_.fit_transform(y.reshape(-1, 1)).flatten()

            X_std = self.scaler_X_.scale_

            Sigma_X = (X_scaled.T @ X_scaled) / n_samples
            min_eig = np.min(np.linalg.eigvalsh(Sigma_X))
            if min_eig < 1e-10:
                Sigma_X += np.eye(n_features) * (1e-10 - min_eig)

            rho = (X_scaled.T @ y_centered) / n_samples

            beta_scaled = self._solve_linear_system(Sigma_X, rho)

            beta_original_scale = beta_scaled / X_std
            self.coef_ = beta_original_scale
            self.intercept_ = self.scaler_y_.mean_ - np.dot(self.scaler_X_.mean_, beta_original_scale)
        else:
            if self.fit_intercept:
                X_mean = np.mean(X, axis=0)
                y_mean = np.mean(y)

                X_centered = X - X_mean
                y_centered = y - y_mean
            else:
                X_mean = np.zeros(n_features)
                y_mean = 0.0

                X_centered = X
                y_centered = y

            Sigma_X = (X_centered.T @ X_centered) / n_samples
            min_eig = np.min(np.linalg.eigvalsh(Sigma_X))
            if min_eig < 1e-10:
                Sigma_X += np.eye(n_features) * (1e-10 - min_eig)

            rho = (X_centered.T @ y_centered) / n_samples

            self.coef_ = self._solve_linear_system(Sigma_X, rho)

            if self.fit_intercept:
                self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
            else:
                self.intercept_ = 0.0

        return self

    def predict(self, X):
        """
        预测

        参数
        ----------
        X : np.ndarray
            协变量矩阵 (n_samples, n_features)

        返回
        -------
        y_pred : np.ndarray
            预测的响应变量 (n_samples,)
        """
        return X @ self.coef_ + self.intercept_
