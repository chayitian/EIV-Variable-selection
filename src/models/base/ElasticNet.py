import numpy as np


class ElasticNet:
    """
    弹性网络回归（Elastic Net，坐标下降法实现）

    目标函数：
    (1 / (2n)) * ||y - Xβ||^2
    + alpha * l1_ratio * ||β||_1
    + 0.5 * alpha * (1 - l1_ratio) * ||β||^2

    参数
    ----------
    alpha : float
        正则化强度
    l1_ratio : float
        L1 比例，取值范围 [0, 1]
        - l1_ratio=1.0 时退化为 Lasso
        - l1_ratio=0.0 时退化为 Ridge
    max_iter : int
        最大迭代次数
    tol : float
        收敛阈值
    solver : str
        求解方法，可选 'cd'（坐标下降）、'admm' 或 'ista'，默认 'cd'
    admm_rho : float
        ADMM 增广拉格朗日参数（solver='admm' 时生效）
    ista_step_size : float or None
        ISTA 步长；None 表示按 Lipschitz 常数自动设置
    """

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4,
                 solver='cd', admm_rho=1.0, ista_step_size=None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.admm_rho = admm_rho
        self.ista_step_size = ista_step_size

        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0

    def _soft_threshold(self, rho, lam):
        """
        软阈值函数
        """
        return np.sign(rho) * np.maximum(np.abs(rho) - lam, 0.0)

    def _fit_coordinate_descent(self, X_centered, y_centered):
        n_samples, n_features = X_centered.shape
        beta = np.zeros(n_features)

        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1.0 - self.l1_ratio)

        for iteration in range(self.max_iter):
            beta_old = beta.copy()

            for j in range(n_features):
                residual = y_centered - X_centered @ beta + beta[j] * X_centered[:, j]
                rho = np.dot(X_centered[:, j], residual) / n_samples
                z = np.sum(X_centered[:, j] ** 2) / n_samples + l2_penalty

                z_safe = max(z, 1e-12)
                beta[j] = self._soft_threshold(rho, l1_penalty) / z_safe

            if np.max(np.abs(beta - beta_old)) < self.tol:
                self.n_iter_ = iteration + 1
                return beta

        self.n_iter_ = self.max_iter
        return beta

    def _fit_admm(self, X_centered, y_centered):
        n_samples, n_features = X_centered.shape

        rho = self.admm_rho
        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1.0 - self.l1_ratio)

        gram = (X_centered.T @ X_centered) / n_samples
        Xy = (X_centered.T @ y_centered) / n_samples

        beta = np.zeros(n_features)
        z = np.zeros(n_features)
        u = np.zeros(n_features)

        system_matrix = gram + (rho + l2_penalty) * np.eye(n_features)

        for iteration in range(self.max_iter):
            beta = np.linalg.solve(system_matrix, Xy + rho * (z - u))

            z_old = z.copy()
            z = self._soft_threshold(beta + u, l1_penalty / rho)
            u = u + beta - z

            primal_residual = np.linalg.norm(beta - z)
            dual_residual = np.linalg.norm(rho * (z - z_old))
            if max(primal_residual, dual_residual) < self.tol:
                self.n_iter_ = iteration + 1
                return z

        self.n_iter_ = self.max_iter
        return z

    def _fit_ista(self, X_centered, y_centered):
        n_samples, n_features = X_centered.shape

        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1.0 - self.l1_ratio)

        gram = (X_centered.T @ X_centered) / n_samples
        Xy = (X_centered.T @ y_centered) / n_samples

        lipschitz = max(np.max(np.linalg.eigvalsh(gram)) + l2_penalty, 1e-12)
        if self.ista_step_size is None:
            step = 1.0 / lipschitz
        else:
            if self.ista_step_size <= 0:
                raise ValueError("ista_step_size must be positive when provided")
            step = self.ista_step_size

        beta = np.zeros(n_features)
        threshold = l1_penalty * step

        for iteration in range(self.max_iter):
            grad = gram @ beta - Xy + l2_penalty * beta
            beta_new = self._soft_threshold(beta - step * grad, threshold)

            if np.max(np.abs(beta_new - beta)) < self.tol:
                self.n_iter_ = iteration + 1
                return beta_new
            beta = beta_new

        self.n_iter_ = self.max_iter
        return beta

    def fit(self, X, y):
        """
        拟合弹性网络
        """
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not (0.0 <= self.l1_ratio <= 1.0):
            raise ValueError("l1_ratio must be in [0, 1]")
        if self.solver not in ('cd', 'admm', 'ista'):
            raise ValueError("solver must be 'cd', 'admm' or 'ista'")
        if self.solver == 'admm' and self.admm_rho <= 0:
            raise ValueError("admm_rho must be positive")

        n_samples, _ = X.shape

        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)

        X_centered = X - X_mean
        y_centered = y - y_mean

        if self.solver == 'cd':
            beta = self._fit_coordinate_descent(X_centered, y_centered)
        elif self.solver == 'admm':
            beta = self._fit_admm(X_centered, y_centered)
        else:
            beta = self._fit_ista(X_centered, y_centered)

        self.coef_ = beta
        self.intercept_ = y_mean - X_mean @ beta

        return self

    def predict(self, X):
        """
        预测
        """
        return X @ self.coef_ + self.intercept_
