import numpy as np
from sklearn.linear_model import LassoLars


class ALasso:
    """
    自适应Lasso回归（Adaptive Lasso）

    目标函数：
    (1 / (2n)) * ||y - Xβ||^2 + λ * Σ w_j|β_j|

    其中自适应权重 w_j = 1 / |β̂_j^init|^γ

    参考文献：
    Zou, H. (2006). The Adaptive Lasso and Its Oracle Properties.
    Journal of the American Statistical Association, 101(476), 1418-1429.

    参数
    ----------
    alpha : float
        最终加权Lasso阶段的 L1 正则化参数 λ
    gamma : float
        权重指数参数，默认为1.0
    max_iter : int
        坐标下降法最大迭代次数
    tol : float
        收敛阈值
    solver : str
        求解方法，可选 'cd'（坐标下降）、'admm'、'ista' 或 'lars'，默认 'cd'
    admm_rho : float
        ADMM 增广拉格朗日参数（solver='admm' 时生效）
    ista_step_size : float or None
        ISTA 步长；None 表示按 Lipschitz 常数自动设置
    init_coef : np.ndarray or None
        外部传入的初始估计系数向量，长度应为 n_features。
        按 1/(|init_coef|+epsilon)^gamma 计算权重。
    epsilon : float
        避免除零的小常数
    """

    def __init__(self, alpha=1.0, gamma=1.0, max_iter=1000, tol=1e-4, epsilon=1e-6,
                 init_coef=None, solver='cd', admm_rho=1.0, ista_step_size=None):
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

    def _soft_threshold(self, rho, lam):
        """
        软阈值函数
        S(rho, lam) = sign(rho) * max(|rho| - lam, 0)
        """
        return np.sign(rho) * np.maximum(np.abs(rho) - lam, 0.0)

    def _fit_coordinate_descent(self, X_centered, y_centered, weights):
        n_samples, n_features = X_centered.shape

        X_weighted = X_centered / weights
        beta_weighted = np.zeros(n_features)

        for iteration in range(self.max_iter):
            beta_old = beta_weighted.copy()

            for j in range(n_features):
                residual = y_centered - X_weighted @ beta_weighted + beta_weighted[j] * X_weighted[:, j]
                rho_j = np.dot(X_weighted[:, j], residual) / n_samples
                z_j = np.sum(X_weighted[:, j] ** 2) / n_samples

                z_j_safe = max(z_j, 1e-12)  # 防止加权后近零方差列导致除零
                beta_weighted[j] = self._soft_threshold(rho_j, self.alpha) / z_j_safe

            if np.max(np.abs(beta_weighted - beta_old)) < self.tol:
                self.n_iter_ = iteration + 1
                return beta_weighted / weights

        self.n_iter_ = self.max_iter
        return beta_weighted / weights

    def _fit_admm(self, X_centered, y_centered, weights):
        n_samples, n_features = X_centered.shape

        rho = self.admm_rho
        gram = (X_centered.T @ X_centered) / n_samples
        Xy = (X_centered.T @ y_centered) / n_samples

        beta = np.zeros(n_features)
        z = np.zeros(n_features)
        u = np.zeros(n_features)

        system_matrix = gram + rho * np.eye(n_features)
        thresholds = self.alpha * weights / rho

        for iteration in range(self.max_iter):
            beta = np.linalg.solve(system_matrix, Xy + rho * (z - u))

            z_old = z.copy()
            z = self._soft_threshold(beta + u, thresholds)
            u = u + beta - z

            primal_residual = np.linalg.norm(beta - z)
            dual_residual = np.linalg.norm(rho * (z - z_old))
            if max(primal_residual, dual_residual) < self.tol:
                self.n_iter_ = iteration + 1
                return z

        self.n_iter_ = self.max_iter
        return z

    def _fit_ista(self, X_centered, y_centered, weights):
        n_samples, n_features = X_centered.shape

        gram = (X_centered.T @ X_centered) / n_samples
        Xy = (X_centered.T @ y_centered) / n_samples

        lipschitz = max(np.max(np.linalg.eigvalsh(gram)), 1e-12)
        if self.ista_step_size is None:
            step = 1.0 / lipschitz
        else:
            if self.ista_step_size <= 0:
                raise ValueError("ista_step_size must be positive when provided")
            step = self.ista_step_size

        beta = np.zeros(n_features)
        thresholds = self.alpha * weights * step

        for iteration in range(self.max_iter):
            grad = gram @ beta - Xy
            beta_new = self._soft_threshold(beta - step * grad, thresholds)

            if np.max(np.abs(beta_new - beta)) < self.tol:
                self.n_iter_ = iteration + 1
                return beta_new
            beta = beta_new

        self.n_iter_ = self.max_iter
        return beta

    def _fit_lars(self, X_centered, y_centered, weights):
        X_weighted = X_centered / weights
        model = LassoLars(alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter)
        model.fit(X_weighted, y_centered)
        self.n_iter_ = int(getattr(model, 'n_iter_', self.max_iter))
        return np.asarray(model.coef_, dtype=float) / weights

    def fit(self, X, y):
        """
        拟合自适应Lasso

        参数
        ----------
        X : np.ndarray
            协变量矩阵 (n_samples, n_features)
        y : np.ndarray
            响应变量向量 (n_samples,)
        """
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.solver not in ('cd', 'admm', 'ista', 'lars'):
            raise ValueError("solver must be 'cd', 'admm', 'ista' or 'lars'")
        if self.solver == 'admm' and self.admm_rho <= 0:
            raise ValueError("admm_rho must be positive")

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

        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)

        X_centered = X - X_mean
        y_centered = y - y_mean

        if self.solver == 'cd':
            self.coef_ = self._fit_coordinate_descent(X_centered, y_centered, self.weights_)
        elif self.solver == 'admm':
            self.coef_ = self._fit_admm(X_centered, y_centered, self.weights_)
        elif self.solver == 'ista':
            self.coef_ = self._fit_ista(X_centered, y_centered, self.weights_)
        else:
            self.coef_ = self._fit_lars(X_centered, y_centered, self.weights_)

        self.intercept_ = y_mean - np.dot(X_mean, self.coef_)

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
