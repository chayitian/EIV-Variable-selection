import numpy as np


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
    init_coef : np.ndarray or None
        外部传入的初始估计系数向量，长度应为 n_features。
        按 1/(|init_coef|+epsilon)^gamma 计算权重。
    epsilon : float
        避免除零的小常数
    """

    def __init__(self, alpha=1.0, gamma=1.0, max_iter=1000, tol=1e-4, epsilon=1e-6,
                 init_coef=None):
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = epsilon
        self.init_coef = init_coef

        self.coef_ = None
        self.intercept_ = None
        self.weights_ = None
        self.init_coef_ = None

    def _soft_threshold(self, rho, lam):
        """
        软阈值函数
        S(rho, lam) = sign(rho) * max(|rho| - lam, 0)
        """
        if rho > lam:
            return rho - lam
        elif rho < -lam:
            return rho + lam
        else:
            return 0.0

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

        X_weighted = X_centered / self.weights_

        beta_weighted = np.zeros(n_features)

        for iteration in range(self.max_iter):
            beta_old = beta_weighted.copy()

            for j in range(n_features):
                residual = y_centered - X_weighted @ beta_weighted + beta_weighted[j] * X_weighted[:, j]

                rho_j = np.dot(X_weighted[:, j], residual) / n_samples

                z_j = np.sum(X_weighted[:, j] ** 2) / n_samples

                z_j_safe = max(z_j, 1e-12) ## 防止加权后近零方差列导致除零
                beta_weighted[j] = self._soft_threshold(rho_j, self.alpha) / z_j_safe

            if np.max(np.abs(beta_weighted - beta_old)) < self.tol:
                break

        self.coef_ = beta_weighted / self.weights_

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
