import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge


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
    final_l1_alpha : float
        最终加权Lasso阶段的 L1 正则化参数 λ
    init_l1_alpha : float
        初始估计阶段（当 init_method='lasso'）的 L1 正则化参数
    init_l2_alpha : float
        初始估计阶段（当 init_method='ridge'）的 L2 正则化参数
    gamma : float
        权重指数参数，默认为1.0
    max_iter : int
        坐标下降法最大迭代次数
    tol : float
        收敛阈值
    init_method : str
        初始估计方法，'ols'、'ridge' 或 'lasso'，默认为 'ols'
    epsilon : float
        避免除零的小常数
    """

    def __init__(self, final_l1_alpha=1.0, init_l1_alpha=1.0, init_l2_alpha=1.0,
                 gamma=1.0, max_iter=1000, tol=1e-4, init_method='ols', epsilon=1e-6):
        self.final_l1_alpha = final_l1_alpha
        self.init_l1_alpha = init_l1_alpha
        self.init_l2_alpha = init_l2_alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        self.epsilon = epsilon

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

        if self.init_method == 'ols':
            init_model = LinearRegression(fit_intercept=True)
        elif self.init_method == 'ridge':
            init_model = Ridge(alpha=self.init_l2_alpha, fit_intercept=True)
        elif self.init_method == 'lasso':
            init_model = Lasso(alpha=self.init_l1_alpha, fit_intercept=True, max_iter=self.max_iter, tol=self.tol)
        else:
            raise ValueError(f"init_method must be 'ols', 'ridge' or 'lasso', got '{self.init_method}'")

        init_model.fit(X, y)
        self.init_coef_ = init_model.coef_.copy()

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
                beta_weighted[j] = self._soft_threshold(rho_j, self.final_l1_alpha) / z_j_safe

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
