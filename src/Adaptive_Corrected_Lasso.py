import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from .Corrected_OLS import CorrectedOLS


class AdaptiveCorrectedLasso:
    """
    自适应修正Lasso回归（结合修正Lasso和自适应Lasso）

    参考文献：
    李锋, 盖宇杰, 卢一强. (2014). 测量误差模型的自适应LASSO变量选择方法研究.

    参数
    ----------
    alpha : float
        L1 正则化强度
    Sigma_uu : np.ndarray
        测量误差协方差矩阵
    gamma : float
        自适应权重指数
    max_iter : int
        最大迭代次数
    tol : float
        收敛阈值
    """

    def __init__(self, alpha=1.0, Sigma_uu=None, gamma=1.0, max_iter=10000, tol=1e-6):
        self.alpha = alpha
        self.Sigma_uu = Sigma_uu
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

        self.coef_ = None
        self.intercept_ = None
        self.beta_init_ = None
        self.weights_ = None

    def fit(self, W, y):
        """
        拟合自适应修正Lasso

        参数
        ----------
        W : np.ndarray
            含测量误差的可观测协变量矩阵 (n_samples, n_features)
        y : np.ndarray
            响应变量向量 (n_samples,)
        """
        n_samples, n_features = W.shape

        if self.Sigma_uu is None:
            self.Sigma_uu = np.zeros((n_features, n_features))

        scaler_W = StandardScaler()
        scaler_y = StandardScaler(with_std=False)

        W_scaled = scaler_W.fit_transform(W)
        y_centered = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        W_std = scaler_W.scale_
        Sigma_uu_scaled = self.Sigma_uu / np.outer(W_std, W_std)

        co = CorrectedOLS(Sigma_uu=self.Sigma_uu)
        co.fit(W, y)
        self.beta_init_ = co.coef_.copy()

        beta_init_scaled = self.beta_init_ * W_std
        weights = 1.0 / (np.abs(beta_init_scaled) + 1e-6) ** self.gamma
        self.weights_ = weights.copy()

        W_weighted = W_scaled / weights[np.newaxis, :]
        Sigma_W_weighted = (W_weighted.T @ W_weighted) / n_samples
        Sigma_uu_weighted = np.diag(weights) @ Sigma_uu_scaled @ np.diag(weights)
        Sigma_corrected = Sigma_W_weighted - Sigma_uu_weighted

        Sigma_corrected = (Sigma_corrected + Sigma_corrected.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(Sigma_corrected))
        if min_eig < 1e-3:
            Sigma_corrected += np.eye(n_features) * (1e-3 - min_eig)

        try:
            L = np.linalg.cholesky(Sigma_corrected)
        except np.linalg.LinAlgError:
            Sigma_corrected += np.eye(n_features) * 1e-2
            L = np.linalg.cholesky(Sigma_corrected)

        rho = (W_weighted.T @ y_centered) / n_samples
        W_transformed = L.T * np.sqrt(n_samples)
        y_transformed = np.linalg.solve(L, rho) * np.sqrt(n_samples)

        lasso = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=10000, tol=1e-6)
        lasso.fit(W_transformed, y_transformed)
        alpha_scaled = lasso.coef_

        beta_scaled = alpha_scaled * weights
        beta_original_scale = beta_scaled / W_std
        self.coef_ = beta_original_scale
        self.intercept_ = scaler_y.mean_ - np.dot(scaler_W.mean_, beta_original_scale)

        return self

    def predict(self, W):
        """
        预测

        参数
        ----------
        W : np.ndarray
            含测量误差的可观测协变量矩阵 (n_samples, n_features)

        返回
        -------
        y_pred : np.ndarray
            预测的响应变量 (n_samples,)
        """
        return W @ self.coef_ + self.intercept_
