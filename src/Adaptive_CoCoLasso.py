import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from .Corrected_OLS import CorrectedOLS


class AdaptiveCoCoLasso:
    """
    自适应CoCoLasso（创新模型 - 结合CoCoLasso和自适应Lasso）

    本项目创新提出的方法，将CoCoLasso的凸优化框架与自适应Lasso惩罚项结合，
    在高维测量误差回归场景下实现神谕性质。

    参数
    ----------
    alpha : float
        L1 正则化强度
    Sigma_uu : np.ndarray
        测量误差协方差矩阵
    gamma : float
        自适应权重指数
    max_iter_admm : int
        ADMM算法最大迭代次数
    tol_admm : float
        ADMM算法收敛阈值
    rho : float
        ADMM算法惩罚参数
    """

    def __init__(self, alpha=1.0, Sigma_uu=None, gamma=1.0, max_iter_admm=1000, tol_admm=1e-4, rho=1.0):
        self.alpha = alpha
        self.Sigma_uu = Sigma_uu
        self.gamma = gamma
        self.max_iter_admm = max_iter_admm
        self.tol_admm = tol_admm
        self.rho = rho

        self.coef_ = None
        self.intercept_ = None
        self.beta_init_ = None
        self.weights_ = None

    def _project_psd(self, M):
        """
        将矩阵投影到最近半正定矩阵空间（ADMM算法）
        """
        p = M.shape[0]
        X = np.copy(M)
        Z = np.copy(M)
        U = np.zeros_like(M)

        for _ in range(self.max_iter_admm):
            X_old = X.copy()

            X = (M + self.rho * Z - U) / (1 + self.rho)

            eigvals, eigvecs = np.linalg.eigh(X + U / self.rho)
            eigvals[eigvals < 0] = 0
            Z = eigvecs @ np.diag(eigvals) @ eigvecs.T

            U = U + self.rho * (X - Z)

            if np.linalg.norm(X - X_old) < self.tol_admm:
                break

        return Z

    def fit(self, W, y):
        """
        拟合自适应CoCoLasso

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

        self.scaler_W_ = StandardScaler()
        self.scaler_y_ = StandardScaler(with_std=False)

        W_scaled = self.scaler_W_.fit_transform(W)
        y_centered = self.scaler_y_.fit_transform(y.reshape(-1, 1)).flatten()

        W_std = self.scaler_W_.scale_
        Sigma_uu_scaled = self.Sigma_uu / np.outer(W_std, W_std)

        co = CorrectedOLS(Sigma_uu=self.Sigma_uu)
        co.fit(W, y)
        self.beta_init_ = co.coef_.copy()

        beta_init_scaled = self.beta_init_ * W_std
        weights = 1.0 / (np.abs(beta_init_scaled) + 1e-8) ** self.gamma
        self.weights_ = weights.copy()

        W_diag = np.diag(weights)

        Sigma_hat = (W_scaled.T @ W_scaled) / n_samples - Sigma_uu_scaled
        rho_hat = (W_scaled.T @ y_centered) / n_samples

        Sigma_tilde = self._project_psd(Sigma_hat)

        Sigma_transformed = W_diag.T @ Sigma_tilde @ W_diag
        rho_transformed = W_diag.T @ rho_hat

        Sigma_transformed = (Sigma_transformed + Sigma_transformed.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(Sigma_transformed))
        if min_eig < 1e-8:
            Sigma_transformed += np.eye(n_features) * (1e-8 - min_eig)
        
        L = np.linalg.cholesky(Sigma_transformed)

        W_tilde = L.T * np.sqrt(n_samples)
        y_tilde = np.linalg.solve(L, rho_transformed) * np.sqrt(n_samples)

        lasso = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=10000, tol=1e-6)
        lasso.fit(W_tilde, y_tilde)
        alpha_scaled = lasso.coef_

        beta_scaled = W_diag @ alpha_scaled
        beta_original_scale = beta_scaled / W_std
        self.coef_ = beta_original_scale
        self.intercept_ = self.scaler_y_.mean_ - np.dot(self.scaler_W_.mean_, beta_original_scale)

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
