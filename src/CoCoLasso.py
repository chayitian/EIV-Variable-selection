import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


class CoCoLasso:
    """
    CoCoLasso (Convex Constrained Lasso) - 凸约束Lasso

    参考文献：
    Datta, A., & Zou, H. (2017). CoCoLasso for High-dimensional Error-in-variables Regression.

    参数
    ----------
    alpha : float
        L1 正则化强度
    Sigma_uu : np.ndarray
        测量误差协方差矩阵
    max_iter_admm : int
        ADMM算法最大迭代次数
    tol_admm : float
        ADMM算法收敛阈值
    rho : float
        ADMM算法惩罚参数
    """

    def __init__(self, alpha=1.0, Sigma_uu=None, max_iter_admm=1000, tol_admm=1e-4, rho=1.0):
        self.alpha = alpha
        self.Sigma_uu = Sigma_uu
        self.max_iter_admm = max_iter_admm
        self.tol_admm = tol_admm
        self.rho = rho

        self.coef_ = None
        self.intercept_ = None
        self.scaler_W_ = None
        self.scaler_y_ = None

    def _project_psd(self, M):
        """
        将矩阵投影到最近半正定矩阵空间（ADMM算法）
        """
        ## ADMM 罚参数必须为正，避免 U / rho 数值异常
        if self.rho <= 0:
            raise ValueError("rho must be positive for ADMM updates")

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
        拟合CoCoLasso

        参数
        ----------
        W : np.ndarray
            含测量误差的可观测协变量矩阵 (n_samples, n_features)
        y : np.ndarray
            响应变量向量 (n_samples,)
        """
        n_samples, n_features = W.shape

        ## 避免在 fit 中改写实例配置，防止复用实例时状态污染
        if self.Sigma_uu is None:
            Sigma_uu = np.zeros((n_features, n_features))
        else:
            Sigma_uu = self.Sigma_uu

        ## 显式校验测量误差协方差维度，避免隐式广播错误
        if Sigma_uu.shape != (n_features, n_features):
            raise ValueError(
                f"Sigma_uu shape must be ({n_features}, {n_features}), got {Sigma_uu.shape}"
            )

        self.scaler_W_ = StandardScaler()
        self.scaler_y_ = StandardScaler(with_std=False)

        W_scaled = self.scaler_W_.fit_transform(W)
        y_centered = self.scaler_y_.fit_transform(y.reshape(-1, 1)).flatten()

        W_std = self.scaler_W_.scale_
        W_std_safe = np.where(W_std > 1e-12, W_std, 1.0) ## 防止零方差特征导致缩放除零
        Sigma_uu_scaled = Sigma_uu / np.outer(W_std_safe, W_std_safe)

        Sigma_hat = (W_scaled.T @ W_scaled) / n_samples - Sigma_uu_scaled
        rho_hat = (W_scaled.T @ y_centered) / n_samples

        Sigma_tilde = self._project_psd(Sigma_hat)

        Sigma_tilde = (Sigma_tilde + Sigma_tilde.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(Sigma_tilde))
        if min_eig < 1e-8:
            Sigma_tilde += np.eye(n_features) * (1e-8 - min_eig)
        
        L = np.linalg.cholesky(Sigma_tilde)

        W_tilde = L.T * np.sqrt(n_samples)
        y_tilde = np.linalg.solve(L, rho_hat) * np.sqrt(n_samples)

        lasso = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=10000, tol=1e-6)
        lasso.fit(W_tilde, y_tilde)
        beta_scaled = lasso.coef_

        beta_original_scale = beta_scaled / W_std_safe
        beta_original_scale = np.where(W_std > 1e-12, beta_original_scale, 0.0) ## 零方差特征不可识别，回写为 0 提升稳定性
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
