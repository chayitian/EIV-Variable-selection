import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


class CLasso:
    """
    修正Lasso回归（处理测量误差的高维变量选择）

    目标函数：
    (1 / (2n)) * ||y - Wβ||^2 - (1/2)β^TΣ_{uu}β + λ||β||_1

    参数
    ----------
    alpha : float
        L1 正则化强度
    Sigma_uu : np.ndarray
        测量误差协方差矩阵
    max_iter : int
        最大迭代次数
    tol : float
        收敛阈值
    """

    def __init__(self, alpha=1.0, Sigma_uu=None, max_iter=10000, tol=1e-6):
        self.alpha = alpha
        self.Sigma_uu = Sigma_uu
        self.max_iter = max_iter
        self.tol = tol

        self.coef_ = None
        self.intercept_ = None
        self.scaler_W_ = None
        self.scaler_y_ = None
        self.Sigma_uu_scaled_ = None

    def fit(self, W, y):
        """
        拟合修正Lasso（使用CVXPY求解或坐标下降法）

        参数
        ----------
        W : np.ndarray
            含测量误差的可观测协变量矩阵 (n_samples, n_features)
        y : np.ndarray
            响应变量向量 (n_samples,)
        """
        n_samples, n_features = W.shape

        if self.Sigma_uu is None:
            Sigma_uu = np.zeros((n_features, n_features))
        else:
            Sigma_uu = self.Sigma_uu

        if Sigma_uu.shape != (n_features, n_features):
            raise ValueError(
                f"Sigma_uu shape must be ({n_features}, {n_features}), got {Sigma_uu.shape}"
            )

        self.scaler_W_ = StandardScaler()
        self.scaler_y_ = StandardScaler(with_std=False)

        W_scaled = self.scaler_W_.fit_transform(W)
        y_centered = self.scaler_y_.fit_transform(y.reshape(-1, 1)).flatten()

        W_std = self.scaler_W_.scale_
        W_std_safe = np.where(W_std > 1e-12, W_std, 1.0)
        Sigma_uu_scaled = Sigma_uu / np.outer(W_std_safe, W_std_safe)

        Sigma_W = (W_scaled.T @ W_scaled) / n_samples
        Sigma_corrected = Sigma_W - Sigma_uu_scaled

        min_eig = np.min(np.linalg.eigvalsh(Sigma_corrected))
        if min_eig < 1e-4:
            Sigma_corrected += np.eye(n_features) * (1e-4 - min_eig)

        try:
            L = np.linalg.cholesky(Sigma_corrected)
        except np.linalg.LinAlgError:
            Sigma_corrected = (Sigma_corrected + Sigma_corrected.T) / 2
            min_eig = np.min(np.linalg.eigvalsh(Sigma_corrected))
            if min_eig < 1e-4:
                Sigma_corrected += np.eye(n_features) * (1e-4 - min_eig)
            L = np.linalg.cholesky(Sigma_corrected)

        rho = (W_scaled.T @ y_centered) / n_samples

        W_transformed = L.T * np.sqrt(n_samples)
        y_transformed = np.linalg.solve(L, rho) * np.sqrt(n_samples)

        lasso = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=10000, tol=1e-6)
        lasso.fit(W_transformed, y_transformed)
        beta_scaled = lasso.coef_

        beta_original_scale = beta_scaled / W_std_safe
        beta_original_scale = np.where(W_std > 1e-12, beta_original_scale, 0.0)
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
