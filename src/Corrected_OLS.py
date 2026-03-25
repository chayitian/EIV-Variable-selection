import numpy as np
from sklearn.preprocessing import StandardScaler


class CorrectedOLS:
    """
    修正OLS回归（处理测量误差的最小二乘估计）

    用于获得自适应修正Lasso和自适应CoCoLasso的初始参数估计值

    参数
    ----------
    Sigma_uu : np.ndarray
        测量误差协方差矩阵
    """

    def __init__(self, Sigma_uu=None):
        self.Sigma_uu = Sigma_uu

        self.coef_ = None
        self.intercept_ = None
        self.scaler_W_ = None
        self.scaler_y_ = None

    def fit(self, W, y):
        """
        拟合修正OLS

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

        Sigma_W = (W_scaled.T @ W_scaled) / n_samples
        Sigma_corrected = Sigma_W - Sigma_uu_scaled

        Sigma_corrected = (Sigma_corrected + Sigma_corrected.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(Sigma_corrected))
        if min_eig < 1e-4:
            Sigma_corrected += np.eye(n_features) * (1e-4 - min_eig)

        rho = (W_scaled.T @ y_centered) / n_samples

        beta_scaled = np.linalg.solve(Sigma_corrected, rho)

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
