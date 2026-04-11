import numpy as np
from sklearn.preprocessing import StandardScaler


class COLS:
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

        Sigma_W = (W_scaled.T @ W_scaled) / n_samples
        Sigma_corrected = Sigma_W - Sigma_uu_scaled

        Sigma_corrected = (Sigma_corrected + Sigma_corrected.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(Sigma_corrected))
        if min_eig < 1e-4:
            Sigma_corrected += np.eye(n_features) * (1e-4 - min_eig)

        rho = (W_scaled.T @ y_centered) / n_samples

        beta_scaled = np.linalg.solve(Sigma_corrected, rho)

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
