import numpy as np
from sklearn.preprocessing import StandardScaler


class CorrectedRidge:
    """
    修正Ridge回归（处理测量误差的L2正则回归）

    目标函数：
    (1 / (2n)) * ||y - Wβ||^2 - (1/2)β^TΣ_{uu}β + (alpha / 2)||β||_2^2

    通过标准化后等价的二次型闭式解求解：
    beta = (Sigma_corrected + alpha * I)^(-1) rho

    参数
    ----------
    alpha : float
        L2 正则化强度
    Sigma_uu : np.ndarray
        测量误差协方差矩阵
    min_eig : float
        数值稳定化时的最小特征值下限
    """

    def __init__(self, alpha=1.0, Sigma_uu=None, min_eig=1e-4):
        self.alpha = alpha
        self.Sigma_uu = Sigma_uu
        self.min_eig = min_eig

        self.coef_ = None
        self.intercept_ = None
        self.scaler_W_ = None
        self.scaler_y_ = None

    def fit(self, W, y):
        """
        拟合修正Ridge

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

        current_min_eig = np.min(np.linalg.eigvalsh(Sigma_corrected))
        if current_min_eig < self.min_eig:
            Sigma_corrected += np.eye(n_features) * (self.min_eig - current_min_eig)

        rho = (W_scaled.T @ y_centered) / n_samples

        ridge_system = Sigma_corrected + self.alpha * np.eye(n_features)
        beta_scaled = np.linalg.solve(ridge_system, rho)

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
