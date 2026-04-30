import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


class ACLasso:
    """
    自适应修正Lasso回归（ACLasso，结合CLasso和ALasso）

    参考文献：
    李锋, 盖宇杰, 卢一强. (2014). 测量误差模型的自适应LASSO变量选择方法研究.

    参数
    ----------
    alpha : float
        最终加权Lasso阶段的 L1 正则化强度
    Sigma_uu : np.ndarray
        测量误差协方差矩阵
    gamma : float
        自适应权重指数
    init_coef : np.ndarray or None
        外部传入的初始估计系数向量，长度应为 n_features。
        按 1/(|init_coef_scaled|+1e-6)^gamma 计算权重。
    max_iter : int
        最大迭代次数
    tol : float
        收敛阈值
    """

    def __init__(self, alpha=1.0, Sigma_uu=None, gamma=1.0,
                 max_iter=10000, tol=1e-6, init_coef=None):
        self.alpha = alpha
        self.Sigma_uu = Sigma_uu
        self.gamma = gamma
        self.init_coef = init_coef
        self.max_iter = max_iter
        self.tol = tol

        self.coef_ = None
        self.intercept_ = None
        self.beta_init_ = None
        self.weights_ = None

    def fit(self, W, y):
        """
        拟合ACLasso

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

        scaler_W = StandardScaler()
        scaler_y = StandardScaler(with_std=False)

        W_scaled = scaler_W.fit_transform(W)
        y_centered = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        W_std = scaler_W.scale_
        W_std_safe = np.where(W_std > 1e-12, W_std, 1.0)
        Sigma_uu_scaled = Sigma_uu / np.outer(W_std_safe, W_std_safe)

        if self.init_coef is None:
            raise ValueError("ACLasso requires externally provided init_coef.")
        self.beta_init_ = np.asarray(self.init_coef, dtype=float).reshape(-1)
        if self.beta_init_.size != n_features:
            raise ValueError(
                f"init_coef length must be {n_features}, got {self.beta_init_.size}"
            )
        if not np.all(np.isfinite(self.beta_init_)):
            raise ValueError("init_coef must contain only finite values")
        beta_init_scaled = self.beta_init_ * W_std
        weights = 1.0 / (np.abs(beta_init_scaled) + 1e-6) ** self.gamma

        self.weights_ = weights.copy()
        inv_weights = 1.0 / weights

        W_weighted = W_scaled * inv_weights[np.newaxis, :]
        Sigma_W_weighted = (W_weighted.T @ W_weighted) / n_samples
        Sigma_uu_weighted = np.diag(inv_weights) @ Sigma_uu_scaled @ np.diag(inv_weights)
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

        lasso = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter, tol=self.tol)
        lasso.fit(W_transformed, y_transformed)
        alpha_scaled = lasso.coef_

        beta_scaled = alpha_scaled * inv_weights
        beta_original_scale = beta_scaled / W_std_safe
        beta_original_scale = np.where(W_std > 1e-12, beta_original_scale, 0.0)
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
