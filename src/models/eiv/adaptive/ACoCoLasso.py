import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


class ACoCoLasso:
    """
    自适应CoCoLasso（ACoCoLasso，结合CoCoLasso和ALasso）

    本项目创新提出的方法，将CoCoLasso的凸优化框架与自适应Lasso惩罚项结合，
    在高维测量误差回归场景下实现神谕性质。

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
        按 1/(|init_coef_scaled|+1e-8)^gamma 计算权重。
    max_iter_admm : int
        ADMM算法最大迭代次数
    tol_admm : float
        ADMM算法收敛阈值
    rho : float
        ADMM算法惩罚参数
    """

    def __init__(self, alpha=1.0, Sigma_uu=None, gamma=1.0,
                 max_iter_admm=1000, tol_admm=1e-4, rho=1.0, init_coef=None):
        self.alpha = alpha
        self.Sigma_uu = Sigma_uu
        self.gamma = gamma
        self.init_coef = init_coef
        self.max_iter_admm = max_iter_admm
        self.tol_admm = tol_admm
        self.rho = rho

        self.coef_ = None
        self.intercept_ = None
        self.beta_init_ = None
        self.weights_ = None

        ### 新增诊断字段，不改动原有输出协议，用于实验报告中的可解释性与稳定性分析
        self.fit_time_ = None
        self.admm_n_iter_ = None
        self.admm_converged_ = None
        self.admm_step_norm_history_ = None
        self.psd_min_eig_before_ = None
        self.psd_jitter_added_ = None
        self.cholesky_retried_ = None
        self.cholesky_jitter_added_ = None
        self.init_method_used_ = None

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

        ### 保留每轮迭代步长，便于后续比较 ADMM 参数 rho/tol 的影响
        step_norm_history = []
        admm_converged = False
        n_iter = 0

        for iter_idx in range(1, self.max_iter_admm + 1):
            X_old = X.copy()

            X = (M + self.rho * Z - U) / (1 + self.rho)

            eigvals, eigvecs = np.linalg.eigh(X + U / self.rho)
            eigvals[eigvals < 0] = 0
            Z = eigvecs @ np.diag(eigvals) @ eigvecs.T

            U = U + self.rho * (X - Z)

            step_norm = float(np.linalg.norm(X - X_old))
            step_norm_history.append(step_norm)
            n_iter = iter_idx

            if step_norm < self.tol_admm:
                admm_converged = True
                break

        self.admm_n_iter_ = n_iter
        self.admm_converged_ = admm_converged
        self.admm_step_norm_history_ = step_norm_history

        return Z

    def fit(self, W, y):
        """
        拟合ACoCoLasso

        参数
        ----------
        W : np.ndarray
            含测量误差的可观测协变量矩阵 (n_samples, n_features)
        y : np.ndarray
            响应变量向量 (n_samples,)
        """
        ### 记录模型总耗时，便于与 CoCoLasso/树加权方法做时间开销对比
        fit_start = time.perf_counter()

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

        if self.init_coef is None:
            raise ValueError("ACoCoLasso requires externally provided init_coef.")
        self.beta_init_ = np.asarray(self.init_coef, dtype=float).reshape(-1)
        if self.beta_init_.size != n_features:
            raise ValueError(
                f"init_coef length must be {n_features}, got {self.beta_init_.size}"
            )
        if not np.all(np.isfinite(self.beta_init_)):
            raise ValueError("init_coef must contain only finite values")
        self.init_method_used_ = 'external_init_coef'
        beta_init_scaled = self.beta_init_ * W_std
        weights = 1.0 / (np.abs(beta_init_scaled) + 1e-8) ** self.gamma

        self.weights_ = weights.copy()

        inv_weights = 1.0 / weights
        W_inv_diag = np.diag(inv_weights)

        Sigma_hat = (W_scaled.T @ W_scaled) / n_samples - Sigma_uu_scaled
        rho_hat = (W_scaled.T @ y_centered) / n_samples

        Sigma_tilde = self._project_psd(Sigma_hat)

        Sigma_transformed = W_inv_diag.T @ Sigma_tilde @ W_inv_diag
        rho_transformed = W_inv_diag.T @ rho_hat

        Sigma_transformed = (Sigma_transformed + Sigma_transformed.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(Sigma_transformed))
        self.psd_min_eig_before_ = float(min_eig)
        ### 记录 PSD 抬升幅度，便于诊断不同数据设置下的矩阵病态程度
        psd_jitter = max(0.0, float(1e-8 - min_eig))
        if psd_jitter > 0:
            Sigma_transformed += np.eye(n_features) * psd_jitter
        self.psd_jitter_added_ = psd_jitter

        self.cholesky_retried_ = False
        self.cholesky_jitter_added_ = 0.0
        try:
            L = np.linalg.cholesky(Sigma_transformed)
        except np.linalg.LinAlgError:
            ### 分解失败时做微扰重试，减少因极端样本导致的偶发失败
            self.cholesky_retried_ = True
            self.cholesky_jitter_added_ = 1e-10
            Sigma_transformed = Sigma_transformed + np.eye(n_features) * self.cholesky_jitter_added_
            L = np.linalg.cholesky(Sigma_transformed)

        W_tilde = L.T * np.sqrt(n_samples)
        y_tilde = np.linalg.solve(L, rho_transformed) * np.sqrt(n_samples)

        lasso = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=10000, tol=1e-6)
        lasso.fit(W_tilde, y_tilde)
        alpha_scaled = lasso.coef_

        beta_scaled = W_inv_diag @ alpha_scaled
        beta_original_scale = beta_scaled / W_std_safe
        beta_original_scale = np.where(W_std > 1e-12, beta_original_scale, 0.0) ## 零方差特征不可识别，回写为 0 提升稳定性
        self.coef_ = beta_original_scale
        self.intercept_ = self.scaler_y_.mean_ - np.dot(self.scaler_W_.mean_, beta_original_scale)

        self.fit_time_ = float(time.perf_counter() - fit_start)

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
