import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet as SklearnElasticNet


class CoCoElasticNet:
    """
    CoCoElasticNet (Convex Constrained Elastic Net)

    在 CoCoLasso 的误差修正框架下，将内层稀疏回归器替换为 ElasticNet。

    参数
    ----------
    alpha : float
        ElasticNet 正则化强度
    l1_ratio : float
        ElasticNet 的 L1 比例，取值范围 [0, 1]
    Sigma_uu : np.ndarray or None
        测量误差协方差矩阵
    max_iter_admm : int
        PSD 投影 ADMM 的最大迭代次数
    tol_admm : float
        PSD 投影 ADMM 的收敛阈值
    rho : float
        PSD 投影 ADMM 的惩罚参数
    enet_solver : str
        内层 ElasticNet 求解器。当前为避免中心化逻辑冲突，仅支持 'cd'
    enet_max_iter : int
        内层 ElasticNet 最大迭代次数
    enet_tol : float
        内层 ElasticNet 收敛阈值
    enet_admm_rho : float
        兼容参数，当前实现中未使用
    enet_ista_step_size : float or None
        兼容参数，当前实现中未使用
    """

    def __init__(
        self,
        alpha=1.0,
        l1_ratio=0.5,
        Sigma_uu=None,
        max_iter_admm=1000,
        tol_admm=1e-4,
        rho=1.0,
        enet_solver='cd',
        enet_max_iter=1000,
        enet_tol=1e-4,
        enet_admm_rho=1.0,
        enet_ista_step_size=None,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.Sigma_uu = Sigma_uu
        self.max_iter_admm = max_iter_admm
        self.tol_admm = tol_admm
        self.rho = rho

        self.enet_solver = enet_solver
        self.enet_max_iter = enet_max_iter
        self.enet_tol = enet_tol
        self.enet_admm_rho = enet_admm_rho
        self.enet_ista_step_size = enet_ista_step_size

        self.coef_ = None
        self.intercept_ = None
        self.scaler_W_ = None
        self.scaler_y_ = None

        self.fit_time_ = None
        self.admm_n_iter_ = None
        self.admm_converged_ = None
        self.admm_step_norm_history_ = None
        self.psd_min_eig_before_ = None
        self.psd_jitter_added_ = None
        self.cholesky_retried_ = None
        self.cholesky_jitter_added_ = None
        self.enet_n_iter_ = None

    def _project_psd(self, M):
        """
        将矩阵投影到最近半正定矩阵空间（ADMM）
        """
        if self.rho <= 0:
            raise ValueError("rho must be positive for ADMM updates")

        X = np.copy(M)
        Z = np.copy(M)
        U = np.zeros_like(M)

        step_norm_history = []
        admm_converged = False
        n_iter = 0

        for iter_idx in range(1, self.max_iter_admm + 1):
            X_old = X.copy()

            X = (M + self.rho * Z - U) / (1.0 + self.rho)

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
        拟合 CoCoElasticNet
        """
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not (0.0 <= self.l1_ratio <= 1.0):
            raise ValueError("l1_ratio must be in [0, 1]")
        if self.enet_solver != 'cd':
            raise ValueError("For CoCoElasticNet, enet_solver currently supports only 'cd'")

        fit_start = time.perf_counter()

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

        Sigma_hat = (W_scaled.T @ W_scaled) / n_samples - Sigma_uu_scaled
        rho_hat = (W_scaled.T @ y_centered) / n_samples

        Sigma_tilde = self._project_psd(Sigma_hat)
        Sigma_tilde = (Sigma_tilde + Sigma_tilde.T) / 2.0

        min_eig = np.min(np.linalg.eigvalsh(Sigma_tilde))
        self.psd_min_eig_before_ = float(min_eig)

        psd_jitter = max(0.0, float(1e-8 - min_eig))
        if psd_jitter > 0:
            Sigma_tilde += np.eye(n_features) * psd_jitter
        self.psd_jitter_added_ = psd_jitter

        self.cholesky_retried_ = False
        self.cholesky_jitter_added_ = 0.0
        try:
            L = np.linalg.cholesky(Sigma_tilde)
        except np.linalg.LinAlgError:
            self.cholesky_retried_ = True
            self.cholesky_jitter_added_ = 1e-10
            Sigma_tilde = Sigma_tilde + np.eye(n_features) * self.cholesky_jitter_added_
            L = np.linalg.cholesky(Sigma_tilde)

        W_tilde = L.T * np.sqrt(n_samples)
        y_tilde = np.linalg.solve(L, rho_hat) * np.sqrt(n_samples)

        enet = SklearnElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=False,
            max_iter=self.enet_max_iter,
            tol=self.enet_tol,
        )
        enet.fit(W_tilde, y_tilde)
        beta_scaled = enet.coef_
        self.enet_n_iter_ = int(getattr(enet, 'n_iter_', 0))

        beta_original_scale = beta_scaled / W_std_safe
        beta_original_scale = np.where(W_std > 1e-12, beta_original_scale, 0.0)

        self.coef_ = beta_original_scale
        self.intercept_ = self.scaler_y_.mean_ - np.dot(self.scaler_W_.mean_, beta_original_scale)
        self.fit_time_ = float(time.perf_counter() - fit_start)

        return self

    def predict(self, W):
        """
        预测
        """
        return W @ self.coef_ + self.intercept_
