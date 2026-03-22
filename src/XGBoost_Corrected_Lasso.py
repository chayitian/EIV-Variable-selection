import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import xgboost as xgb
from .Corrected_Lasso import CorrectedLasso


class XGBoostCorrectedLasso:
    """
    XGBoost修正Lasso回归（结合修正Lasso和XGBoost特征重要性）

    使用XGBoost的特征重要性作为自适应权重，替换传统的基于系数估计的权重。

    参数
    ----------
    alpha : float
        L1 正则化强度
    Sigma_uu : np.ndarray
        测量误差协方差矩阵
    n_estimators : int
        XGBoost的树数量
    max_depth : int
        XGBoost的最大深度
    learning_rate : float
        XGBoost的学习率
    gamma : float
        权重指数
    max_iter : int
        最大迭代次数
    tol : float
        收敛阈值
    """

    def __init__(self, alpha=1.0, Sigma_uu=None, n_estimators=100, max_depth=6, 
                 learning_rate=0.1, gamma=1.0, max_iter=10000, tol=1e-6):
        self.alpha = alpha
        self.Sigma_uu = Sigma_uu
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

        self.coef_ = None
        self.intercept_ = None
        self.feature_importances_ = None
        self.weights_ = None
        self.xgb_model_ = None

    def fit(self, W, y):
        """
        拟合XGBoost修正Lasso

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

        # 使用XGBoost获取特征重要性
        xgb_model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb_model.fit(W_scaled, y_centered)
        self.xgb_model_ = xgb_model
        feature_importances = xgb_model.feature_importances_
        self.feature_importances_ = feature_importances.copy()

        # 基于特征重要性计算权重
        # 特征越重要，权重越小（惩罚越小）
        max_importance = np.max(feature_importances) if np.max(feature_importances) > 0 else 1.0
        weights = 1.0 / (feature_importances / max_importance + 1e-8) ** self.gamma
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
