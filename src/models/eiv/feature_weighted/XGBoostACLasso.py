import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import xgboost as xgb


class XGBoostACLasso:
    """
    XGBoost自适应修正Lasso回归（结合自适应修正Lasso和XGBoost特征重要性）

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
    importance_type : str
        特征重要性类型: 'gain', 'weight', 'cover'
    subsample : float
        每棵树的样本采样比例
    colsample_bytree : float
        每棵树的特征采样比例
    min_child_weight : float
        子节点最小样本权重和
    xgb_gamma : float
        节点分裂所需的最小损失下降（XGBoost 原生 gamma）
    reg_alpha : float
        L1 正则项系数
    reg_lambda : float
        L2 正则项系数
    objective : str
        学习目标函数
    random_state : int or None
        随机种子
    n_jobs : int or None
        并行线程数
    verbosity : int
        日志级别
    gamma : float
        权重指数
    weight_method : str
        权重计算方法: 'normalized'（默认，归一化后计算）或 'max_scaled'（最大值缩放后计算）
    max_iter : int
        最大迭代次数
    tol : float
        收敛阈值
    """

    def __init__(self, alpha=1.0, Sigma_uu=None, n_estimators=100, max_depth=10,
                 learning_rate=0.1, importance_type='gain',
                 subsample=0.8, colsample_bytree=0.8, min_child_weight=1.0,
                 xgb_gamma=0.0, reg_alpha=0.0, reg_lambda=1.0,
                 objective='reg:squarederror', random_state=42, n_jobs=1, verbosity=0,
                 gamma=1.0, weight_method='normalized', max_iter=10000, tol=1e-6):
        self.alpha = alpha
        self.Sigma_uu = Sigma_uu
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.importance_type = importance_type
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.xgb_gamma = xgb_gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.gamma = gamma
        self.weight_method = weight_method
        self.max_iter = max_iter
        self.tol = tol

        self.coef_ = None
        self.intercept_ = None
        self.feature_importances_ = None
        self.weights_ = None
        self.xgb_model_ = None

        ### 新增诊断字段，不改变原有 fit/predict 接口，用于时间与数值稳定性分析
        self.fit_time_ = None
        self.psd_min_eig_before_ = None
        self.psd_jitter_added_ = None
        self.cholesky_retried_ = None
        self.cholesky_jitter_added_ = None
        self.weight_method_used_ = None

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
        ### 记录模型总耗时，便于与 CoCoLasso/ACoCoLasso 做计算代价对比
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

        scaler_W = StandardScaler()
        scaler_y = StandardScaler(with_std=False)

        W_scaled = scaler_W.fit_transform(W)
        y_centered = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        W_std = scaler_W.scale_
        W_std_safe = np.where(W_std > 1e-12, W_std, 1.0) ## 防止零方差特征导致缩放除零
        Sigma_uu_scaled = Sigma_uu / np.outer(W_std_safe, W_std_safe)

        # 使用XGBoost获取特征重要性
        xgb_model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            importance_type=self.importance_type,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.xgb_gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective=self.objective,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity
        )
        xgb_model.fit(W_scaled, y_centered)
        self.xgb_model_ = xgb_model
        feature_importances = xgb_model.feature_importances_
        self.feature_importances_ = feature_importances.copy()

        # 基于特征重要性计算权重
        # 特征越重要，权重越小（惩罚越小）
        if self.weight_method == 'normalized':
            # 方法1：归一化后计算（默认，更稳定）
            fi_normalized = feature_importances / (np.sum(feature_importances) + 1e-10)
            fi_normalized = np.clip(fi_normalized, 1e-10, 1.0)
            weights = 1.0 / (fi_normalized + 1e-8) ** self.gamma
            weights = np.clip(weights, 1e-3, 1e6)
        elif self.weight_method == 'max_scaled':
            # 方法2：最大值缩放后计算（备选）
            max_importance = np.max(feature_importances) if np.max(feature_importances) > 0 else 1.0
            weights = 1.0 / (feature_importances / max_importance + 1e-8) ** self.gamma
        else:
            raise ValueError(f"未知的权重计算方法: {self.weight_method}，请使用 'normalized' 或 'max_scaled'")
        self.weight_method_used_ = self.weight_method
        self.weights_ = weights.copy()

        W_weighted = W_scaled / weights[np.newaxis, :]
        Sigma_W_weighted = (W_weighted.T @ W_weighted) / n_samples
        Sigma_uu_weighted = np.diag(weights) @ Sigma_uu_scaled @ np.diag(weights)
        Sigma_corrected = Sigma_W_weighted - Sigma_uu_weighted

        Sigma_corrected = (Sigma_corrected + Sigma_corrected.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(Sigma_corrected))
        self.psd_min_eig_before_ = float(min_eig)
        ### 显式记录 PSD 抬升幅度，便于定位病态矩阵导致的不稳定
        psd_jitter = max(0.0, float(1e-3 - min_eig))
        if psd_jitter > 0:
            Sigma_corrected += np.eye(n_features) * psd_jitter
        self.psd_jitter_added_ = psd_jitter

        self.cholesky_retried_ = False
        self.cholesky_jitter_added_ = 0.0
        try:
            L = np.linalg.cholesky(Sigma_corrected)
        except np.linalg.LinAlgError:
            ### 首次分解失败时做微扰重试，减少偶发数值失败
            self.cholesky_retried_ = True
            self.cholesky_jitter_added_ = 1e-2
            Sigma_corrected += np.eye(n_features) * self.cholesky_jitter_added_
            L = np.linalg.cholesky(Sigma_corrected)

        rho = (W_weighted.T @ y_centered) / n_samples
        W_transformed = L.T * np.sqrt(n_samples)
        y_transformed = np.linalg.solve(L, rho) * np.sqrt(n_samples)

        lasso = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=10000, tol=1e-6)
        lasso.fit(W_transformed, y_transformed)
        alpha_scaled = lasso.coef_

        beta_scaled = alpha_scaled * weights
        beta_original_scale = beta_scaled / W_std_safe
        beta_original_scale = np.where(W_std > 1e-12, beta_original_scale, 0.0) ## 零方差特征不可识别，回写为 0 提升稳定性
        self.coef_ = beta_original_scale
        self.intercept_ = scaler_y.mean_ - np.dot(scaler_W.mean_, beta_original_scale)

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
