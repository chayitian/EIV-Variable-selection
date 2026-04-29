# src.models.eiv.adaptive.ACLasso

## 概述
- 实现自适应修正 Lasso（ACLasso），用于 EIV 回归。
- 将修正协方差与基于初始估计的自适应权重结合。

## 理论与公式
- 修正 Lasso 目标:
  $$
  \min_{\beta} \frac{1}{2n}\lVert y - W\beta \rVert_2^2 - \frac{1}{2}\beta^T\Sigma_{uu}\beta + \lambda\sum_j w_j |\beta_j|
  $$
- 自适应权重:
  $$
  w_j = \frac{1}{(|\hat{\beta}_j^{\text{init}}| + 10^{-6})^{\gamma}}
  $$
- 通过 $W / w$ 的列缩放将加权 L1 转换为标准 L1。

## 逐行说明
- `import numpy as np`: 数值计算。
- `from sklearn.preprocessing import StandardScaler`: 标准化。
- `from sklearn.linear_model import Lasso`: 内层求解器。
- `class ACLasso:`: 定义估计器。
- 文档字符串行: 描述方法与参数。
- `def __init__(...)`: 保存 `alpha`、`Sigma_uu`、`gamma` 与 `init_coef`。
- `self.coef_`, `self.intercept_`, `self.beta_init_`, `self.weights_`: 初始化状态。
- `def fit(self, W, y):`: 训练入口。
- `n_samples, n_features = W.shape`: 维度。
- `Sigma_uu` 未提供时默认零矩阵。
- 校验 `Sigma_uu` 形状。
- `scaler_W` 与 `scaler_y`: 标准化特征并中心化响应。
- `W_scaled = scaler_W.fit_transform(W)`: 标准化 W。
- `y_centered = scaler_y.fit_transform(...).flatten()`: 中心化 y。
- `W_std` 与 `W_std_safe`: 缩放因子与零方差保护。
- `Sigma_uu_scaled = Sigma_uu / outer(W_std_safe, W_std_safe)`: 缩放误差协方差。
- `if self.init_coef is None`: 需要初始估计用于权重。
- `self.beta_init_ = np.asarray(self.init_coef, ...)`: 校验初始系数。
- `beta_init_scaled = self.beta_init_ * W_std`: 缩放初始系数。
- `weights = 1.0 / (|beta_init_scaled| + 1e-6)^gamma`: 计算自适应权重。
- `inv_weights = 1.0 / weights`: 预计算逆权重。
- `W_weighted = W_scaled * inv_weights[np.newaxis, :]`: 列加权。
- `Sigma_W_weighted = (W_weighted.T @ W_weighted) / n`: 加权协方差。
- `Sigma_uu_weighted = diag(inv_weights) @ Sigma_uu_scaled @ diag(inv_weights)`: 加权误差协方差。
- `Sigma_corrected = Sigma_W_weighted - Sigma_uu_weighted`: 加权空间的修正协方差。
- 对称化并加入抖动以保证 PSD。
- Cholesky 分解失败时重试以提高稳定性。
- `rho = (W_weighted.T @ y_centered) / n`: 加权交叉项。
- `W_transformed` 与 `y_transformed`: 变换后的 Lasso 系统。
- `lasso = Lasso(...)` 与 `lasso.fit(...)`: 在变换空间求解。
- `alpha_scaled = lasso.coef_`: 加权空间的解。
- `beta_scaled = alpha_scaled * inv_weights`: 还原权重影响。
- `beta_original_scale = beta_scaled / W_std_safe`: 反缩放。
- `beta_original_scale = np.where(W_std > 1e-12, ...)`: 零方差特征置零。
- `self.coef_` 与 `self.intercept_`: 保存拟合结果。
- `return self`: 链式调用。
- `def predict(self, W)`: 预测方法。
- `return W @ self.coef_ + self.intercept_`: 线性预测。

## 备注
- `init_coef` 通常来自 COLS 或 CRidge，用于稳定权重。
- 缩放对数值稳定性至关重要。
