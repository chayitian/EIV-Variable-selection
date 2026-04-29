# src.models.eiv.adaptive.ACoCoLasso

## 概述
- 实现自适应 CoCoLasso，用于 EIV 回归。
- 将凸约束协方差修正与自适应 L1 权重结合。
- 记录 ADMM 收敛与数值稳定性诊断信息。

## 理论与公式
- 修正协方差估计:
  $$
  \Sigma_{\text{hat}} = \frac{W^T W}{n} - \Sigma_{uu}
  $$
- PSD 投影:
  $$
  \Sigma_{\text{tilde}} = \operatorname{argmin}_{S \succeq 0} \lVert S - \Sigma_{\text{hat}} \rVert_F^2
  $$
- 自适应加权 Lasso 目标:
  $$
  \min_{\beta} \frac{1}{2n}\lVert y - W\beta \rVert_2^2 + \lambda\sum_j w_j |\beta_j|,
  \quad w_j = \frac{1}{(|\hat{\beta}_j^{\text{init}}| + 10^{-8})^{\gamma}}
  $$

## 逐行说明
- `import numpy as np`, `import time`: 数值运算与计时。
- `from sklearn.preprocessing import StandardScaler`: 标准化。
- `from sklearn.linear_model import Lasso`: 内层 Lasso 求解器。
- `class ACoCoLasso:`: 定义估计器。
- 文档字符串行: 描述方法、参数与创新点。
- `def __init__(...)`: 保存超参数、ADMM 设置与初始系数。
- `self.coef_`, `self.intercept_`, `self.beta_init_`, `self.weights_`: 拟合状态。
- 诊断字段（耗时、ADMM 统计、PSD 抖动、Cholesky 重试等）: 便于分析。
- `_project_psd`: ADMM 投影到 PSD 圆锥。
- `if self.rho <= 0`: 校验 ADMM 罚参数。
- `X`, `Z`, `U` 初始化: ADMM 变量。
- ADMM 循环: 更新 `X`，特征分解投影 `Z`，更新对偶 `U`。
- `step_norm_history`: 记录收敛轨迹。
- `if step_norm < tol_admm`: 判断收敛并停止。
- `def fit(self, W, y):`: 训练入口。
- `fit_start = time.perf_counter()`: 开始计时。
- `n_samples, n_features = W.shape`: 维度。
- `Sigma_uu` 未提供时默认零矩阵。
- 校验 `Sigma_uu` 形状。
- 标准化: `W_scaled`、`y_centered` 与 `W_std_safe`。
- `Sigma_uu_scaled = Sigma_uu / outer(W_std_safe, W_std_safe)`: 缩放误差协方差。
- `if self.init_coef is None`: 自适应方法需要初始系数。
- `self.beta_init_ = np.asarray(self.init_coef, ...)`: 校验初始系数。
- `beta_init_scaled = self.beta_init_ * W_std`: 缩放初始系数。
- `weights = 1.0 / (|beta_init_scaled| + 1e-8)^gamma`: 计算权重。
- `self.weights_ = weights.copy()`: 保存权重。
- `inv_weights = 1.0 / weights`: 预计算逆权重。
- `W_inv_diag = np.diag(inv_weights)`: 对角缩放矩阵。
- `Sigma_hat = (W_scaled.T @ W_scaled)/n - Sigma_uu_scaled`: 修正协方差。
- `rho_hat = (W_scaled.T @ y_centered)/n`: 交叉项。
- `Sigma_tilde = self._project_psd(Sigma_hat)`: PSD 投影。
- `Sigma_transformed = W_inv_diag.T @ Sigma_tilde @ W_inv_diag`: 应用权重变换。
- `rho_transformed = W_inv_diag.T @ rho_hat`: 交叉项变换。
- 对称化并检查最小特征值，必要时加入抖动并记录。
- Cholesky 分解失败则重试，并记录重试与抖动幅度。
- `W_tilde = L.T * sqrt(n_samples)`: 变换后的设计矩阵。
- `y_tilde = solve(L, rho_transformed) * sqrt(n_samples)`: 变换后的响应。
- `lasso = Lasso(...)` 与 `lasso.fit(...)`: 变换空间内求解 Lasso。
- `alpha_scaled = lasso.coef_`: 缩放空间系数。
- `beta_scaled = W_inv_diag @ alpha_scaled`: 还原加权影响。
- `beta_original_scale = beta_scaled / W_std_safe`: 反缩放到原尺度。
- `beta_original_scale = np.where(W_std > 1e-12, ...)`: 零方差特征置零。
- `self.coef_` 与 `self.intercept_`: 保存拟合结果。
- `self.fit_time_ = ...`: 记录耗时。
- `return self`: 链式调用。
- `def predict(self, W)`: 预测方法。
- `return W @ self.coef_ + self.intercept_`: 线性预测。

## 备注
- 自适应权重会更强地惩罚小的初始系数，从而增强稀疏性。
- 诊断信息用于评估 PSD 投影与 Cholesky 抖动对稳定性的影响。
