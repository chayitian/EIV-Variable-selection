# src.models.eiv.canonical.CoCoLasso

## 概述
- 实现 CoCoLasso（凸约束 Lasso），用于 EIV 回归。
- 在求解 Lasso 前使用 ADMM 将修正协方差投影到 PSD 圆锥。

## 理论与公式
- 修正协方差估计:
  $$
  \Sigma_{\text{hat}} = \frac{W^T W}{n} - \Sigma_{uu}
  $$
- PSD 投影:
  $$
  \Sigma_{\text{tilde}} = \operatorname{argmin}_{S \succeq 0} \lVert S - \Sigma_{\text{hat}} \rVert_F^2
  $$
- 随后在使用 $\Sigma_{\text{tilde}}$ 的变换系统中求解 Lasso。

## 逐行说明
- `import numpy as np`, `import time`: 数值运算与计时。
- `from sklearn.preprocessing import StandardScaler`: 标准化。
- `from sklearn.linear_model import Lasso`: 内层求解器。
- `class CoCoLasso:`: 定义估计器。
- 文档字符串行: 描述方法与参数。
- `def __init__(...)`: 保存超参数与诊断字段。
- `self.coef_`, `self.intercept_`, `self.scaler_W_`, `self.scaler_y_`: 拟合状态占位。
- 诊断字段（`fit_time_`, `admm_n_iter_` 等）: 记录收敛与稳定性信息。
- `_project_psd`: ADMM 投影到 PSD 圆锥。
- `if self.rho <= 0`: 校验 ADMM 罚参数。
- `X`, `Z`, `U` 初始化: ADMM 变量。
- `for iter_idx in range(1, self.max_iter_admm + 1)`: ADMM 迭代。
- `X = (M + rho * Z - U) / (1 + rho)`: 原变量更新。
- `eigvals, eigvecs = np.linalg.eigh(X + U / rho)`: 特征分解。
- `eigvals[eigvals < 0] = 0`: 截断负特征值。
- `Z = eigvecs @ diag(eigvals) @ eigvecs.T`: PSD 投影步。
- `U = U + rho * (X - Z)`: 对偶更新。
- `step_norm = norm(X - X_old)`: 收敛度量。
- `if step_norm < tol_admm`: 收敛检查。
- 保存 `admm_n_iter_`, `admm_converged_`, `admm_step_norm_history_`。
- `def fit(self, W, y):`: 训练入口。
- `fit_start = time.perf_counter()`: 计时开始。
- `n_samples, n_features = W.shape`: 维度。
- `Sigma_uu` 未提供时默认零矩阵。
- 校验 `Sigma_uu` 形状。
- 标准化: `W_scaled`, `y_centered`, `W_std_safe`。
- `Sigma_uu_scaled = Sigma_uu / outer(W_std_safe, W_std_safe)`: 缩放误差协方差。
- `Sigma_hat = (W_scaled.T @ W_scaled) / n - Sigma_uu_scaled`: 修正协方差估计。
- `rho_hat = (W_scaled.T @ y_centered) / n`: 交叉项。
- `Sigma_tilde = self._project_psd(Sigma_hat)`: PSD 投影。
- 对称化与最小特征值检查: 记录 `psd_min_eig_before_` 并加入抖动。
- Cholesky 分解失败时重试，并记录抖动幅度。
- `W_tilde = L.T * sqrt(n_samples)`: 变换后的设计矩阵。
- `y_tilde = solve(L, rho_hat) * sqrt(n_samples)`: 变换后的响应。
- `lasso.fit(W_tilde, y_tilde)`: 变换空间内求解 Lasso。
- `beta_scaled = lasso.coef_`: 缩放空间解。
- `beta_original_scale = beta_scaled / W_std_safe`: 反缩放。
- `beta_original_scale = np.where(W_std > 1e-12, ...)`: 零方差特征置零。
- `self.coef_`, `self.intercept_`: 保存拟合结果。
- `self.fit_time_ = ...`: 记录耗时。
- `return self`: 链式调用。
- `def predict(self, W)`: 预测方法。
- `return W @ self.coef_ + self.intercept_`: 线性预测。

## 备注
- PSD 投影是必要步骤，否则修正后协方差可能非正定。
- 诊断信息用于比较不同数据设置下的数值稳定性。
