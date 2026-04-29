# src.models.eiv.canonical.CLasso

## 概述
- 实现修正 Lasso，用于含测量误差的 EIV 回归。
- 通过减去测量误差协方差 $\Sigma_{uu}$ 来修正协方差。

## 理论与公式
- 修正 Lasso 目标函数:
  $$
  \min_{\beta} \frac{1}{2n}\lVert y - W\beta \rVert_2^2 - \frac{1}{2}\beta^T\Sigma_{uu}\beta + \lambda\lVert \beta \rVert_1
  $$
- 标准化后修正协方差:
  $$
  \Sigma_{\text{corrected}} = \frac{W^T W}{n} - \Sigma_{uu}
  $$
- 通过 Cholesky 因子 $L$ 将问题变换为标准 Lasso:
  $$
  W_{\text{tilde}} = L^T\sqrt{n},\quad y_{\text{tilde}} = L^{-1}\rho\sqrt{n},\quad \rho = \frac{W^T y}{n}
  $$

## 逐行说明
- `import numpy as np`: 数值计算。
- `from sklearn.preprocessing import StandardScaler`: 标准化工具。
- `from sklearn.linear_model import Lasso`: 内层 Lasso 求解器。
- `class CLasso:`: 定义修正 Lasso 估计器。
- 文档字符串行: 描述目标函数与参数。
- `def __init__(...)`: 保存超参数与拟合状态占位。
- `self.coef_`, `self.intercept_`, `self.scaler_W_`, `self.scaler_y_`, `self.Sigma_uu_scaled_`: 初始化属性。
- `def fit(self, W, y):`: 训练入口。
- `n_samples, n_features = W.shape`: 维度。
- `if self.Sigma_uu is None`: 默认测量误差为零。
- `if Sigma_uu.shape != (n_features, n_features)`: 校验协方差形状。
- `self.scaler_W_ = StandardScaler()`: 标准化特征。
- `self.scaler_y_ = StandardScaler(with_std=False)`: 仅中心化响应。
- `W_scaled = self.scaler_W_.fit_transform(W)`: 标准化 W。
- `y_centered = self.scaler_y_.fit_transform(...).flatten()`: 中心化 y。
- `W_std = self.scaler_W_.scale_`: 特征标准差。
- `W_std_safe = np.where(W_std > 1e-12, W_std, 1.0)`: 防止除零。
- `Sigma_uu_scaled = Sigma_uu / np.outer(W_std_safe, W_std_safe)`: 缩放误差协方差。
- `Sigma_W = (W_scaled.T @ W_scaled) / n_samples`: 经验协方差。
- `Sigma_corrected = Sigma_W - Sigma_uu_scaled`: 修正协方差。
- `min_eig = np.min(np.linalg.eigvalsh(Sigma_corrected))`: 检查 PSD。
- `if min_eig < 1e-4: ...`: 加入对角扰动提升稳定性。
- `try: L = np.linalg.cholesky(Sigma_corrected)`: 分解用于变换。
- `except np.linalg.LinAlgError`: 对称化并重试。
- `rho = (W_scaled.T @ y_centered) / n_samples`: 交叉项。
- `W_transformed = L.T * np.sqrt(n_samples)`: 变换后的设计矩阵。
- `y_transformed = np.linalg.solve(L, rho) * np.sqrt(n_samples)`: 变换后的响应。
- `lasso = Lasso(...)` 与 `lasso.fit(...)`: 变换空间求解 Lasso。
- `beta_scaled = lasso.coef_`: 标准化空间解。
- `beta_original_scale = beta_scaled / W_std_safe`: 反缩放。
- `beta_original_scale = np.where(W_std > 1e-12, beta_original_scale, 0.0)`: 零方差特征置零。
- `self.coef_ = beta_original_scale`: 保存系数。
- `self.intercept_ = scaler_y.mean_ - scaler_W.mean_ @ beta_original_scale`: 恢复截距。
- `return self`: 链式调用。
- `def predict(self, W):`: 预测方法。
- `return W @ self.coef_ + self.intercept_`: 线性预测。

## 备注
- 对角扰动确保 Cholesky 分解前协方差为 PSD。
- 零方差特征对应的系数强制为 0 以避免不稳定。
