# src.models.eiv.canonical.CRidge

## 概述
- 实现修正 Ridge 回归，适用于含测量误差的数据。
- 在修正协方差系统中加入 L2 正则化。

## 理论与公式
- 目标函数:
  $$
  \min_{\beta} \frac{1}{2n}\lVert y - W\beta \rVert_2^2 - \frac{1}{2}\beta^T\Sigma_{uu}\beta + \frac{\alpha}{2}\lVert \beta \rVert_2^2
  $$
- 闭式解:
  $$
  \beta = (\Sigma_{\text{corrected}} + \alpha I)^{-1}\rho,
  \quad \Sigma_{\text{corrected}} = \frac{W^T W}{n} - \Sigma_{uu}
  $$

## 逐行说明
- `import numpy as np`: 数值计算。
- `from sklearn.preprocessing import StandardScaler`: 标准化工具。
- `class CRidge:`: 定义修正 Ridge 估计器。
- 文档字符串行: 描述目标函数与参数。
- `def __init__(...)`: 设置 `alpha`、`Sigma_uu` 与 `min_eig`。
- `self.coef_`, `self.intercept_`, `self.scaler_W_`, `self.scaler_y_`: 初始化状态。
- `def fit(self, W, y):`: 训练入口。
- `n_samples, n_features = W.shape`: 维度。
- `if self.Sigma_uu is None`: 默认测量误差为零。
- `if Sigma_uu.shape != (n_features, n_features)`: 校验协方差形状。
- `self.scaler_W_` 与 `self.scaler_y_`: 标准化准备。
- `W_scaled = self.scaler_W_.fit_transform(W)`: 标准化特征。
- `y_centered = self.scaler_y_.fit_transform(...).flatten()`: 中心化响应。
- `W_std` 与 `W_std_safe`: 缩放因子与零方差保护。
- `Sigma_uu_scaled = Sigma_uu / np.outer(W_std_safe, W_std_safe)`: 缩放误差协方差。
- `Sigma_W = (W_scaled.T @ W_scaled) / n_samples`: 经验协方差。
- `Sigma_corrected = Sigma_W - Sigma_uu_scaled`: 修正协方差。
- `Sigma_corrected = (Sigma_corrected + Sigma_corrected.T) / 2`: 对称化。
- `current_min_eig` 检查并在必要时加入抖动。
- `rho = (W_scaled.T @ y_centered) / n_samples`: 交叉项。
- `ridge_system = Sigma_corrected + self.alpha * I`: Ridge 线性系统。
- `beta_scaled = np.linalg.solve(ridge_system, rho)`: 求解系数。
- `beta_original_scale = beta_scaled / W_std_safe`: 反缩放。
- `beta_original_scale = np.where(W_std > 1e-12, ...)`: 零方差特征置零。
- `self.coef_` 与 `self.intercept_`: 保存拟合结果。
- `return self`: 链式调用。
- `def predict(self, W)`: 预测方法。
- `return W @ self.coef_ + self.intercept_`: 线性预测。

## 备注
- `min_eig` 用于在协方差近奇异时进行稳定化。
- Ridge 适用于稠密信号或作为稳定的初始估计。
