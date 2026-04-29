# src.models.eiv.canonical.CoCoElasticNet

## 概述
- 实现 CoCoElasticNet：在 CoCoLasso 框架中使用 Elastic Net 作为内层求解器。
- 在求解 Elastic Net 前通过 ADMM 将修正协方差投影到 PSD 圆锥。

## 理论与公式
- 修正协方差估计:
  $$
  \Sigma_{\text{hat}} = \frac{W^T W}{n} - \Sigma_{uu}
  $$
- PSD 投影与 CoCoLasso 相同。
- 内层 Elastic Net 目标:
  $$
  \min_{\beta} \frac{1}{2n}\lVert y - W\beta \rVert_2^2 + \alpha\,\text{l1\_ratio}\,\lVert \beta \rVert_1 + \frac{\alpha(1-\text{l1\_ratio})}{2}\lVert \beta \rVert_2^2
  $$

## 逐行说明
- `import time`, `import numpy as np`: 计时与数值运算。
- `from sklearn.preprocessing import StandardScaler`: 标准化工具。
- `from sklearn.linear_model import ElasticNet as SklearnElasticNet`: 内层求解器。
- `class CoCoElasticNet:`: 定义估计器。
- 文档字符串行: 描述参数与求解器限制。
- `def __init__(...)`: 保存超参数、ADMM 设置与 Elastic Net 设置。
- `self.coef_`, `self.intercept_`, `self.scaler_W_`, `self.scaler_y_`: 拟合状态占位。
- 诊断字段（`fit_time_`, `admm_n_iter_` 等）: 记录耗时与稳定性。
- `_project_psd`: ADMM PSD 投影。
- `if self.rho <= 0`: 校验 ADMM 罚参数。
- `X`, `Z`, `U`: ADMM 变量初始化。
- 循环: 更新 `X`、特征分解、截断特征值、更新 `Z` 与 `U`。
- `step_norm` 与 `if step_norm < tol_admm`: 收敛判断与记录。
- `def fit(self, W, y):`: 训练入口。
- 校验 `alpha >= 0`、`l1_ratio` 在 [0,1] 且仅支持 `'cd'`。
- `fit_start = time.perf_counter()`: 计时开始。
- `n_samples, n_features = W.shape`: 维度。
- `Sigma_uu` 未提供时默认零矩阵。
- 校验 `Sigma_uu` 形状。
- 标准化: `W_scaled`、`y_centered` 与 `W_std_safe`。
- `Sigma_uu_scaled = Sigma_uu / outer(W_std_safe, W_std_safe)`: 缩放误差协方差。
- `Sigma_hat` 与 `rho_hat`: 修正协方差估计与交叉项。
- `Sigma_tilde = self._project_psd(Sigma_hat)`: PSD 投影。
- 对称化与抖动: 记录最小特征值并必要时加入抖动。
- Cholesky 分解失败时重试并记录抖动幅度。
- `W_tilde`, `y_tilde`: 变换后的设计矩阵与响应。
- `enet = SklearnElasticNet(...)`: 配置内层求解器。
- `enet.fit(W_tilde, y_tilde)`: 求解弹性网络。
- `beta_scaled = enet.coef_`: 标准化空间系数。
- `self.enet_n_iter_`: 记录内层迭代次数。
- 反缩放系数并处理零方差特征。
- `self.coef_`, `self.intercept_`: 保存拟合结果。
- `self.fit_time_ = ...`: 记录耗时。
- `return self`: 链式调用。
- `def predict(self, W)`: 预测方法。
- `return W @ self.coef_ + self.intercept_`: 线性预测。

## 备注
- 由于中心化逻辑限制，内层 Elastic Net 仅支持坐标下降。
- PSD 投影诊断有助于定位高维场景下的不稳定性。
