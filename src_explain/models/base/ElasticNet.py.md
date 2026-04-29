# src.models.base.ElasticNet

## 概述
- 实现 Elastic Net 回归，支持坐标下降、ADMM 或 ISTA 求解。
- 通过 `l1_ratio` 在稀疏性（L1）与收缩性（L2）之间折中。

## 理论与公式
- 目标函数:
  $$
  \min_{\beta} \frac{1}{2n}\lVert y - X\beta \rVert_2^2 + \alpha\,\text{l1\_ratio}\,\lVert \beta \rVert_1 + \frac{\alpha(1-\text{l1\_ratio})}{2}\lVert \beta \rVert_2^2
  $$
- 软阈值算子: 与 Lasso 相同的 $S(\rho,\lambda)$。
- 坐标下降更新使用 $z_j = \frac{\lVert x_j \rVert_2^2}{n} + \alpha(1-\text{l1\_ratio})$。

## 逐行说明
- `import numpy as np`: 数值计算后端。
- `class ElasticNet:`: 定义估计器。
- 文档字符串行: 描述目标函数与参数。
- `def __init__(...)`: 设置惩罚强度、混合比例与求解器选项。
- `self.alpha` 到 `self.ista_step_size`: 保存配置。
- `self.coef_`, `self.intercept_`, `self.n_iter_`: 初始化拟合状态。
- `_soft_threshold`: 软阈值算子。
- `_fit_coordinate_descent`: 坐标下降求解器。
- `l1_penalty` 与 `l2_penalty`: 将弹性网络惩罚拆成 L1 与 L2。
- 内层循环: 计算残差、$\rho_j$ 与包含 L2 的 $z_j$。
- `beta[j] = self._soft_threshold(...)/z_safe`: 更新当前坐标。
- 通过系数最大变化量判断收敛。
- `_fit_admm`: 初始化 ADMM 变量并预计算 Gram 与交叉项。
- `system_matrix = gram + (rho + l2_penalty) * I`: beta 更新的线性系统。
- `z = self._soft_threshold(beta + u, l1_penalty / rho)`: L1 的近端步。
- 通过残差判断收敛。
- `_fit_ista`: 计算包含 L2 的 Lipschitz 常数 $L$。
- `grad = gram @ beta - Xy + l2_penalty * beta`: 光滑项梯度。
- `beta_new = self._soft_threshold(...)`: ISTA 更新。
- `def fit(self, X, y):`: 参数校验与求解器分派。
- 校验 `alpha` 非负、`l1_ratio` 在 [0,1]、`solver` 合法。
- 中心化: 计算均值并去除截距影响。
- 求解器分支设置 `self.coef_`。
- `self.intercept_ = y_mean - X_mean @ beta`: 恢复截距。
- `def predict(self, X):`: 预测方法。
- `return X @ self.coef_ + self.intercept_`: 线性预测。

## 备注
- `l1_ratio=1.0` 退化为 Lasso，`l1_ratio=0.0` 退化为 Ridge。
- 默认不做特征标准化，必要时由调用方自行处理。
