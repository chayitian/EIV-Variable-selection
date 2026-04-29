# src.models.base.Lasso

## 概述
- 实现 Lasso 回归，支持坐标下降、ADMM、ISTA 与 LARS 等多种求解器。
- 拟合完成后保存 `coef_` 与 `intercept_`，并在训练中做中心化处理。

## 理论与公式
- 目标函数:
  $$
  \min_{\beta} \frac{1}{2n}\lVert y - X\beta \rVert_2^2 + \alpha \lVert \beta \rVert_1
  $$
- 软阈值算子:
  $$
  S(\rho, \lambda) = \operatorname{sign}(\rho)\max(|\rho| - \lambda, 0)
  $$
- 坐标下降更新（特征 $j$）:
  $$
  \beta_j \leftarrow \frac{S(\rho_j, \alpha)}{z_j},\quad
  \rho_j = \frac{x_j^T r_j}{n},\quad
  z_j = \frac{\lVert x_j \rVert_2^2}{n}
  $$
- ADMM 线性子问题:
  $$
  (\Sigma + \rho I)\beta = X^T y / n + \rho(z - u)
  $$
  然后对 $\beta + u$ 做软阈值。
- ISTA 使用步长 $1/L$，其中 $L$ 为 $\nabla (\frac{1}{2n}\lVert y-X\beta\rVert_2^2)$ 的 Lipschitz 常数。

## 逐行说明
- `import numpy as np`: 数组与线性代数运算。
- `from sklearn.linear_model import LassoLars`: 提供 LARS 路径求解器。
- `class Lasso:`: 定义估计器类。
- 文档字符串行: 说明目标函数、参数与求解器选项。
- `def __init__(...)`: 接收惩罚强度、求解器与相关超参数。
- `self.alpha = alpha` 到 `self.ista_step_size = ista_step_size`: 保存超参数。
- `self.coef_ = None` 到 `self.n_iter_ = 0`: 初始化拟合状态。
- `def _soft_threshold(self, rho, lam):`: 软阈值函数。
- `return np.sign(rho) * np.maximum(np.abs(rho) - lam, 0.0)`: 实现 $S(\rho,\lambda)$。
- `def _fit_coordinate_descent(...)`: 坐标下降求解器。
- `n_samples, n_features = X_centered.shape`: 获取维度。
- `beta = np.zeros(n_features)`: 系数初始化为 0。
- `for iteration in range(self.max_iter):`: 迭代更新。
- `beta_old = beta.copy()`: 保存上一轮系数用于收敛判断。
- `for j in range(n_features):`: 逐坐标更新。
- `residual = ...`: 计算剔除第 $j$ 列的残差。
- `rho = ...`: 相关项 $\rho_j$。
- `z = ...`: 曲率项 $z_j$。
- `z_safe = max(z, 1e-12)`: 防止除零。
- `beta[j] = ...`: 软阈值更新。
- `if np.max(np.abs(beta - beta_old)) < self.tol:`: 收敛判断。
- `self.n_iter_ = iteration + 1` 与 `return beta`: 收敛时提前返回。
- `self.n_iter_ = self.max_iter` 与 `return beta`: 未收敛时返回最后结果。
- `def _fit_admm(...)`: ADMM 求解器。
- `rho = self.admm_rho`: ADMM 罚参数。
- `gram = ...`, `Xy = ...`: 预计算 $X^T X / n$ 与 $X^T y / n$。
- `beta`, `z`, `u` zeros: 初始化 ADMM 变量。
- `system_matrix = gram + rho * np.eye(n_features)`: 系数矩阵缓存。
- ADMM 循环: 更新 `beta`、阈值化 `z`、更新对偶 `u`。
- `primal_residual` 与 `dual_residual`: 监控收敛。
- `if max(primal_residual, dual_residual) < self.tol:`: 双残差同时足够小则停止。
- `def _fit_ista(...)`: ISTA 求解器。
- `lipschitz = max(np.max(np.linalg.eigvalsh(gram)), 1e-12)`: 计算 $L$。
- `step = 1.0 / lipschitz` 或 `self.ista_step_size`: 步长选择。
- `threshold = self.alpha * step`: 每步阈值。
- `grad = gram @ beta - Xy`: 光滑项梯度。
- `beta_new = self._soft_threshold(...)`: 梯度步 + 软阈值。
- 收敛判断并更新 `beta` 和 `self.n_iter_`。
- `def _fit_lars(...)`: 使用 LARS。
- `model = LassoLars(...)` 与 `model.fit(...)`: 计算 Lasso 路径。
- `self.n_iter_ = int(getattr(model, 'n_iter_', self.max_iter))`: 记录迭代次数。
- `return np.asarray(model.coef_, dtype=float)`: 返回系数。
- `def fit(self, X, y):`: 训练入口。
- 校验 `alpha` 非负、`solver` 合法、`admm_rho` 正值。
- `X_mean`, `y_mean`: 计算均值。
- `X_centered`, `y_centered`: 中心化以剔除截距。
- `if self.solver == ...`: 根据求解器分派。
- `self.coef_ = beta`: 保存系数。
- `self.intercept_ = y_mean - X_mean @ beta`: 恢复截距。
- `return self`: 链式调用。
- `def predict(self, X):`: 预测方法。
- `return X @ self.coef_ + self.intercept_`: 线性预测。

## 备注
- 该实现只做中心化，不做标准化；需要时由调用方自行标准化。
- LARS 适合中小规模特征；CD/ADMM/ISTA 在高维场景下各有侧重。
