# src.evaluation.vs_evaluate

## 概述
- 根据真实变量索引与被选择变量索引计算变量选择指标。
- 可选地计算系数估计误差与预测误差。

## 理论与公式
设真实集合为 $T$、选择集合为 $S$、总特征数为 $p$。
- $TP = |T \cap S|$, $FP = |S \setminus T|$, $FN = |T \setminus S|$, $TN = p - TP - FP - FN$。
- 精确率: $TP/(TP+FP)$。
- 召回率: $TP/(TP+FN)$。
- F1: $2\cdot\text{Precision}\cdot\text{Recall}/(\text{Precision}+\text{Recall})$。
- FDR: $FP/(TP+FP)$。
- 精确选择率: $S=T$ 时为 1，否则为 0。
- 特异度: $TN/(TN+FP)$。
- 汉明距离: $\frac{1}{p}\sum_{j=1}^p 1[\hat{z}_j \neq z_j]$。
- MCC:
  $$
  \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
  $$
- 系数误差（提供 $\beta$ 与 $\hat{\beta}$ 时）:
  $$\lVert \hat{\beta}-\beta \rVert_1,\; \lVert \hat{\beta}-\beta \rVert_2,\; \lVert \hat{\beta}-\beta \rVert_{\infty}$$
- MSE: $\lVert \hat{\beta}-\beta \rVert_2^2$。
- 预测误差（提供 $X$ 时）:
  $$
  (\hat{\beta}-\beta)^T\Sigma_X(\hat{\beta}-\beta),\quad \Sigma_X = X^T X / n
  $$

## 逐行说明
- `import numpy as np`: 数值计算与范数运算。
- `def selection_accuracy(...)`: 定义主评估函数，支持可选误差输入。
- 文档字符串行: 描述输入与输出指标。
- `true_set = set(true_indices)` 与 `selected_set = set(selected_indices)`: 转为集合便于快速运算。
- `all_features = set(range(total_features))`: 构造全集索引。
- `TP`, `FP`, `FN`, `TN`: 通过集合运算计算混淆矩阵指标。
- `if total_features <= 0`: 防止无效特征数。
- `model_size`, `accuracy`, `precision`, `recall`, `f1`, `fdr`, `exact_selection`, `specificity`: 基础指标，含零除保护。
- `mcc_numerator` 与 `mcc_denominator`: 数值稳定的 MCC 计算。
- `true_vector` 与 `selected_vector`: 构造二值选择向量。
- `hamming = np.sum(true_vector != selected_vector) / total_features`: 归一化汉明距离。
- `l1_error`, `l2_error`, `linf_error`, `mse`, `pe`: 初始化可选误差指标为 `None`。
- `if beta_true is not None and beta_hat is not None`: 计算系数误差。
- `diff = beta_hat_arr - beta_true_arr`: 系数差向量。
- `np.linalg.norm` 调用: 计算 L1、L2、Linf 误差。
- `mse = float(np.dot(diff, diff))`: L2 范数平方。
- `if x_true is not None`: 计算预测误差。
- `x_true` 的输入校验: 确保形状与样本数有效。
- `sigma_x = (x_true_arr.T @ x_true_arr) / x_true_arr.shape[0]`: 经验协方差。
- `pe = float(diff.T @ sigma_x @ diff)`: 预测误差标量。
- 最后 `return { ... }`: 将指标打包为字典返回。

## 备注
- 该函数用于模拟研究，真实数据通常只能计算部分指标。
- 所有指标基于索引集合计算，需保证与模型输出的特征索引一致。
