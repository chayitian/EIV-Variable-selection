# CoCoLasso 理论推导与 Python 实现

本文档合并了 CoCoLasso 的理论推导和 Python 实现说明。对应代码位于：

```text
src/CoCoLasso.py
```

---

## 第一部分：理论推导

### 1. 问题背景

经典高维线性模型假设：

$$
y = X\beta^\star + \varepsilon,
$$

其中：

- $X \in \mathbb{R}^{n\times p}$ 是真实设计矩阵；
- $y \in \mathbb{R}^{n}$ 是响应变量；
- $\beta^\star \in \mathbb{R}^{p}$ 是真实稀疏系数；
- $\varepsilon$ 是噪声；
- 高维场景常有 $p \gg n$。

如果能直接观测到 $X$，Lasso 解：

$$
\widehat{\beta}_{\text{lasso}}
=
\arg\min_{\beta}
\left\{
\frac{1}{2n}\lVert y-X\beta\rVert_2^2
+\lambda\lVert\beta\rVert_1
\right\}.
$$

展开平方项：

$$
\frac{1}{2n}\lVert y-X\beta\rVert_2^2
=
\frac{1}{2}\beta^\top
\left(\frac{X^\top X}{n}\right)\beta
-
\left(\frac{X^\top y}{n}\right)^\top\beta
+
\frac{y^\top y}{2n}.
$$

最后一项与 $\beta$ 无关，可以忽略。因此 Lasso 等价于协方差形式：

$$
\min_{\beta}
\frac{1}{2}\beta^\top \widehat{\Sigma}\beta
-\widehat{\rho}^{\top}\beta
+\lambda\lVert\beta\rVert_1,
$$

其中：

$$
\widehat{\Sigma}=\frac{X^\top X}{n},
\qquad
\widehat{\rho}=\frac{X^\top y}{n}.
$$

CoCoLasso 的核心问题是：真实的 $X$ 不可见，只能观测到污染后的 $Z$。因此不能直接用 $Z^\top Z/n$ 替代 $X^\top X/n$，否则会产生系统性偏差。

### 2. CoCoLasso 的核心思想

CoCoLasso 的名称可以理解为 Convex Conditioned Lasso。它做两件事：

1. 根据污染机制构造对真实协方差 $\Sigma_X$ 和真实相关向量 $\rho_X$ 的无偏或近似无偏估计：

$$
\widetilde{\Sigma}
\approx
\frac{X^\top X}{n},
\qquad
\widetilde{\rho}
\approx
\frac{X^\top y}{n}.
$$

2. 因为 $\widetilde{\Sigma}$ 可能不是正半定矩阵，将其投影到正半定集合，得到 $\check{\Sigma}$：

$$
\check{\Sigma}
=
\Pi_{\text{PSD}}(\widetilde{\Sigma}).
$$

然后解：

$$
\widehat{\beta}
=
\arg\min_{\beta}
\left\{
\frac{1}{2}\beta^\top \check{\Sigma}\beta
-\widetilde{\rho}^{\top}\beta
+\lambda\lVert\beta\rVert_1
\right\}.
$$

这一步让目标函数重新变成凸问题，便于稳定优化。

### 3. 加性误差模型推导

#### 3.1 模型假设

加性误差下观测矩阵为：

$$
Z = X + W,
$$

其中 $W$ 是测量误差。通常假设：

$$
\mathbb{E}(W)=0,
\qquad
W\perp X,
\qquad
W\perp\varepsilon,
\qquad
\operatorname{Cov}(W_i)=\Sigma_W.
$$

本项目 R 包与 Python 实现采用简化情形：

$$
\Sigma_W = \tau^2 I.
$$

#### 3.2 协方差校正

观察样本协方差：

$$
\frac{1}{n}Z^\top Z
=
\frac{1}{n}(X+W)^\top(X+W).
$$

展开：

$$
\frac{1}{n}Z^\top Z
=
\frac{1}{n}X^\top X
+
\frac{1}{n}X^\top W
+
\frac{1}{n}W^\top X
+
\frac{1}{n}W^\top W.
$$

在独立、零均值条件下：

$$
\mathbb{E}\left(\frac{1}{n}X^\top W\right)=0,
\qquad
\mathbb{E}\left(\frac{1}{n}W^\top X\right)=0,
$$

并且：

$$
\mathbb{E}\left(\frac{1}{n}W^\top W\right)=\Sigma_W.
$$

所以：

$$
\mathbb{E}\left(\frac{1}{n}Z^\top Z\right)
=
\Sigma_X+\Sigma_W.
$$

因此真实协方差的校正估计为：

$$
\widetilde{\Sigma}
=
\frac{1}{n}Z^\top Z-\Sigma_W.
$$

在 $\Sigma_W=\tau^2 I$ 时：

$$
\widetilde{\Sigma}
=
\frac{1}{n}Z^\top Z-\tau^2 I.
$$

代码对应：

```python
cov = (z.T @ z) / n
sigma_tilde = cov - tau**2 * np.eye(p)
```

Python 中对应函数：

```python
corrected_covariance(z, noise="additive", tau=tau)
```

注意：`tau` 必须与当前输入矩阵的尺度一致。如果 `scale_z=True`，理论上也应同步调整 `tau`。

#### 3.3 响应相关向量校正

响应相关项：

$$
\frac{1}{n}Z^\top y
=
\frac{1}{n}(X+W)^\top y
=
\frac{1}{n}X^\top y
+
\frac{1}{n}W^\top y.
$$

若 $W$ 与 $y$ 独立或至少满足 $\mathbb{E}(W^\top y/n)=0$，则：

$$
\mathbb{E}\left(\frac{1}{n}Z^\top y\right)
=
\mathbb{E}\left(\frac{1}{n}X^\top y\right).
$$

所以：

$$
\widetilde{\rho}
=
\frac{1}{n}Z^\top y.
$$

代码对应：

```python
rho_tilde = z.T @ y / n
corrected_rho(z, y, noise="additive")
```

### 4. 缺失数据模型推导

#### 4.1 模型设定

设 $M_{ij}$ 是观测指示变量：

$$
M_{ij}
=
\begin{cases}
1, & X_{ij}\text{ 被观测到},\\
0, & X_{ij}\text{ 缺失}.
\end{cases}
$$

中心化后将缺失项填为 0，记：

$$
\widetilde{Z}_{ij}=M_{ij}X_{ij}.
$$

这里为了推导清晰，假设 $X$ 已中心化，且缺失机制与 $X,y$ 独立或满足 missing completely at random 的近似条件。

#### 4.2 共同观测比例

第 $j$ 列和第 $k$ 列同时被观测的比例：

$$
R_{jk}
=
\frac{1}{n}\sum_{i=1}^{n}M_{ij}M_{ik}.
$$

在代码中：

```python
observed = (~np.isnan(z)).astype(float)
ratio = observed.T @ observed / n
```

Python 中对应函数：

```python
ratio = _ratio_matrix_missing(z)
```

#### 4.3 协方差校正

考虑填零后的协方差元素：

$$
\frac{1}{n}\widetilde{Z}_j^\top \widetilde{Z}_k
=
\frac{1}{n}\sum_{i=1}^{n}M_{ij}M_{ik}X_{ij}X_{ik}.
$$

如果 $M$ 与 $X$ 独立，则：

$$
\mathbb{E}\left(
\frac{1}{n}\widetilde{Z}_j^\top \widetilde{Z}_k
\right)
\approx
R_{jk}\Sigma_{X,jk}.
$$

所以：

$$
\widetilde{\Sigma}_{jk}
=
\frac{
\frac{1}{n}\widetilde{Z}_j^\top \widetilde{Z}_k
}{R_{jk}}.
$$

矩阵形式：

$$
\widetilde{\Sigma}
=
\left(\frac{1}{n}\widetilde{Z}^{\top}\widetilde{Z}\right)
\oslash R,
$$

其中 $\oslash$ 表示逐元素除法。

代码对应：

```python
sigma_tilde = (z_filled.T @ z_filled / n) / ratio_matrix
corrected_covariance(z_work, noise="missing", ratio_matrix=ratio)
```

#### 4.4 响应相关向量校正

第 $j$ 个变量与响应的相关项：

$$
\frac{1}{n}\widetilde{Z}_j^\top y
=
\frac{1}{n}\sum_{i=1}^{n}M_{ij}X_{ij}y_i.
$$

记：

$$
R_{jj}
=
\frac{1}{n}\sum_{i=1}^{n}M_{ij}.
$$

则：

$$
\mathbb{E}\left(
\frac{1}{n}\widetilde{Z}_j^\top y
\right)
\approx
R_{jj}\rho_{X,j}.
$$

因此：

$$
\widetilde{\rho}_j
=
\frac{
\frac{1}{n}\widetilde{Z}_j^\top y
}{R_{jj}}.
$$

代码对应：

```python
rho_tilde = (z_filled.T @ y / n) / np.diag(ratio_matrix)
corrected_rho(z_work, y_work, noise="missing", ratio_matrix=ratio)
```

### 5. 为什么要做 PSD 投影

Lasso 协方差形式中的二次项是：

$$
\frac{1}{2}\beta^\top \Sigma\beta.
$$

如果 $\Sigma\succeq 0$，该二次项是凸函数。若 $\widetilde{\Sigma}$ 存在负特征值，则目标函数可能非凸：

$$
\frac{1}{2}\beta^\top \widetilde{\Sigma}\beta
-\widetilde{\rho}^{\top}\beta
+\lambda\lVert\beta\rVert_1.
$$

非凸目标可能导致局部极小、不稳定路径和坐标下降发散。CoCoLasso 的关键修复是：

$$
\check{\Sigma}
=
\arg\min_{\Sigma\succeq \epsilon I}
\lVert \Sigma-\widetilde{\Sigma}\rVert_{\max}.
$$

这里：

$$
\lVert A\rVert_{\max}
=
\max_{j,k}|A_{jk}|.
$$

投影后优化：

$$
\min_{\beta}
\frac{1}{2}\beta^\top \check{\Sigma}\beta
-\widetilde{\rho}^{\top}\beta
+\lambda\lVert\beta\rVert_1
$$

又成为凸问题。

默认使用 ADMM：

```python
sigma_psd = admm_proj(sigma_tilde, mu=10, etol=1e-4)
```

也可以使用 HM 投影：

```python
sigma_psd = hm_proj(sigma_tilde, ratio=ratio, mu=10, tolerance=1e-4)
```

入口参数中使用 `mode="ADMM"` 或 `mode="HM"` 选择。

### 6. ADMM 投影推导视角

原问题：

$$
\min_{R}
\lVert R-\widetilde{\Sigma}\rVert_{\max}
\quad
\text{s.t.}
\quad
R\succeq \epsilon I.
$$

R 包实现通过引入辅助变量 $S$，令：

$$
R - S = \widetilde{\Sigma}.
$$

于是 $S$ 表示投影矩阵与原矩阵的差：

$$
S = R-\widetilde{\Sigma}.
$$

最大范数约束可通过对 $S$ 的下三角向量做 $L_1$ 球投影间接处理。每轮 ADMM 包含三步：

#### 6.1 R 步：PSD 投影

构造：

$$
W = \widetilde{\Sigma}+S+\mu L.
$$

对 $W$ 做特征分解：

$$
W = Q\Lambda Q^\top.
$$

将特征值截断到至少 $\epsilon$：

$$
R
\leftarrow
Q\max(\Lambda,\epsilon I)Q^\top.
$$

代码对应：

```python
eigvals, eigvecs = np.linalg.eigh((w + w.T) / 2.0)
r = eigvecs @ np.diag(np.maximum(eigvals, epsilon)) @ eigvecs.T
```

#### 6.2 S 步：最大范数相关更新

令：

$$
M = R-\widetilde{\Sigma}-\mu L.
$$

R 代码中对 $M$ 的下三角部分使用 $L_1$ 球投影：

$$
S_{\operatorname{tri}}
\leftarrow
M_{\operatorname{tri}}
-
\Pi_{\lVert\cdot\rVert_1\le \mu/2}(M_{\operatorname{tri}}).
$$

Python 对应：

```python
s[tril] = m[tril] - l1_projection(m[tril], radius=mu / 2.0)
```

#### 6.3 L 步：乘子更新

$$
L
\leftarrow
L-\frac{R-S-\widetilde{\Sigma}}{\mu}.
$$

代码对应：

```python
lagrange = lagrange - (r - s - mat) / mu
```

### 7. 协方差形式 Lasso 的坐标更新推导

目标函数：

$$
f(\beta)
=
\frac{1}{2}\beta^\top\check{\Sigma}\beta
-\widetilde{\rho}^{\top}\beta
+\lambda\lVert\beta\rVert_1.
$$

固定除 $\beta_j$ 以外的坐标，只看与 $\beta_j$ 有关的项：

$$
f_j(\beta_j)
=
\frac{1}{2}\check{\Sigma}_{jj}\beta_j^2
+
\beta_j\sum_{k\ne j}\check{\Sigma}_{jk}\beta_k
-
\widetilde{\rho}_j\beta_j
+
\lambda|\beta_j|
+C.
$$

令：

$$
a_j
=
\widetilde{\rho}_j
-
\sum_{k\ne j}\check{\Sigma}_{jk}\beta_k.
$$

则子问题为：

$$
\min_{\beta_j}
\frac{1}{2}\check{\Sigma}_{jj}\beta_j^2
-a_j\beta_j
+\lambda|\beta_j|.
$$

软阈值解：

$$
\beta_j
\leftarrow
\frac{S(a_j,\lambda)}{\check{\Sigma}_{jj}},
$$

其中：

$$
S(a,\lambda)
=
\begin{cases}
a-\lambda, & a>\lambda,\\
0, & |a|\le \lambda,\\
a+\lambda, & a<-\lambda.
\end{cases}
$$

Python 实现中维护：

$$
s=\check{\Sigma}\beta,
$$

这样每次更新一个坐标后只需：

$$
s
\leftarrow
s+\check{\Sigma}_{:j}(\beta_j^{\text{new}}-\beta_j^{\text{old}}),
$$

避免每次完整计算 $\check{\Sigma}\beta$。

代码支持两种惩罚：

- `penalty="lasso"`：普通 $L_1$ 惩罚。
- `penalty="SCAD"`：按 R 包逻辑实现的局部线性近似权重。

### 8. Lambda 路径

当 $\beta=0$ 是最优解时，Lasso 的 KKT 条件要求：

$$
|\widetilde{\rho}_j|\le \lambda,
\qquad
j=1,\dots,p.
$$

因此最大 lambda 取：

$$
\lambda_{\max}
=
\lVert \widetilde{\rho}\rVert_\infty.
$$

算法从 $\lambda_{\max}$ 开始，逐渐减小到：

$$
\lambda_{\min}
=
\text{lambda.factor}\cdot\lambda_{\max}.
$$

Python 中使用对数等距路径：

$$
\lambda_1>\lambda_2>\cdots>\lambda_m.
$$

代码对应：

```python
lambda_values = np.exp(
    np.linspace(np.log(lambda_max), np.log(lambda_min), steps)
)
```

### 9. 交叉验证误差

在第 $k$ 折中，用训练折求得 $\widehat{\beta}_{-k}(\lambda)$，然后用测试折的校正协方差和相关向量评价：

$$
\operatorname{Err}_k(\lambda)
=
\widehat{\beta}_{-k}^{\top}
\check{\Sigma}_{k}
\widehat{\beta}_{-k}
-
2\widetilde{\rho}_{k}^{\top}
\widehat{\beta}_{-k}.
$$

这里省略了与 $\beta$ 无关的 $y^\top y/n$ 项，因为不同 $\lambda$ 比较时它不影响选择。

最终：

$$
\operatorname{Err}(\lambda)
=
\frac{1}{K}\sum_{k=1}^{K}\operatorname{Err}_k(\lambda).
$$

选择：

$$
\lambda_{\text{opt}}
=
\arg\min_{\lambda}\operatorname{Err}(\lambda).
$$

one-standard-error 规则选择更稀疏的：

$$
\lambda_{\text{sd}}
=
\max
\left\{
\lambda:
\operatorname{Err}(\lambda)
\le
\operatorname{Err}(\lambda_{\text{opt}})
+
\operatorname{sd}(\lambda_{\text{opt}})
\right\}.
$$

### 10. BD-CoCoLasso 的分块推导

BD-CoCoLasso 适合只有部分变量被污染的场景。设：

$$
Z = [X_1 \mid Z_2],
$$

其中：

- $X_1\in\mathbb{R}^{n\times p_1}$：未污染变量；
- $Z_2\in\mathbb{R}^{n\times p_2}$：污染变量；
- $\beta=(\beta_1^\top,\beta_2^\top)^\top$。

#### 10.1 分块目标

对未污染块直接使用普通协方差：

$$
\Sigma_1
=
\frac{1}{n}X_1^\top X_1.
$$

对污染块使用 CoCoLasso 校正并投影：

$$
\check{\Sigma}_2
=
\Pi_{\text{PSD}}(\widetilde{\Sigma}_2).
$$

加性误差时：

$$
\widetilde{\Sigma}_2
=
\frac{1}{n}Z_2^\top Z_2-\tau^2 I.
$$

缺失数据时：

$$
\widetilde{\Sigma}_2
=
\left(\frac{1}{n}\widetilde{Z}_2^\top\widetilde{Z}_2\right)\oslash R.
$$

BD-CoCoLasso 交替解决两个子问题：

$$
\beta_1
\leftarrow
\arg\min_{\beta_1}
\frac{1}{2}\beta_1^\top\Sigma_1\beta_1
-
\rho_1(\beta_2)^\top\beta_1
+
\lambda\lVert\beta_1\rVert_1,
$$

$$
\beta_2
\leftarrow
\arg\min_{\beta_2}
\frac{1}{2}\beta_2^\top\check{\Sigma}_2\beta_2
-
\rho_2(\beta_1)^\top\beta_2
+
\lambda\lVert\beta_2\rVert_1.
$$

这两个子问题都可以调用同一个 `lasso_covariance()`。

#### 10.2 加性误差下的块更新

固定 $\beta_2$，更新未污染块：

$$
\rho_1(\beta_2)
=
\frac{1}{n}X_1^\top(y-Z_2\beta_2).
$$

固定 $\beta_1$，更新污染块：

$$
\rho_2(\beta_1)
=
\frac{1}{n}Z_2^\top(y-X_1\beta_1).
$$

代码对应：

```python
rho1_partial = x1.T @ (y - z2 @ beta2) / n
beta1 = lasso_covariance(sigma1, rho1_partial, lam, beta1)

rho2_partial = z2.T @ (y - x1 @ beta1) / n
beta2 = lasso_covariance(sigma2, rho2_partial, lam, beta2)
```

#### 10.3 缺失数据下的块更新

缺失污染块需要用观测比例修正。记：

$$
D=\operatorname{diag}(R),
\qquad
Z_2^{\dagger}=Z_2D^{-1}.
$$

更新 $\beta_1$：

$$
\rho_1(\beta_2)
=
\frac{1}{n}X_1^\top(y-Z_2^{\dagger}\beta_2).
$$

更新 $\beta_2$：

$$
\rho_2(\beta_1)
=
D^{-1}
\left[
\frac{1}{n}Z_2^\top(y-X_1\beta_1)
\right].
$$

代码对应：

```python
ratio_diag = np.diag(ratio_matrix)
z2_tilde = z2 / ratio_diag

rho1_partial = x1.T @ (y - z2_tilde @ beta2) / n
rho2_partial = z2.T @ (y - x1 @ beta1) / n / ratio_diag
```

### 11. SCAD 惩罚在实现中的近似

R 包也支持 `penalty="SCAD"`。SCAD 是非凸惩罚，原始形式较复杂。代码采用局部线性近似，即将第 $j$ 个坐标的惩罚写成加权 Lasso：

$$
\lambda_j = w_j\lambda.
$$

权重为：

$$
w_j
=
\begin{cases}
1, & |\beta_j|\le \lambda,\\
\dfrac{a\lambda-|\beta_j|}{(a-1)\lambda}, & \lambda<|\beta_j|\le a\lambda,\\
0, & |\beta_j|>a\lambda,
\end{cases}
$$

其中项目中使用：

$$
a=3.7.
$$

然后坐标更新变为：

$$
\beta_j
\leftarrow
\frac{S(a_j,w_j\lambda)}{\check{\Sigma}_{jj}}.
$$

---

## 第二部分：Python 实现说明

### 12. 原 R 包的算法主线

本项目处理高维 error-in-variables 回归。普通 Lasso 假设观测到的设计矩阵就是干净矩阵 $X$，但 CoCoLasso 面对的是被污染的观测矩阵 $Z$。

两类污染：

- 加性误差：$Z = X + W$，其中 $W$ 是测量误差，参数 `tau` 表示其标准差。
- 缺失数据：$Z$ 中存在缺失值，算法通过共同观测比例修正协方差。

R 包中的主入口是 `coco()`：

- `block = FALSE`：普通 CoCoLasso，对所有变量统一进行协方差校正。
- `block = TRUE`：BD-CoCoLasso，将变量分为未污染块 $X_1$ 和污染块 $Z_2$，交替做块坐标下降。

核心目标函数写成协方差形式：

$$
\min_{\beta \in \mathbb{R}^p}
\frac{1}{2}\beta^\top \Sigma \beta
- \rho^\top \beta
+ \lambda \lVert \beta \rVert_1.
$$

其中 $\Sigma$ 和 $\rho$ 不是普通样本协方差，而是根据污染类型修正后的版本。

### 13. R 函数与 Python 函数对应关系

| R 文件/函数 | Python 函数 | 作用 |
|---|---|---|
| `R/coco.R::coco` | `coco()` | 总入口，选择普通 CoCoLasso 或 BD-CoCoLasso |
| `R/pathwise_coordinate_descent.R` | `pathwise_coordinate_descent()` | 普通 CoCoLasso 的 lambda 路径和交叉验证 |
| `R/blockwise_coordinate_descent.R` | `blockwise_coordinate_descent()` | 两块 BD-CoCoLasso |
| `R/lasso_covariance.R` | `lasso_covariance()` | 协方差形式 Lasso 的坐标下降 |
| `R/lasso_covariance_block.R` | `lasso_covariance_block()` | BD-CoCoLasso 的块坐标下降 |
| `R/ADMM_proj.R` | `admm_proj()` | 将修正协方差投影到 PSD 矩阵 |
| `R/HM_proj.R` | `hm_proj()` | HM-Lasso 风格的 PSD 投影 |
| `R/cv_covariance_matrices*.R` | `_cv_matrices()` / `_cv_matrices_block()` | 交叉验证中训练折、测试折的协方差准备 |

### 14. 实现流程总览

普通 CoCoLasso：

```text
输入 Z, y
  |
  |-- 中心化/标准化；missing 数据额外计算共同观测比例 R
  |
  |-- 构造校正协方差 Sigma_tilde 与 rho_tilde
  |
  |-- PSD 投影得到 Sigma_check
  |
  |-- 生成 lambda 路径
  |
  |-- 对每个 lambda:
        |-- K 折交叉验证
        |-- 协方差形式 Lasso 坐标下降
        |-- 记录误差和系数
  |
  |-- 选择 lambda_opt 与 lambda_sd
  |
输出 CocoResult
```

BD-CoCoLasso：

```text
输入 Z=[X1 | Z2], y, p1, p2
  |
  |-- X1 普通协方差
  |-- Z2 校正协方差 + PSD 投影
  |
  |-- 对每个 lambda:
        |-- 固定 beta2，更新 beta1
        |-- 固定 beta1，更新 beta2
        |-- 重复直到收敛
  |
输出 CocoResult
```

### 15. 代码阅读建议

建议按这个顺序阅读 `src/CoCoLasso.py`：

1. `corrected_covariance()` 和 `corrected_rho()`：理解污染校正。
2. `admm_proj()`：理解为什么投影到 PSD。
3. `lasso_covariance()`：理解坐标下降更新。
4. `pathwise_coordinate_descent()`：理解普通 CoCoLasso 的完整训练流程。
5. `lasso_covariance_block()`：理解 BD-CoCoLasso 的交替更新。
6. `blockwise_coordinate_descent()`：理解分块版本的完整训练流程。

---

## 第三部分：使用示例

### 16. 普通 CoCoLasso 示例

```python
import numpy as np
from src import CoCoLasso

rng = np.random.default_rng(0)
n, p = 80, 20
x = rng.normal(size=(n, p))
beta_true = np.zeros(p)
beta_true[[0, 3, 7]] = [1.5, -1.0, 0.8]

tau = 0.3
z = x + rng.normal(scale=tau, size=(n, p))
y = x @ beta_true + rng.normal(scale=0.5, size=n)

model = CoCoLasso(
    block=False,
    noise="additive",
    tau=tau,
    steps=50,
    k=4,
    center_z=True,
    scale_z=False,
    random_state=123,
)
model.fit(z, y)

print(model.lambda_opt_)
print(model.coef_)
print(model.predict(z[:5]))
```

### 17. 缺失数据 CoCoLasso 示例

```python
import numpy as np
from src import CoCoLasso

rng = np.random.default_rng(1)
n, p = 80, 20
x = rng.normal(size=(n, p))
beta_true = np.zeros(p)
beta_true[[0, 4, 9]] = [1.2, -0.8, 1.0]

z = x.copy()
mask = rng.uniform(size=z.shape) < 0.15
z[mask] = np.nan
y = x @ beta_true + rng.normal(scale=0.5, size=n)

model = CoCoLasso(
    block=False,
    noise="missing",
    steps=50,
    k=4,
    center_z=True,
    scale_z=True,
    random_state=123,
)
model.fit(z, y)

print(model.lambda_sd_)
print(model.coef_)
```

### 18. BD-CoCoLasso 示例

BD-CoCoLasso 要求矩阵列顺序如下：

$$
Z = [X_1 \mid Z_2],
$$

其中 $p_1$ 是未污染变量个数，$p_2$ 是污染变量个数。

```python
import numpy as np
from src import CoCoLasso

rng = np.random.default_rng(2)
n, p1, p2 = 100, 30, 10
p = p1 + p2

x = rng.normal(size=(n, p))
beta_true = np.zeros(p)
beta_true[[1, 12, 35]] = [1.0, -1.2, 0.9]

tau = 0.25
z = x.copy()
z[:, p1:] = z[:, p1:] + rng.normal(scale=tau, size=(n, p2))
y = x @ beta_true + rng.normal(scale=0.5, size=n)

model = CoCoLasso(
    block=True,
    p1=p1,
    p2=p2,
    noise="additive",
    tau=tau,
    steps=50,
    k=4,
    center_z=True,
    scale_z=False,
    random_state=123,
)
model.fit(z, y)

print(model.coef_)
```

---

## 第四部分：返回对象与字段对应

### 19. 返回对象

`CoCoLasso.fit()` 后，模型实例包含以下主要属性：

| 属性 | 含义 |
|---|---|
| `lambda_opt_` | 交叉验证平均误差最小的 $\lambda$ |
| `lambda_sd_` | one-standard-error 规则选择的更保守 $\lambda$ |
| `coef_` | `lambda_sd` 对应系数（原始尺度） |
| `intercept_` | 对应截距 |
| `coef_opt_` | `lambda_opt` 对应系数（原始尺度） |
| `intercept_opt_` | 对应截距 |
| `data_error_` | 每个 $\lambda$ 的误差路径，列为 `lambda, error, error_q10, error_q90, error_sd` |
| `data_beta_` | 每个 $\lambda$ 的系数路径，第一列为 $\lambda$ |
| `early_stopping_` | 实际运行的 lambda 步数 |
| `result_` | 完整的 `CocoResult` 对象 |

`CocoResult` 方法：

```python
model.result_.coef()              # 返回整个路径的系数，形状为 p x steps
model.result_.coef(s=fit.lambda_sd) # 返回最接近该 lambda 的系数
model.predict(new_x)              # 使用 beta_sd 预测
```

### 20. 理论对象与 Python 字段对应

| 理论对象 | Python 字段/变量 |
|---|---|
| $Z$ | `z` |
| $y$ | `y` |
| $R$ | `ratio_matrix` |
| $\widetilde{\Sigma}$ | `corrected_covariance(...)` 的返回值 |
| $\widetilde{\rho}$ | `corrected_rho(...)` 的返回值 |
| $\check{\Sigma}$ | `project_covariance(...)` 的返回值 |
| $\lambda_{\max}$ | `lambda_max` |
| $\lambda$ 路径 | `lambda_values` |
| $\widehat{\beta}(\lambda)$ | `beta_path` |
| $\lambda_{\text{opt}}$ | `model.lambda_opt_` |
| $\lambda_{\text{sd}}$ | `model.lambda_sd_` |
| $\widehat{\beta}_{\text{opt}}$ | `model.coef_opt_` |
| $\widehat{\beta}_{\text{sd}}$ | `model.coef_` |

---

## 第五部分：注意事项与局限

### 21. 与 R 包的差异

1. Python 版本使用 NumPy 实现，没有依赖 R 的 `emdbook::lseq`、`rlist` 等包。
2. 交叉验证 folds 通过 `random_state` 控制可重复性；R 版本使用 `sample()`。
3. `lambda_sd` 使用常见的 one-standard-error 规则：在误差不超过 $\min(\text{error}) + \text{sd}_{\min}$ 的 lambda 中选最大的 lambda。
4. 当前 Python 文件实现了普通 CoCoLasso 和两块 BD-CoCoLasso。R 包的 `generalcoco()` 三块 mixed-noise 版本可按 `lasso_covariance_block()` 扩展为三块交替更新。
5. 高维场景下 PSD 投影是最耗时步骤。如果 $p$ 很大，优先考虑 BD-CoCoLasso，只对污染块做 PSD 投影。
6. 对 missing data，不能有整列完全缺失，也不能有两列从未共同观测；否则共同观测比例矩阵不可用。

### 22. 重要假设与局限

1. 加性误差需要已知或估计 $\tau$。若 $\tau$ 设置错误，校正协方差会偏差。
2. 缺失数据推导依赖缺失机制近似独立。若是 MNAR，校正可能失效。
3. 缺失数据不能有整列完全缺失，也不能有两列从未共同观测，否则 $R_{jk}=0$ 会导致除零。
4. PSD 投影在高维下很耗时，主要成本来自特征分解。
5. BD-CoCoLasso 假设污染变量只占较小比例，因此只对污染块做昂贵的 PSD 投影。
6. 当前 Python 实现覆盖普通 CoCoLasso 与两块 BD-CoCoLasso；三块 `generalcoco()` 可按相同思想扩展。
