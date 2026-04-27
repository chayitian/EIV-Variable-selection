# EIV-Variable-selection

高维测量误差变量选择研究与实验代码库。

本仓库当前围绕 Error-in-Variables (EIV) 线性回归，统一实现了：

1. 基础回归模型（无测量误差）
2. EIV 经典修正模型
3. EIV 自适应模型
4. 基于 SHAP/树重要性的加权实验策略

并在 test 目录提供了可复现实验脚本（CSV 输出）。

---

## 1. 问题设定

### 1.1 标准线性回归

$$
y = X\beta^* + \varepsilon
$$

### 1.2 含测量误差的 EIV 回归

$$
\begin{cases}
y = X\beta^* + \varepsilon \\
W = X + U
\end{cases}
$$

其中：

- $X$ 是不可观测真实协变量
- $W$ 是可观测协变量
- $U$ 是测量误差
- 常用假设是已知或可估计 $\Sigma_{uu}=\mathrm{Cov}(U)$

测量误差会导致直接基于 $W$ 的估计出现偏差，因此需要纠偏建模。

---

## 2. 当前代码结构（与仓库一致）

```text
src/
  __init__.py
  evaluation/
    __init__.py
    vs_evaluate.py
  experiments/
    __init__.py
    comparison_common.py
  models/
    __init__.py
    base/
      __init__.py
      OLS.py
      Lasso.py
      ALasso.py
      ElasticNet.py
    eiv/
      __init__.py
      canonical/
        __init__.py
        COLS.py
        CRidge.py
        CLasso.py
        CoCoLasso.py
        CoCoElasticNet.py
      adaptive/
        __init__.py
        ACLasso.py
        ACoCoLasso.py

test/
  comparison_baseline.py
  ACoCoLasso_reproduction.py
  ShapACoCoLasso_reproduction.py
  CoCoElasticNet.py
  RFACoCoLasso.py
  XGBoostACoCoLasso.py
```

---

## 3. src 框架详解

### 3.1 src/models/base

无测量误差基线模型：

- OLS
- Lasso
- ALasso
- ElasticNet

用途：基线对照、接口验证、无误差场景建模。

### 3.2 src/models/eiv/canonical

EIV 经典修正模型：

- COLS：修正 OLS（常用作自适应模型初始估计）
- CRidge：修正 Ridge
- CLasso：修正 Lasso
- CoCoLasso：凸约束误差修正 Lasso
- CoCoElasticNet：CoCo 框架下 ElasticNet 版本

### 3.3 src/models/eiv/adaptive

EIV 自适应模型：

- ACLasso
- ACoCoLasso

这两个类都遵循同一范式：

1. 外部传入 init_coef
2. 类内部将 init_coef 转换为权重
3. 用重参数化把加权问题转为标准 Lasso 子问题

### 3.4 src/evaluation

selection_accuracy 是统一评估函数，返回：

- 选择指标：TP、FP、FN、Precision、Recall、F1、FDR、MCC、Hamming 等
- 估计误差：L1/L2/Linf、MSE
- 预测误差：PE

### 3.5 src/experiments

comparison_common.py 提供通用实验流水线函数：

- generate_data
- evaluate_model_once
- monte_carlo_evaluation
- run_parameter_test
- plot_comparison

---

## 4. 三个核心方法详解

本节重点解释：

1. CoCoLasso
2. ACoCoLasso
3. ShapACoCoLasso（当前为 test 中的策略实现）

### 4.1 CoCoLasso

实现位置：src/models/eiv/canonical/CoCoLasso.py

#### 4.1.1 标准化与误差协方差缩放

代码中先对特征标准化、对响应中心化：

$$
\widetilde W=(W-\mu_W)D^{-1},
\qquad
\widetilde y=y-\bar y,
$$

其中 $D=\mathrm{diag}(s_1,\dots,s_p)$。

测量误差协方差同步缩放：

$$
\widetilde\Sigma_{uu}=D^{-1}\Sigma_{uu}D^{-1}
$$

（代码中等价为逐元素除以 $s_is_j$）。

#### 4.1.2 CoCo 核心统计量

$$
\widehat\Sigma = \frac{1}{n}\widetilde W^\top\widetilde W - \widetilde\Sigma_{uu},
\qquad
\widehat\rho = \frac{1}{n}\widetilde W^\top\widetilde y
$$

#### 4.1.3 PSD 投影（ADMM）

由于 $\widehat\Sigma$ 可能非半正定，代码通过 ADMM 做投影：

$$
\widetilde\Sigma = \Pi_{\mathbb S_+}(\widehat\Sigma)
$$

并记录：

- ADMM 迭代次数
- 收敛标记
- 步长历史

随后还会做：

- 对称化
- 最小特征值抬升
- Cholesky 失败时微扰重试

#### 4.1.4 转化为标准 Lasso 子问题

原目标可写为：

$$
\hat\beta_s
= \arg\min_{\beta}
\left(
\frac{1}{2}\beta^\top\widetilde\Sigma\beta
-\widehat\rho^\top\beta
+\alpha\|\beta\|_1
\right)
$$

令 $\widetilde\Sigma=LL^\top$，构造：

$$
\widetilde W_{lasso}=\sqrt n\,L^\top,
\qquad
\widetilde y_{lasso}=\sqrt n\,L^{-1}\widehat\rho
$$

再调用 sklearn Lasso 求解 $\hat\beta_s$。

#### 4.1.5 反标准化

$$
\hat\beta_j=\hat\beta_{s,j}/s_j,
\qquad
\hat b=\bar y-\mu_W^\top\hat\beta
$$

---

### 4.2 ACoCoLasso

实现位置：src/models/eiv/adaptive/ACoCoLasso.py

ACoCoLasso 是“给定初值的自适应 CoCo 求解器”，不在类内部生成初始估计。

#### 4.2.1 初始估计与权重

给定外部初值 $\hat\beta^{(0)}$（原尺度），先转标准化尺度：

$$
\hat\beta^{(0)}_{s,j}=\hat\beta^{(0)}_j\cdot s_j
$$

权重定义：

$$
\omega_j=\frac{1}{\left(|\hat\beta^{(0)}_{s,j}|+\varepsilon\right)^\gamma}
$$

代码默认 $\varepsilon=10^{-8}$。

#### 4.2.2 重参数化

设

$$
D_\omega=\mathrm{diag}(1/\omega_1,\dots,1/\omega_p)
$$

并令

$$
\beta_s=D_\omega\alpha
$$

则目标转为：

$$
\min_\alpha
\left(
\frac12\alpha^\top(D_\omega^\top\widetilde\Sigma D_\omega)\alpha
-(D_\omega^\top\widehat\rho)^\top\alpha
+\alpha_{reg}\|\alpha\|_1
\right)
$$

这对应代码中的：

- Sigma_transformed = D_omega^T Sigma_tilde D_omega
- rho_transformed = D_omega^T rho_hat

求解 $\alpha$ 后回代 $\beta_s$，再反标准化得到最终系数。

#### 4.2.3 test 中 ACoCoLasso 的初值来源

在复现实验脚本里，ACoCoLasso 的常见初值是 CoCoLasso 一阶段系数（先 CV 选一阶段 alpha）：

$$
\hat\beta^{(0)} \leftarrow \hat\beta^{CoCo}
$$

---

### 4.3 ShapACoCoLasso（策略版）

实现位置：test/ShapACoCoLasso_reproduction.py

当前仓库里，ShapACoCoLasso 不是 src 下独立模型类，而是一个方法标签：

- 求解器仍然调用 ACoCoLasso
- 区别在于 init_coef 的构造引入 SHAP

#### 4.3.1 计算流程

1) 先拟合 CoCoLasso（并 CV 选一阶段 alpha）

$$
\hat\beta^{coco} \leftarrow \text{CoCoLasso}(\alpha_{init})
$$

2) 用 shap.LinearExplainer 对该模型计算 SHAP 值 $\phi_{ij}$

3) 计算全局重要性

$$
I_j=\frac{1}{n}\sum_{i=1}^n |\phi_{ij}|
$$

4) 映射到 ACoCoLasso 可接受的 init_coef

$$
\hat\beta^{(0),shap}_j=\frac{I_j}{s_j}
$$

其中 $s_j=\mathrm{std}(W_j)$。

5) 传入 ACoCoLasso

$$
\text{ACoCoLasso}(\text{init\_coef}=\hat\beta^{(0),shap})
$$

#### 4.3.2 该映射的数学意义

由于 ACoCoLasso 内部按 $|\text{init\_coef}_j\cdot s_j|$ 构造权重，代入后得到：

$$
|\hat\beta^{(0),shap}_j\cdot s_j|=I_j
$$

因此最终权重为：

$$
\omega_j=\frac{1}{(I_j+\varepsilon)^\gamma}
$$

即“直接由 SHAP 重要性控制惩罚强度”。

---

## 5. test 模拟设置（按当前脚本）

### 5.1 脚本列表

- comparison_baseline.py
- ACoCoLasso_reproduction.py
- ShapACoCoLasso_reproduction.py
- CoCoElasticNet.py
- RFACoCoLasso.py
- XGBoostACoCoLasso.py

### 5.2 五个复现实验脚本的统一设置

以下设置在 ACoCoLasso_reproduction.py、ShapACoCoLasso_reproduction.py、CoCoElasticNet.py、RFACoCoLasso.py、XGBoostACoCoLasso.py 中一致：

1. 协方差结构

- AR(0.5): $\Sigma_{x,ij}=0.5^{|i-j|}$
- Compound Symmetry: $0.3\mathbf{1}\mathbf{1}^\top+0.7I$

2. 数据生成

$$
X\sim N(0,\Sigma_X),
\quad
y=X\beta^*+\epsilon,
\quad
W=X+A,
\quad
A\sim N(0,\tau^2I)
$$

并取

$$
\Sigma_{uu}=\tau^2I
$$

3. CV 与阈值

- cv_folds = 10
- alpha_grid = logspace(-2, 0.6, 10)
- selection_threshold = 1e-6

4. Monte Carlo

- 每个设定重复 n_simulations = 100

5. 两组实验

- high_dim: n=100, p=250, sigma=3.0, taus=[0.75,1.25], 结构=AR+CS
- ultra_high_dim: n=80, p=1000, sigma=1.0, taus=[0.25,0.5], 结构=AR

6. 真系数（非零项）

- high_dim: [3.0, 1.5, 2.0]
- ultra_high_dim: [1.0, -0.5, 0.7, -1.2, -0.9, 0.3, 0.55]

7. CSV 指标

- C: TP
- IC: FP
- PE: 预测误差
- MSE: 系数误差

输出均为 mean (std)。

### 5.3 comparison_baseline.py 说明

comparison_baseline.py 使用 src/experiments/comparison_common.py 做参数扫描：

- alpha
- p
- n
- sigma_u

并输出 Recall/F1/FDR/Exact Selection/Hamming/MCC 曲线图和 pkl。

该脚本主要用于通用对比流程演示与参数敏感性分析；自适应模型的严格复现建议优先参考 reproduction 系列脚本。

---

## 6. 导入与接口

### 6.1 常用导入

```python
from src.models.base import OLS, Lasso, ALasso, ElasticNet
from src.models.eiv.canonical import COLS, CRidge, CLasso, CoCoLasso, CoCoElasticNet
from src.models.eiv.adaptive import ACLasso, ACoCoLasso
from src.evaluation import selection_accuracy
```

### 6.2 顶层导入

```python
from src import CoCoLasso, ACoCoLasso, selection_accuracy
```

### 6.3 统一接口

所有模型遵循 sklearn 风格：

- fit(X_or_W, y)
- predict(X_or_W)
- coef_
- intercept_

---

## 7. 运行方式（Windows + venv）

```bash
.\venv\Scripts\python.exe .\test\ACoCoLasso_reproduction.py
.\venv\Scripts\python.exe .\test\ShapACoCoLasso_reproduction.py
.\venv\Scripts\python.exe .\test\CoCoElasticNet.py
.\venv\Scripts\python.exe .\test\RFACoCoLasso.py
.\venv\Scripts\python.exe .\test\XGBoostACoCoLasso.py
```

如果在新环境运行（当前仓库无 requirements.txt），可按需安装：

```bash
pip install numpy scipy scikit-learn matplotlib shap xgboost
```

---

## 8. 数值稳定性与实现约定

src 中 EIV 模型普遍采用以下策略：

1. 标准化/中心化处理
2. 协方差维度显式校验
3. PSD 修正与特征值下界
4. Cholesky 失败微扰重试
5. 零方差特征安全处理

这些机制在 CoCoLasso、ACoCoLasso、CoCoElasticNet、COLS、CRidge 等类中均有体现。

---

## 9. 方法关系图

```text
CoCoLasso
  -> ACoCoLasso (CoCo 初值 -> 自适应权重)
  -> ShapACoCoLasso (CoCo + SHAP -> init_coef -> ACoCoLasso)

ACoCoLasso
  -> RFACoCoLasso (随机森林重要性 -> init_coef)
  -> XGBoostACoCoLasso (XGBoost重要性 -> init_coef)
```

可以将 ACoCoLasso 理解为统一求解器，不同方法主要差异在权重来源。

---

## 10. 参考文献

1. Datta, A., and Zou, H. (2017). CoCoLasso for High-dimensional Error-in-variables Regression.
2. Zou, H. (2006). The Adaptive Lasso and Its Oracle Properties.
3. Loh, P. L., and Wainwright, M. J. (2012). High-dimensional Regression with Noisy and Missing Data.
4. 李锋, 盖宇杰, 卢一强 (2014). 测量误差模型的自适应 LASSO 变量选择方法研究.
