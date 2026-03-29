# EIV-Variable-selection

高维测量误差变量选择研究与实验框架。

本仓库聚焦两类回归场景：

1. 无测量误差的标准线性回归（用于方法基线与理论对照）
2. 带加性测量误差的高维线性回归（项目主问题）

项目已将基础模型、EIV 经典修正模型、自适应改进模型与树模型增强模型统一到一个分层代码框架中，并提供三套可直接运行的对比实验脚本。

---

## 1. 问题设定

### 1.1 标准线性回归

$$
y = X\beta^* + \varepsilon
$$

其中 $y \in \mathbb{R}^n$，$X \in \mathbb{R}^{n \times p}$，$\beta^* \in \mathbb{R}^p$。

### 1.2 带测量误差的 EIV 回归

$$
\begin{cases}
y = X\beta^* + \varepsilon \\
W = X + U
\end{cases}
$$

其中 $W$ 为可观测协变量，$X$ 不可观测，$U$ 为测量误差；通常假设已知或可估计误差协方差矩阵 $\Sigma_{uu}$。

---

## 2. 方法演进主线

### 2.1 标准模型主线

```text
OLS
	-> Lasso
	-> Adaptive Lasso
```

### 2.2 EIV 模型主线

```text
Naive Lasso
	-> Corrected OLS / Corrected Ridge（初始化与校正基础）
	-> Corrected Lasso
	-> CoCoLasso
	-> Adaptive Corrected Lasso / Adaptive CoCoLasso
	-> RandomForest Corrected Lasso / XGBoost Corrected Lasso
```

---

## 3. 当前代码结构（已统一）

```text
src/
	__init__.py
	evaluation/
		__init__.py
		vs_evaluate.py
	experiments/
		__init__.py
		comparison_common.py
		results_flatten.py
	models/
		__init__.py
		base/
			__init__.py
			OLS.py
			Lasso.py
			Adaptive_Lasso.py
		eiv/
			__init__.py
			canonical/
				__init__.py
				Corrected_OLS.py
				Corrected_Ridge.py
				Corrected_Lasso.py
				CoCoLasso.py
			adaptive/
				__init__.py
				Adaptive_Corrected_Lasso.py
				Adaptive_CoCoLasso.py
			feature_weighted/
				__init__.py
				RandomForest_Corrected_Lasso.py
				XGBoost_Corrected_Lasso.py

test/
	comparison_baseline.py
	comparison_with_randomforest.py
	comparison_with_xgboost.py
```

说明：历史兼容层 prop 已移除，导入路径统一使用 src.models.* 与 src.evaluation。`results` 目录会随实验持续新增文件，以上为当前核验快照。

---

## 4. 模型清单与文件映射

### 4.1 基础模型（无测量误差）

| 模型 | 类名 | 文件 |
|---|---|---|
| OLS | OLS | src/models/base/OLS.py |
| Lasso | LassoRegression | src/models/base/Lasso.py |
| Adaptive Lasso | AdaptiveLasso | src/models/base/Adaptive_Lasso.py |

### 4.2 EIV 经典修正模型

| 模型 | 类名 | 文件 |
|---|---|---|
| Corrected OLS | CorrectedOLS | src/models/eiv/canonical/Corrected_OLS.py |
| Corrected Ridge | CorrectedRidge | src/models/eiv/canonical/Corrected_Ridge.py |
| Corrected Lasso | CorrectedLasso | src/models/eiv/canonical/Corrected_Lasso.py |
| CoCoLasso | CoCoLasso | src/models/eiv/canonical/CoCoLasso.py |

### 4.3 EIV 自适应与创新模型

| 模型 | 类名 | 文件 |
|---|---|---|
| Adaptive Corrected Lasso | AdaptiveCorrectedLasso | src/models/eiv/adaptive/Adaptive_Corrected_Lasso.py |
| Adaptive CoCoLasso | AdaptiveCoCoLasso | src/models/eiv/adaptive/Adaptive_CoCoLasso.py |
| RandomForest Corrected Lasso | RandomForestCorrectedLasso | src/models/eiv/feature_weighted/RandomForest_Corrected_Lasso.py |
| XGBoost Corrected Lasso | XGBoostCorrectedLasso | src/models/eiv/feature_weighted/XGBoost_Corrected_Lasso.py |

评估模块：

- selection_accuracy: src/evaluation/vs_evaluate.py

---

## 5. 关键目标函数（统一表述）

### 5.1 OLS

$$
\hat{\beta}_{\text{OLS}} = \arg\min_{\beta}\left(\frac{1}{2n}\|y - X\beta\|_2^2\right)
$$

### 5.2 Lasso

$$
\hat{\beta}_{\text{Lasso}} = \arg\min_{\beta}\left(\frac{1}{2n}\|y - X\beta\|_2^2 + \lambda\|\beta\|_1\right)
$$

### 5.3 Adaptive Lasso

$$
\hat{\beta}_{\text{Ada}} = \arg\min_{\beta}\left(\frac{1}{2n}\|y - X\beta\|_2^2 + \lambda\sum_{j=1}^p w_j|\beta_j|\right),
\quad
w_j = \frac{1}{(|\hat{\beta}^{\text{init}}_j| + \epsilon)^\gamma}
$$

### 5.4 Corrected Lasso（EIV）

$$
\hat{\beta}_{\text{Corrected}} = \arg\min_{\beta}\left(
\frac{1}{2n}\|y - W\beta\|_2^2 - \frac{1}{2}\beta^T\Sigma_{uu}\beta + \lambda\|\beta\|_1
\right)
$$

### 5.5 CoCoLasso（EIV）

先构造

$$
\widehat{\Sigma} = \frac{1}{n}W^TW - \Sigma_{uu},
\quad
\widetilde{\rho} = \frac{1}{n}W^Ty
$$

再投影到最近半正定矩阵 $\tilde{\Sigma}$，求解

$$
\hat{\beta}_{\text{CoCo}} =
\arg\min_{\beta}
\left(\frac{1}{2}\beta^T\tilde{\Sigma}\beta - \tilde{\rho}^T\beta + \lambda\|\beta\|_1\right)
$$

### 5.6 自适应修正类方法

Adaptive Corrected Lasso 与 Adaptive CoCoLasso 都采用“先初始估计，再自适应加权”的范式：

$$
w_j = \frac{1}{(|\hat{\beta}^{\text{init}}_j| + \epsilon)^\gamma}
$$

并通过重参数化把加权 L1 惩罚转化为标准 Lasso 子问题求解。

### 5.7 树模型加权修正类方法

RandomForest / XGBoost 版本用特征重要性替代系数初值构造权重：

$$
w_j = \frac{1}{(FI_j^{\text{norm}} + \epsilon)^\gamma}
$$

其中 $FI_j^{\text{norm}}$ 为归一化特征重要性。

---

## 6. 实验框架（以 test 为准）

### 6.1 三套主实验脚本

| 脚本 | 对比模型集合 |
|---|---|
| test/comparison_baseline.py | Naive Lasso, Corrected Lasso, CoCoLasso, Adaptive Corrected Lasso, Adaptive CoCoLasso |
| test/comparison_with_randomforest.py | Naive Lasso, Corrected Lasso, CoCoLasso, Adaptive Corrected Lasso, RandomForest Corrected Lasso |
| test/comparison_with_xgboost.py | Naive Lasso, Corrected Lasso, CoCoLasso, Adaptive Corrected Lasso, XGBoost Corrected Lasso |

### 6.2 每个脚本的统一流程

1. 生成带测量误差模拟数据（W, y, Sigma_uu）
2. 进行 Monte Carlo 重复实验（默认 n_simulations = 100，可通过命令行参数调整）
3. 对四组参数做灵敏度分析：
   - alpha（正则化强度）
   - p（特征数）
   - n（样本量）
   - sigma_u（测量误差强度）
4. 输出 6 个核心指标的对比图：Recall, F1, FDR, Exact Selection Rate, Hamming Distance, MCC
5. 保存 PNG 图与 PKL 汇总结果到 results 目录

脚本级参数约定：

- comparison_baseline.py
	- 包含 Adaptive CoCoLasso
	- 支持参数：--n_simulations, --selection_threshold
- comparison_with_randomforest.py
	- 使用 RandomForestCorrectedLasso
	- 支持参数：--n_simulations, --selection_threshold, --weight_method
	- --weight_method 可选值：normalized, max_scaled（默认 max_scaled）
- comparison_with_xgboost.py
	- 使用 XGBoostCorrectedLasso
	- 支持参数：--n_simulations, --selection_threshold, --weight_method
	- --weight_method 可选值：normalized, max_scaled（默认 max_scaled）

结果文件约定：

- 三个脚本保存的 PKL 均包含 config 字段，用于记录本次运行的关键参数（如 n_simulations、selection_threshold、weight_method）。

### 6.3 评估指标

selection_accuracy 支持并返回：

- TP, FP, FN, TN
- Precision, Recall, F1, FDR
- Exact_Selection_Rate, Specificity, MCC
- Hamming_Distance, Accuracy
- L1_Error, L2_Error, Linf_Error（当提供真值与估计系数时）

---

## 7. 统一导入规范

### 7.1 基础与 EIV 模型导入

```python
from src.models.base import OLS, LassoRegression, AdaptiveLasso
from src.models.eiv.canonical import CorrectedOLS, CorrectedRidge, CorrectedLasso, CoCoLasso
from src.models.eiv.adaptive import AdaptiveCorrectedLasso, AdaptiveCoCoLasso
from src.models.eiv.feature_weighted import RandomForestCorrectedLasso, XGBoostCorrectedLasso
from src.evaluation import selection_accuracy
```

### 7.2 顶层聚合导入（可选）

```python
from src import (
	OLS,
	LassoRegression,
	AdaptiveLasso,
	CorrectedOLS,
	CorrectedRidge,
	CorrectedLasso,
	CoCoLasso,
	AdaptiveCorrectedLasso,
	AdaptiveCoCoLasso,
	RandomForestCorrectedLasso,
	XGBoostCorrectedLasso,
	selection_accuracy,
	flatten_results_to_excel,
)
```

实验公共函数导入：

```python
from src import (
	generate_data,
	evaluate_model_once,
	monte_carlo_evaluation,
	run_parameter_test,
	plot_comparison,
	flatten_results_to_excel,
)
```

---

## 8. 快速开始

### 8.1 安装依赖

```bash
pip install -r requirements.txt
```

### 8.2 运行实验

```bash
python test/comparison_baseline.py
python test/comparison_with_randomforest.py
python test/comparison_with_xgboost.py
```

常见参数示例：

```bash
python test/comparison_baseline.py --n_simulations 50 --selection_threshold 1e-5
python test/comparison_with_randomforest.py --n_simulations 50 --selection_threshold 1e-5 --weight_method normalized
python test/comparison_with_xgboost.py --n_simulations 50 --selection_threshold 1e-5 --weight_method normalized
```

在当前仓库环境（Windows + venv）也可使用：

```bash
.\venv\Scripts\python.exe .\test\comparison_baseline.py
.\venv\Scripts\python.exe .\test\comparison_with_randomforest.py
.\venv\Scripts\python.exe .\test\comparison_with_xgboost.py
```

带参数示例（Windows + venv）：

```bash
.\venv\Scripts\python.exe .\test\comparison_baseline.py --n_simulations 50 --selection_threshold 1e-5
.\venv\Scripts\python.exe .\test\comparison_with_randomforest.py --n_simulations 50 --selection_threshold 1e-5 --weight_method normalized
.\venv\Scripts\python.exe .\test\comparison_with_xgboost.py --n_simulations 50 --selection_threshold 1e-5 --weight_method normalized
```

### 8.3 多次实验结果拍平与汇总导出

支持将 `results` 目录及其子目录中的 `all_results_*.pkl` 递归读取并汇总：

- 输出一个 xlsx 文件，包含 5 个 sheet：
	- `long_table`：全量长表
	- `alpha`、`p`、`n`、`sigma_u`：按参数维度拆分
- 可选同时导出一份长表 csv（csv 仅单表）

运行示例（默认输入目录为 `results`）：

```bash
python -m src.experiments.results_flatten
```

指定输入与输出：

```bash
python -m src.experiments.results_flatten --input_root results --output_xlsx results/flattened/all_runs_summary.xlsx --with_csv
```

Windows + venv 示例：

```bash
.\venv\Scripts\python.exe -m src.experiments.results_flatten --input_root results --output_xlsx results\flattened\all_runs_summary.xlsx --with_csv
```

参数说明：

- `--input_root`：输入目录（递归查找 pkl）
- `--pattern`：查找模式，默认 `all_results_*.pkl`
- `--output_xlsx`：输出 xlsx 路径（不指定则默认输出到 `results/flattened/`）
- `--with_csv`：同时导出长表 csv

---

## 9. 数值稳定性与实现约定

1. 各类 EIV 模型在拟合前执行标准化，拟合后将系数映射回原尺度。
2. 对涉及 Cholesky 分解的矩阵进行半正定修正，避免数值不可逆。
3. 对零方差特征执行安全处理，避免除零和不稳定系数。
4. 所有测试脚本支持固定随机种子，保证结果可复现。
5. 使用统一 sklearn 风格接口：fit / predict / coef_ / intercept_。

---

## 10. 方法对比（基础层）

| 方法 | 稀疏性 | 变量选择 | 神谕性质 | 适用场景 |
|---|---|---|---|---|
| OLS | 否 | 否 | - | 低维回归、基准模型 |
| Lasso | 是 | 是 | 否 | 高维变量选择 |
| Adaptive Lasso | 是 | 是 | 是 | 高维变量选择（理论最优） |

---

## 11. 统一接口

所有模型遵循 sklearn 风格：

```python
class ModelName:
		def __init__(self, ...):
				...

		def fit(self, X_or_W, y):
				...
				return self

		def predict(self, X_or_W):
				return X_or_W @ self.coef_ + self.intercept_
```

约定输出属性：

- coef_：系数向量
- intercept_：截距

---

## 12. 数学符号说明

| 符号 | 含义 |
|---|---|
| n | 样本数量 |
| p | 特征数量 |
| X | 无误差协变量矩阵 |
| W | 含测量误差协变量矩阵 |
| y | 响应变量向量 |
| beta | 系数向量 |
| lambda | 正则化参数 |
| gamma | 自适应权重指数 |
| Sigma_uu | 测量误差协方差矩阵 |
| FI_j | 第 j 个特征的重要性 |

---

## 13. 算法关系图

```text
Naive Lasso
	-> Corrected Lasso -> Adaptive Corrected Lasso
	-> CoCoLasso       -> Adaptive CoCoLasso

Adaptive Corrected Lasso / Adaptive CoCoLasso
	-> RandomForest Corrected Lasso（重要性加权）
	-> XGBoost Corrected Lasso（重要性加权）
```

---

## 14. 自适应权重与重参数化

四种代表性权重来源：

| 方法 | 权重来源 |
|---|---|
| Adaptive Corrected Lasso | 修正线性初值系数 |
| Adaptive CoCoLasso | 修正线性初值系数 |
| RandomForest Corrected Lasso | 随机森林特征重要性 |
| XGBoost Corrected Lasso | XGBoost 特征重要性 |

重参数化核心思想：

$$
\lambda\sum_{j=1}^p w_j|\beta_j|
\xrightarrow{\beta_j = \alpha_j \cdot w_j}
\lambda\sum_{j=1}^p |\alpha_j|
$$

这样可以把加权 L1 惩罚问题转化为标准 Lasso 子问题，再逆变换回原系数。

---

## 15. 参考文献

1. Zou, H. (2006). The adaptive lasso and its oracle properties. Journal of the American Statistical Association, 101(476), 1418-1429.
2. Datta, A., & Zou, H. (2017). CoCoLasso for High-dimensional Error-in-variables Regression. The Annals of Statistics, 45(6), 2400-2426.
3. Loh, P. L., & Wainwright, M. J. (2012). High-dimensional regression with noisy and missing data: Provable guarantees with non-convexity. The Annals of Statistics, 40(3), 1637-1664.
4. 李锋, 盖宇杰, 卢一强. (2014). 测量误差模型的自适应 LASSO 变量选择方法研究. 中国科学: 数学, 44(9), 983-1006.
