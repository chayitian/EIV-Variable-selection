# XGBoost Corrected Lasso 在高维 EIV 变量选择中的研究

## 摘要
本文聚焦项目创新方法 XGBoost Corrected Lasso。该方法通过 XGBoost 特征重要性构建惩罚权重，并嵌入测量误差修正 Lasso 的优化流程，形成“梯度提升先验 + 统计校正”组合。基于 test/comparison_with_xgboost.py 与 results_20260328_1 的实证数据，本文对其性能进行系统分析。结果显示：该方法在默认设置下具备较高召回，但 F1 与误报控制未优于 Naive Lasso；在 alpha 调优与低测量误差条件下可获得明显提升，但对高维 p 增长的稳定性不足。

## 1. 研究动机
在测量误差存在时，线性模型系数初值常不稳定。与其使用线性初值生成权重，XGBoost Corrected Lasso 试图借助提升树对非线性结构和交互效应的捕捉能力，提取更具辨识度的特征优先级，再用于 EIV 场景下的加权惩罚。

## 2. 方法框架

### 2.1 权重定义
在脚本配置 weight_method='max_scaled' 下，利用 XGBoost 重要性 $FI_j$ 定义：

$$
w_j = \frac{1}{\left(FI_j / \max_k FI_k + \epsilon\right)^\gamma},\qquad \gamma=1.0
$$

### 2.2 优化流程
实现位于 src/models/eiv/feature_weighted/XGBoost_Corrected_Lasso.py：
1. 标准化与中心化处理。
2. 训练 XGBRegressor，提取 feature_importances_。
3. 权重化后构建校正矩阵并做 PSD 稳定化。
4. 通过 Cholesky 变换，将目标转为标准 Lasso 子问题。
5. 反变换得到原尺度系数与截距。

### 2.3 XGBoost 参数（实验脚本固定）
n_estimators=150, max_depth=6, learning_rate=0.05, importance_type='gain', subsample=0.8, colsample_bytree=0.8, min_child_weight=1.0, xgb_gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, objective='reg:squarederror', random_state=42, n_jobs=1, verbosity=0。

## 3. 实验设计
实验脚本：test/comparison_with_xgboost.py。

### 3.1 数据与扫描设置
1. Monte Carlo: 每参数点 100 次。
2. 默认场景：n=100, p=10, s=5, alpha=0.1, sigma=1.0, sigma_u=0.5。
3. 四组敏感性：alpha、p、n、sigma_u（均 20 点）。

### 3.2 对比模型
Naive Lasso、Corrected Lasso、CoCoLasso、Adaptive Corrected Lasso、XGBoost Corrected Lasso。

## 4. 实证结果
结果来源：results/results_20260328_1/all_results_xgb_20260328_160701.pkl。

### 4.1 默认场景表现
| 指标 | XGBoost Corrected Lasso |
|---|---:|
| Recall | 0.9700 |
| F1 | 0.6707 |
| FDR | 0.4865 |
| MCC | 0.0683 |
| Hamming Distance | 0.4770 |
| Selected Count | 9.47 |
| Exact Selection Rate | 0.0000 |

与同脚本 F1 最优方法对比（Naive Lasso）：
- XGB-Corrected: F1=0.6707
- Naive Lasso: F1=0.7220
- 差值: -0.0513

### 4.2 参数敏感性解读
1. alpha 扫描：
   - 最佳 F1=0.7364（alpha=1.0）
   - 对应 Recall=0.8140, FDR=0.2884, Hamming=0.2860
   - 表明较强正则可显著削减误报。
2. p 扫描：
   - 在 p=10 时为最佳（F1=0.6707），
   - 在 p=223 时出现最差（F1=0.0000，Recall=0），表现出高维脆弱性。
3. n 扫描：
   - 总体较稳，F1 约 0.660 到 0.678。
4. sigma_u 扫描：
   - 低测量误差下更优（sigma_u=0.1 时 F1=0.6951），
   - 中高误差区间回落到约 0.663 附近。

## 5. 讨论
### 5.1 方法优势
1. 借助 boosting 的重要性估计，能在部分设置中改善 FDR 与 Hamming。
2. 在 alpha 充分调节时可获得比默认设置更好的稀疏恢复平衡。

### 5.2 方法局限
1. 对特征维度增长较敏感，存在“重要性塌缩后过度收缩”风险。
2. 权重质量依赖树模型训练质量与超参数；可迁移性有限。
3. 默认参数并非最优，导致主流程结果低估方法潜力。

### 5.3 可能改进
1. 针对 p 扩张场景，采用分层特征筛选后再进行 XGB 权重构造。
2. 引入权重下界与平滑策略，避免重要性接近 0 时惩罚过强。
3. 对重要性采用多次重采样平均，降低单次训练波动。

## 6. 结论
XGBoost Corrected Lasso 体现了本项目将非线性学习器与 EIV 统计校正融合的创新路径。其在参数调优后能够改善误报控制，但对高维扩张的鲁棒性仍需增强。该方法在“可解释线性目标 + 非线性先验注入”方向具有研究价值，适合继续沿稳健权重估计与自动调参方向深化。

## 参考文献
1. CoCoLasso for High-dimensional Error-in-variables Regression（References 文件夹）.
2. Linear and Conic Programming Estimators in High Dimensional Errors-in-variables Models（References 文件夹）.
3. Measurement error in LASSO Impact and likelihood bias correction（References 文件夹）.
4. 测量误差模型的自适应LASSO变量选择方法研究_李锋（References 文件夹）.
