# RandomForest Corrected Lasso 在测量误差变量选择中的研究

## 摘要
本文研究项目创新方法 RandomForest Corrected Lasso。该方法以随机森林特征重要性替代传统“初始系数驱动”的自适应权重，在测量误差校正 Lasso 框架中引入非线性结构先验。基于 test/comparison_with_randomforest.py 的统一实验设置和 results_20260328_1 的结果文件，本文系统分析其方法机制、参数行为与性能特征。结果表明：该方法在默认设置下保持较高召回，并在 alpha 调优后可显著提升 F1 与误报控制，但在高维 p 扩张下仍存在性能下降。

## 1. 研究背景
在 EIV 高维模型中，传统修正 Lasso 的难点有二：
1. 测量误差造成 Gram 估计偏差，直接套用 Lasso 会引起系统性偏差。
2. 统一惩罚强度难以兼顾强弱信号，容易出现过惩罚或误报。

RandomForest Corrected Lasso 的思路是：先由随机森林学习特征相对重要性，再将其转化为惩罚权重，形成“重要特征轻惩罚、次要特征重惩罚”的非均匀收缩结构。

## 2. 方法与实现

### 2.1 权重构造
设随机森林重要性为 $FI_j$，在 weight_method='max_scaled' 下：

$$
w_j = \frac{1}{\left(FI_j / \max_k FI_k + \epsilon\right)^\gamma}
$$

其中 $\gamma=1.0$（脚本默认）。重要性越大，权重越小，对应惩罚越弱。

### 2.2 校正优化流程
方法实现位于 src/models/eiv/feature_weighted/RandomForest_Corrected_Lasso.py，主要步骤：
1. 标准化 W 并中心化 y。
2. 用随机森林拟合 $(W,y)$，提取 feature_importances_。
3. 用权重重标度设计矩阵，构造误差校正矩阵：

$$
\Sigma_{corr}=\frac{1}{n}W_w^TW_w - D_w\Sigma_{uu}D_w
$$

4. 对 $\Sigma_{corr}$ 做对称化、特征值抬升及 Cholesky 稳定化。
5. 转换为等价 Lasso 子问题，求解后映射回原尺度。

### 2.3 工程稳定性
代码包含以下稳定化机制：
1. Sigma_uu 维度检查。
2. 零方差特征安全缩放。
3. 最小特征值截断与 Cholesky fallback。
4. 权重裁剪（normalized 模式下）与异常方法名保护。

## 3. 实验设置（按脚本复现）
实验脚本：test/comparison_with_randomforest.py。

### 3.1 参数设置
1. Monte Carlo：每参数点 100 次。
2. 默认场景：n=100, p=10, s=5, alpha=0.1, sigma=1.0, sigma_u=0.5。
3. 扫描范围：
   - alpha: 0.01 到 1.0（20 点）
   - p: 10 到 300（20 点）
   - n: 40 到 1000（20 点）
   - sigma_u: 0.1 到 1.0（20 点）

### 3.2 随机森林配置
n_estimators=100, max_depth=8, max_features='sqrt', min_samples_split=4, min_samples_leaf=2, bootstrap=True, random_state=42, n_jobs=-1。

### 3.3 对比方法
Naive Lasso、Corrected Lasso、CoCoLasso、Adaptive Corrected Lasso、RandomForest Corrected Lasso。

## 4. 结果分析
结果来源：results/results_20260328_1/all_results_rf_20260328_153831.pkl。

### 4.1 默认场景
| 指标 | RandomForest Corrected Lasso |
|---|---:|
| Recall | 0.9800 |
| F1 | 0.6786 |
| FDR | 0.4798 |
| MCC | 0.1004 |
| Hamming Distance | 0.4660 |
| Selected Count | 9.46 |
| Exact Selection Rate | 0.0000 |

与同脚本 F1 最优方法对比（Naive Lasso）
- RF-Corrected: F1=0.6786
- Naive Lasso: F1=0.7220
- 差值: -0.0434

默认点并非该方法最佳工况。

### 4.2 灵敏度结论
1. alpha 扫描收益显著：
   - 最佳 F1=0.7880（alpha=1.0）
   - 对应 Recall=0.8420, FDR=0.2306, Hamming=0.2240
   - 说明加大正则强度可有效抑制误报。
2. p 扫描：
   - 最佳在 p=10（F1=0.6786），较高维时下降明显。
   - 最差出现在 p=71（F1=0.3130），FDR 接近 0.80。
3. n 扫描：
   - 区间内较稳，F1 在 0.665 到 0.689。
4. sigma_u 扫描：
   - 低测量误差时更优，sigma_u=0.1 时 F1=0.7082；
   - 中高误差水平下 F1 回落到约 0.667 附近。

## 5. 讨论
### 5.1 创新价值
1. 使用非线性模型重要性构造权重，可引入超越线性初值的结构信息。
2. 与 EIV 校正耦合后，提供了“树模型先验 + 线性可解释”混合范式。

### 5.2 风险与局限
1. 重要性估计受随机森林超参数影响，权重存在模型依赖性。
2. 在高 p 场景下，重要性稀释与噪声特征竞争会削弱加权效果。
3. 默认 alpha=0.1 偏弱，可能导致误报偏高。

### 5.3 后续优化建议
1. 将 alpha 与树模型参数进行联合搜索，而非固定树参数。
2. 在权重中加入温度或分位数截断，缓解极端重要性带来的不稳定缩放。
3. 增加多次随机森林重采样平均权重，降低单次拟合波动。

## 6. 结论
RandomForest Corrected Lasso 体现了本项目在“统计校正 + 机器学习先验”方向的创新探索。其在适当正则强度下可取得较好的 F1 与误报平衡，但面对高维扩张与测量误差增大仍需更强的参数自适应机制。该方法适合作为 EIV 变量选择的实用增强基线，并为后续集成权重策略提供了可扩展框架。

## 参考文献
1. CoCoLasso for High-dimensional Error-in-variables Regression（References 文件夹）.
2. Measurement error in LASSO Impact and likelihood bias correction（References 文件夹）.
3. The Adaptive Lasso and Its Oracle Properties（References 文件夹）.
4. 测量误差模型的自适应LASSO变量选择方法研究_李锋（References 文件夹）.
