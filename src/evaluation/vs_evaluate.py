import numpy as np

def selection_accuracy(true_indices, selected_indices, total_features, beta_true=None, beta_hat=None, x_true=None):
    """
    评估变量选择的准确性（模拟实验中使用）
    
    参数:
    true_indices: 真实重要变量的索引 (Iterable)
    selected_indices: 方法选中的变量索引 (Iterable)
    total_features: 总特征数 p (int)
    beta_true: 真实的系数向量 (可选，用于计算 L2 Error，长度需为 p)
    beta_hat: 估计的系数向量 (可选，用于计算 L2 Error，长度需为 p)
    x_true: 真实无误差设计矩阵 X (可选，二维数组，形状为 (n_samples, p)，用于计算 PE)
    
    返回 (包含以下评估指标的字典):
    - TP, FP, FN, TN: 基础混淆矩阵指标（真正例，假正例，假反例，真反例）
    - Model_Size: 模型选出的变量总数 (等于 TP + FP)
    - Precision (查准率): 选中变量中真正重要的比例
    - Recall (查全率): 重要变量中被选中的比例 (核心指标)
    - F1: 综合 Precision 和 Recall 的调和平均数
    - FDR (错误发现率): 选中变量中噪声变量的比例 (核心指标)
    - EXACT_Selection_Rate (精确选择率): 完全正确选中所有重要变量且不包含任何噪声变量的概率 (核心指标)
    - Specificity (特异度): 噪声变量被正确排除的比例
    - Hamming_Distance (汉明距离): 选择向量与真实向量的差异比例
    - Accuracy (准确率): 总体分类准确率 (高维场景下由于大面积排除噪音变导致该指标缺乏区分度)
    - MCC (马修斯相关系数): 处理极度不平衡数据（高维稀疏）时更稳健的综合分类指标，范围[-1, 1]
    - L1_Error: 绝对估计误差和 (仅在提供 beta_true 和 beta_hat 时非 None)
    - L2_Error: 估计量与真实系数量级上的 L2 误差（仅在提供 beta_true 和 beta_hat 时非 None）
    - Linf_Error: 最大系数偏差 (Max Deviation, 仅在提供 beta_true 和 beta_hat 时非 None)
    - MSE: 系数估计的 L2 范数平方误差 ||beta - beta_hat||_2^2（仅在提供 beta_true 和 beta_hat 时非 None）
    - PE: 预测误差 (beta - beta_hat)^T Sigma_X (beta - beta_hat)，其中 Sigma_X = X^T X / n（仅在同时提供 beta_true、beta_hat、x_true 时非 None）
    """
    # 转换为集合
    true_set = set(true_indices)
    selected_set = set(selected_indices)
    all_features = set(range(total_features))
    
    # 真正例：正确选中的变量
    TP = len(true_set & selected_set)
    # 假正例：错误选中的噪声变量
    FP = len(selected_set - true_set)
    # 假反例：遗漏的真实变量
    FN = len(true_set - selected_set)
    # 真反例：正确排除的噪声变量
    TN = len((all_features - true_set) & (all_features - selected_set))
    
    ## 防止 total_features 为 0 导致后续分母为 0
    if total_features <= 0:
        raise ValueError("total_features must be a positive integer")
    
    # 指标计算 (按返回字典顺序排列)
    model_size = len(selected_set) # 模型选出的变量总数
    accuracy = (TP + TN) / total_features # 总体分类准确率
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0 # 查准率：选中变量中真正重要的比例
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0 # 查全率：重要变量中被选中的比例
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0 # F1 Score：查准率与查全率的调和平均数
    fdr = FP / (TP + FP) if (TP + FP) > 0 else 0 # 错误发现率 (FDR)：选中变量中噪声变量的比例
    exact_selection = 1 if selected_set == true_set else 0 # 精确选择率：只有完全一致才记为 1
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0 # 特异度：噪声变量被正确排除的比例
    
    # MCC 计算：处理除零异常
    mcc_numerator = (TP * TN) - (FP * FN)
    mcc_denominator = np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0.0

    # 汉明距离（选择向量的差异）
    true_vector = np.zeros(total_features)
    true_vector[list(true_indices)] = 1
    selected_vector = np.zeros(total_features)
    selected_vector[list(selected_indices)] = 1
    hamming = np.sum(true_vector != selected_vector) / total_features
    
    # 估计误差计算：仅在同时提供了 beta_true 和 beta_hat 时非 None
    l1_error = None
    l2_error = None 
    linf_error = None
    mse = None
    pe = None
    if beta_true is not None and beta_hat is not None:
        beta_true_arr = np.asarray(beta_true)
        beta_hat_arr = np.asarray(beta_hat)
        if len(beta_true_arr) != total_features or len(beta_hat_arr) != total_features:
            raise ValueError("beta_true and beta_hat must have length equal to total_features")
        
        diff = beta_hat_arr - beta_true_arr
        l1_error = float(np.linalg.norm(diff, ord=1)) # L1 Error：系数绝对误差和（衡量整体收缩偏差）
        l2_error = float(np.linalg.norm(diff))        # L2 Error：欧几里得距离误差（最为常用的衡量量级偏差的指标）
        linf_error = float(np.linalg.norm(diff, ord=np.inf)) # L无穷 Error：单一系数的最大绝对误差（最坏情况的估计偏差）
        mse = float(np.dot(diff, diff))               # MSE：按定义计算 ||beta - beta_hat||_2^2

        if x_true is not None:
            x_true_arr = np.asarray(x_true)
            if x_true_arr.ndim != 2:
                raise ValueError("x_true must be a 2D array with shape (n_samples, total_features)")
            if x_true_arr.shape[1] != total_features:
                raise ValueError("x_true second dimension must equal total_features")
            if x_true_arr.shape[0] <= 0:
                raise ValueError("x_true must contain at least one sample")

            sigma_x = (x_true_arr.T @ x_true_arr) / x_true_arr.shape[0]
            pe = float(diff.T @ sigma_x @ diff)

    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'Model_Size': model_size,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'FDR': fdr,
        'Exact_Selection_Rate': exact_selection,
        'Specificity': specificity,
        'MCC': mcc,
        'Hamming_Distance': hamming,
        'L1_Error': l1_error,
        'L2_Error': l2_error,
        'Linf_Error': linf_error,
        'MSE': mse,
        'PE': pe
    }