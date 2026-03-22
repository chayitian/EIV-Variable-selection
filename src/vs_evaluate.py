import numpy as np

def selection_accuracy(true_indices, selected_indices, total_features):
    """
    评估变量选择的准确性（模拟实验中使用）
    
    true_indices: 真实重要变量的索引
    selected_indices: 方法选中的变量索引
    total_features: 总特征数 p
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
    
    # 指标计算
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # 查准率
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0      # 查全率
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 汉明距离（选择向量的差异）
    true_vector = np.zeros(total_features)
    true_vector[list(true_indices)] = 1
    selected_vector = np.zeros(total_features)
    selected_vector[list(selected_indices)] = 1
    
    hamming = np.sum(true_vector != selected_vector) / total_features
    
    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'Precision': precision,      # 选中变量中真正重要的比例
        'Recall': recall,            # 重要变量中被选中的比例
        'F1': f1,                    # 综合精确率和召回率
        'Specificity': TN / (TN + FP),  # 噪声变量被正确排除的比例
        'Hamming_Distance': hamming,    # 选择向量的差异比例
        'Accuracy': (TP + TN) / total_features  # 总体选择准确率
    }