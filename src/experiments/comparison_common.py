import numpy as np
import matplotlib.pyplot as plt

from src.evaluation import selection_accuracy


_RESULT_KEYS = [
    'precision', 'recall', 'f1', 'fdr', 'exact_selection_rate', 'mcc',
    'specificity', 'hamming', 'accuracy', 'mse_beta', 'selected_count'
]

_PLOT_METRICS = [
    ('recall', 'Recall'),
    ('f1', 'F1 Score'),
    ('fdr', 'FDR'),
    ('exact_selection_rate', 'EXACT_Selection_Rate'),
    ('hamming', 'Hamming Distance'),
    ('mcc', 'MCC'),
]


def generate_data(n=100, p=200, s=5, sigma=1.0, sigma_u=0.5, seed=None):
    """生成带测量误差的高维线性回归数据。"""
    if seed is not None:
        np.random.seed(seed)

    x_true = np.random.randn(n, p)
    beta_true = np.zeros(p)
    true_indices = list(range(s))
    beta_true[true_indices] = np.random.randn(s) * 2 + 2

    eps = np.random.randn(n) * sigma
    y = x_true @ beta_true + eps

    u = np.random.randn(n, p) * sigma_u
    w = x_true + u
    sigma_uu = np.eye(p) * sigma_u ** 2

    return w, y, true_indices, beta_true, sigma_uu


def evaluate_model_once(model, w, y, true_indices, beta_true, p, selection_threshold=1e-6):
    """评估单个模型一次。"""
    try:
        model.fit(w, y)
        selected_indices = list(np.where(np.abs(model.coef_) > selection_threshold)[0])
        metrics = selection_accuracy(true_indices, selected_indices, p)
        mse_beta = np.mean((model.coef_ - beta_true) ** 2)

        return {
            'success': True,
            'precision': metrics['Precision'],
            'recall': metrics['Recall'],
            'f1': metrics['F1'],
            'fdr': metrics['FDR'],
            'exact_selection_rate': metrics['Exact_Selection_Rate'],
            'mcc': metrics['MCC'],
            'specificity': metrics['Specificity'],
            'hamming': metrics['Hamming_Distance'],
            'accuracy': metrics['Accuracy'],
            'mse_beta': mse_beta,
            'selected_count': len(selected_indices),
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def _build_empty_results(model_names):
    return {
        name: {
            'precision': [],
            'recall': [],
            'f1': [],
            'fdr': [],
            'exact_selection_rate': [],
            'mcc': [],
            'specificity': [],
            'hamming': [],
            'accuracy': [],
            'mse_beta': [],
            'selected_count': [],
            'success_count': 0,
        }
        for name in model_names
    }


def monte_carlo_evaluation(
    n_simulations,
    n,
    p,
    s,
    alpha,
    model_builders,
    sigma=1.0,
    sigma_u=0.5,
    selection_threshold=1e-6,
):
    """通用蒙特卡洛评估：模型实例由 model_builders 提供。"""
    model_names = list(model_builders.keys())
    all_results = _build_empty_results(model_names)

    for i in range(n_simulations):
        w, y, true_indices, beta_true, sigma_uu = generate_data(
            n=n, p=p, s=s, sigma=sigma, sigma_u=sigma_u, seed=i
        )

        for name, builder in model_builders.items():
            model = builder(alpha, sigma_uu)
            result = evaluate_model_once(
                model,
                w,
                y,
                true_indices,
                beta_true,
                p,
                selection_threshold=selection_threshold,
            )
            if result['success']:
                for key in _RESULT_KEYS:
                    all_results[name][key].append(result[key])
                all_results[name]['success_count'] += 1

    avg_results = {}
    for name in model_names:
        count = all_results[name]['success_count']
        if count > 0:
            avg_results[name] = {
                'precision': np.mean(all_results[name]['precision']),
                'precision_std': np.std(all_results[name]['precision']),
                'recall': np.mean(all_results[name]['recall']),
                'recall_std': np.std(all_results[name]['recall']),
                'f1': np.mean(all_results[name]['f1']),
                'f1_std': np.std(all_results[name]['f1']),
                'fdr': np.mean(all_results[name]['fdr']),
                'fdr_std': np.std(all_results[name]['fdr']),
                'exact_selection_rate': np.mean(all_results[name]['exact_selection_rate']),
                'exact_selection_rate_std': np.std(all_results[name]['exact_selection_rate']),
                'mcc': np.mean(all_results[name]['mcc']),
                'mcc_std': np.std(all_results[name]['mcc']),
                'specificity': np.mean(all_results[name]['specificity']),
                'specificity_std': np.std(all_results[name]['specificity']),
                'hamming': np.mean(all_results[name]['hamming']),
                'hamming_std': np.std(all_results[name]['hamming']),
                'accuracy': np.mean(all_results[name]['accuracy']),
                'accuracy_std': np.std(all_results[name]['accuracy']),
                'mse_beta': np.mean(all_results[name]['mse_beta']),
                'mse_beta_std': np.std(all_results[name]['mse_beta']),
                'selected_count': np.mean(all_results[name]['selected_count']),
                'selected_count_std': np.std(all_results[name]['selected_count']),
                'success_rate': count / n_simulations,
            }
        else:
            avg_results[name] = {'success_rate': 0.0}

    return avg_results


def run_parameter_test(
    test_name,
    param_name,
    param_values,
    fixed_params,
    n_simulations,
    model_builders,
    success_rate_threshold=0.5,
    selection_threshold=1e-6,
):
    """运行单个参数变化测试（通用）。"""
    print("\n" + "=" * 80)
    print("测试: {}".format(test_name))
    print("=" * 80)

    model_names = list(model_builders.keys())
    tracked_keys = [
        'f1', 'precision', 'recall', 'fdr', 'exact_selection_rate',
        'mcc', 'specificity', 'hamming', 'accuracy', 'mse_beta'
    ]
    results = {name: {key: [] for key in tracked_keys} for name in model_names}

    for i, val in enumerate(param_values):
        print("\n[{}/{}] {} = {}".format(i + 1, len(param_values), param_name, val))

        params = fixed_params.copy()
        params[param_name] = val

        avg_res = monte_carlo_evaluation(
            n_simulations=n_simulations,
            model_builders=model_builders,
            selection_threshold=selection_threshold,
            **params
        )

        for name in model_names:
            if avg_res[name].get('success_rate', 0) > success_rate_threshold:
                for key in tracked_keys:
                    results[name][key].append(avg_res[name][key])
            else:
                for key in tracked_keys:
                    results[name][key].append(np.nan)

    return param_values, results


def plot_comparison(
    x_values,
    results,
    model_names,
    xlabel,
    title,
    save_path,
    colors,
    marker_map,
    metrics=None,
):
    """绘制对比图。"""
    plot_metrics = _PLOT_METRICS if metrics is None else metrics

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, (metric, metric_name) in enumerate(plot_metrics):
        ax = axes[i // 3, i % 3]
        for j, name in enumerate(model_names):
            ax.plot(
                x_values,
                results[name][metric],
                label=name,
                color=colors[j],
                linewidth=2,
                markersize=6,
                linestyle='-',
                marker=marker_map[name],
            )

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title('{} vs {}'.format(metric_name, title), fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        if 'alpha' in xlabel.lower():
            ax.set_xscale('log')
        if metric == 'hamming':
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print('图已保存到: {}'.format(save_path))
    plt.close()
