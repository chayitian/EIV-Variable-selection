import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.Lasso import LassoRegression
from src.Corrected_Lasso import CorrectedLasso
from src.CoCoLasso import CoCoLasso
from src.Adaptive_Corrected_Lasso import AdaptiveCorrectedLasso
from src.Adaptive_CoCoLasso import AdaptiveCoCoLasso
from src.RandomForest_Corrected_Lasso import RandomForestCorrectedLasso
from src.vs_evaluate import selection_accuracy


def generate_data(n=100, p=200, s=5, sigma=1.0, sigma_u=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)

    X_true = np.random.randn(n, p)
    beta_true = np.zeros(p)
    true_indices = list(range(s))
    beta_true[true_indices] = np.random.randn(s) * 2 + 2

    eps = np.random.randn(n) * sigma
    y = X_true @ beta_true + eps

    U = np.random.randn(n, p) * sigma_u
    W = X_true + U
    Sigma_uu = np.eye(p) * sigma_u ** 2

    return W, y, true_indices, beta_true, Sigma_uu


def evaluate_model_once(model, W, y, true_indices, beta_true, p):
    try:
        model.fit(W, y)
        selected_indices = list(np.where(np.abs(model.coef_) > 1e-6)[0])
        metrics = selection_accuracy(true_indices, selected_indices, p)
        mse_beta = np.mean((model.coef_ - beta_true) ** 2)
        return {
            'success': True,
            'precision': metrics['Precision'],
            'recall': metrics['Recall'],
            'f1': metrics['F1'],
            'specificity': metrics['Specificity'],
            'hamming': metrics['Hamming_Distance'],
            'accuracy': metrics['Accuracy'],
            'mse_beta': mse_beta,
            'selected_count': len(selected_indices)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def monte_carlo_evaluation(n_simulations, n, p, s, alpha, sigma=1.0, sigma_u=0.5):
    model_names = ['Naive Lasso', 'Corrected Lasso', 'CoCoLasso', 
                   'Adaptive Corrected Lasso', 'Adaptive CoCoLasso',
                   'RandomForest Corrected Lasso']
    
    W, y, true_indices, beta_true, Sigma_uu = generate_data(
        n=n, p=p, s=s, sigma=sigma, sigma_u=sigma_u, seed=0
    )

    models = {
        'Naive Lasso': LassoRegression(alpha=alpha),
        'Corrected Lasso': CorrectedLasso(alpha=alpha, Sigma_uu=Sigma_uu),
        'CoCoLasso': CoCoLasso(alpha=alpha, Sigma_uu=Sigma_uu),
        'Adaptive Corrected Lasso': AdaptiveCorrectedLasso(alpha=alpha, Sigma_uu=Sigma_uu),
        'Adaptive CoCoLasso': AdaptiveCoCoLasso(alpha=alpha, Sigma_uu=Sigma_uu),
        'RandomForest Corrected Lasso': RandomForestCorrectedLasso(alpha=alpha, Sigma_uu=Sigma_uu, n_estimators=50, max_depth=5)
    }
    
    all_results = {name: {
        'precision': [], 'recall': [], 'f1': [], 
        'specificity': [], 'hamming': [], 'accuracy': [], 
        'mse_beta': [], 'selected_count': [], 'success_count': 0
    } for name in model_names}

    for i in range(n_simulations):
        W, y, true_indices, beta_true, Sigma_uu = generate_data(
            n=n, p=p, s=s, sigma=sigma, sigma_u=sigma_u, seed=i
        )
        
        models['Corrected Lasso'] = CorrectedLasso(alpha=alpha, Sigma_uu=Sigma_uu)
        models['CoCoLasso'] = CoCoLasso(alpha=alpha, Sigma_uu=Sigma_uu)
        models['Adaptive Corrected Lasso'] = AdaptiveCorrectedLasso(alpha=alpha, Sigma_uu=Sigma_uu)
        models['Adaptive CoCoLasso'] = AdaptiveCoCoLasso(alpha=alpha, Sigma_uu=Sigma_uu)
        models['RandomForest Corrected Lasso'] = RandomForestCorrectedLasso(alpha=alpha, Sigma_uu=Sigma_uu, n_estimators=50, max_depth=5)
        
        for name, model in models.items():
            result = evaluate_model_once(model, W, y, true_indices, beta_true, p)
            if result['success']:
                all_results[name]['precision'].append(result['precision'])
                all_results[name]['recall'].append(result['recall'])
                all_results[name]['f1'].append(result['f1'])
                all_results[name]['specificity'].append(result['specificity'])
                all_results[name]['hamming'].append(result['hamming'])
                all_results[name]['accuracy'].append(result['accuracy'])
                all_results[name]['mse_beta'].append(result['mse_beta'])
                all_results[name]['selected_count'].append(result['selected_count'])
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
                'success_rate': count / n_simulations
            }
        else:
            avg_results[name] = {'success_rate': 0.0}
    return avg_results


def run_parameter_test(test_name, param_name, param_values, fixed_params, n_simulations=20):
    print(f"\n{'='*80}")
    print(f"测试: {test_name}")
    print(f"{'='*80}")
    
    model_names = ['Naive Lasso', 'Corrected Lasso', 'CoCoLasso', 
                   'Adaptive Corrected Lasso', 'Adaptive CoCoLasso',
                   'RandomForest Corrected Lasso']
    
    results = {name: {
        'f1': [], 'precision': [], 'recall': [], 
        'specificity': [], 'hamming': [], 'accuracy': [], 'mse_beta': []
    } for name in model_names}
    
    for i, val in enumerate(param_values):
        print(f"\n[{i+1}/{len(param_values)}] {param_name} = {val}")
        
        params = fixed_params.copy()
        params[param_name] = val
        
        avg_res = monte_carlo_evaluation(
            n_simulations=n_simulations,
            **params
        )
        
        for name in model_names:
            if avg_res[name].get('success_rate', 0) > 0.5:
                results[name]['f1'].append(avg_res[name]['f1'])
                results[name]['precision'].append(avg_res[name]['precision'])
                results[name]['recall'].append(avg_res[name]['recall'])
                results[name]['specificity'].append(avg_res[name]['specificity'])
                results[name]['hamming'].append(avg_res[name]['hamming'])
                results[name]['accuracy'].append(avg_res[name]['accuracy'])
                results[name]['mse_beta'].append(avg_res[name]['mse_beta'])
            else:
                results[name]['f1'].append(np.nan)
                results[name]['precision'].append(np.nan)
                results[name]['recall'].append(np.nan)
                results[name]['specificity'].append(np.nan)
                results[name]['hamming'].append(np.nan)
                results[name]['accuracy'].append(np.nan)
                results[name]['mse_beta'].append(np.nan)
    
    return param_values, results


def plot_comparison(x_values, results, xlabel, title, save_path):
    model_names = ['Naive Lasso', 'Corrected Lasso', 'CoCoLasso', 
                   'Adaptive Corrected Lasso', 'Adaptive CoCoLasso',
                   'RandomForest Corrected Lasso']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2']
    
    metrics = [
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1 Score'),
        ('specificity', 'Specificity'),
        ('accuracy', 'Accuracy'),
        ('hamming', 'Hamming Distance')
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for i, (metric, metric_name) in enumerate(metrics):
        ax = axes[i//3, i%3]
        for j, name in enumerate(model_names):
            ax.plot(x_values, results[name][metric], 'o-', label=name, 
                    color=colors[j], linewidth=2, markersize=6)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} vs {title}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if 'alpha' in xlabel.lower():
            ax.set_xscale('log')
        if metric == 'hamming':
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图已保存到: {save_path}")
    plt.close()


def main():
    print("="*80)
    print("高维测量误差变量选择工具包 - 综合对比测试（含随机森林）")
    print("="*80)
    
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    n_simulations = 100
    
    print(f"\n模拟次数: {n_simulations}")
    
    alphas = np.logspace(-2, -0.5, 20)
    fixed = {'n': 80, 'p': 100, 's': 5, 'sigma': 1.0, 'sigma_u': 0.5}
    x_vals, res_alpha = run_parameter_test(
        "正则化强度变化", "alpha", alphas, fixed, n_simulations
    )
    plot_comparison(x_vals, res_alpha, 'Regularization Parameter (alpha)', 
                 'Regularization', os.path.join(save_dir, f'alpha_comparison_rf_{timestamp}.png'))
    
    p_values = np.linspace(50, 300, 20, dtype=int)
    fixed = {'n': 80, 'alpha': 0.1, 's': 5, 'sigma': 1.0, 'sigma_u': 0.5}
    x_vals, res_p = run_parameter_test(
        "变量个数变化", "p", p_values, fixed, n_simulations
    )
    plot_comparison(x_vals, res_p, 'Number of Features (p)', 
                 'Number of Features', os.path.join(save_dir, f'p_comparison_rf_{timestamp}.png'))
    
    n_values = np.linspace(40, 200, 20, dtype=int)
    fixed = {'p': 100, 'alpha': 0.1, 's': 5, 'sigma': 1.0, 'sigma_u': 0.5}
    x_vals, res_n = run_parameter_test(
        "样本量变化", "n", n_values, fixed, n_simulations
    )
    plot_comparison(x_vals, res_n, 'Number of Samples (n)', 
                 'Number of Samples', os.path.join(save_dir, f'n_comparison_rf_{timestamp}.png'))
    
    sigma_u_values = np.linspace(0.1, 1.0, 20)
    fixed = {'n': 80, 'p': 100, 's': 5, 'alpha': 0.1, 'sigma': 1.0}
    x_vals, res_sigma_u = run_parameter_test(
        "测量误差强度变化", "sigma_u", sigma_u_values, fixed, n_simulations
    )
    plot_comparison(x_vals, res_sigma_u, 'Measurement Error Std (sigma_u)', 
                 'Measurement Error', os.path.join(save_dir, f'sigma_u_comparison_rf_{timestamp}.png'))
    
    default_results = monte_carlo_evaluation(
        n_simulations=n_simulations,
        n=80, p=100, s=5, alpha=0.1, sigma=1.0, sigma_u=0.5
    )
    
    all_results = {
        'alpha': {'x': alphas, 'results': res_alpha},
        'p': {'x': p_values, 'results': res_p},
        'n': {'x': n_values, 'results': res_n},
        'sigma_u': {'x': sigma_u_values, 'results': res_sigma_u},
        'default': default_results
    }
    
    with open(os.path.join(save_dir, f'all_results_rf_{timestamp}.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    
    print("\n" + "="*80)
    print("所有测试完成！结果已保存。")
    print("="*80)


if __name__ == '__main__':
    main()
