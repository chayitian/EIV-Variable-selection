import numpy as np
import matplotlib
matplotlib.use('Agg')
import argparse
import sys
import os
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.base import Lasso
from src.models.eiv.canonical import CLasso, CoCoLasso
from src.models.eiv.adaptive import ACLasso
from src.models.eiv.feature_weighted import RFACLasso
from src.experiments import monte_carlo_evaluation, run_parameter_test, plot_comparison


def parse_args():
    parser = argparse.ArgumentParser(description='综合对比测试（含随机森林）')
    parser.add_argument('--n_simulations', type=int, default=100, help='蒙特卡洛模拟次数')
    parser.add_argument('--selection_threshold', type=float, default=1e-6, help='变量选择阈值 abs(coef) > threshold')
    parser.add_argument('--weight_method', choices=['normalized', 'max_scaled'], default='max_scaled', help='树加权模型权重计算方法')
    return parser.parse_args()


def build_model_builders(weight_method):
    rf_tree_params = {
        'n_estimators': 100,
        'max_depth': 8,
        'max_features': 'sqrt',
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1,
    }

    return {
        'Naive Lasso': lambda alpha, sigma_uu: Lasso(alpha=alpha),
        'Corrected Lasso': lambda alpha, sigma_uu: CLasso(alpha=alpha, Sigma_uu=sigma_uu),
        'CoCoLasso': lambda alpha, sigma_uu: CoCoLasso(alpha=alpha, Sigma_uu=sigma_uu),
        'Adaptive Corrected Lasso': lambda alpha, sigma_uu: ACLasso(
            final_l1_alpha=alpha,
            init_l1_alpha=alpha,
            Sigma_uu=sigma_uu,
        ),
        'RandomForest Corrected Lasso': lambda alpha, sigma_uu: RFACLasso(
            alpha=alpha,
            Sigma_uu=sigma_uu,
            gamma=1.0,
            weight_method=weight_method,
            **rf_tree_params
        ),
    }


def main():
    args = parse_args()

    print('=' * 80)
    print('高维测量误差变量选择工具包 - 综合对比测试（含随机森林）')
    print('=' * 80)

    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    n_simulations = args.n_simulations
    selection_threshold = args.selection_threshold
    weight_method = args.weight_method
    model_builders = build_model_builders(weight_method=weight_method)
    model_names = list(model_builders.keys())

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#e377c2']
    marker_map = {
        'Naive Lasso': 'o',
        'Corrected Lasso': 's',
        'CoCoLasso': '^',
        'Adaptive Corrected Lasso': 'D',
        'RandomForest Corrected Lasso': 'P',
    }

    print('\n模拟次数: {}'.format(n_simulations))
    print('选择阈值: {}'.format(selection_threshold))
    print('权重方法: {}'.format(weight_method))

    alphas = np.logspace(-2, 0, 20)
    fixed = {'n': 100, 'p': 10, 's': 5, 'sigma': 1.0, 'sigma_u': 0.5}
    x_vals, res_alpha = run_parameter_test(
        test_name='正则化强度变化',
        param_name='alpha',
        param_values=alphas,
        fixed_params=fixed,
        n_simulations=n_simulations,
        model_builders=model_builders,
        selection_threshold=selection_threshold,
    )
    plot_comparison(
        x_values=x_vals,
        results=res_alpha,
        model_names=model_names,
        xlabel='Regularization Parameter (alpha)',
        title='Regularization',
        save_path=os.path.join(save_dir, 'alpha_comparison_rf_{}.png'.format(timestamp)),
        colors=colors,
        marker_map=marker_map,
    )

    p_values = np.linspace(10, 300, 20, dtype=int)
    fixed = {'n': 100, 'alpha': 0.1, 's': 5, 'sigma': 1.0, 'sigma_u': 0.5}
    x_vals, res_p = run_parameter_test(
        test_name='变量个数变化',
        param_name='p',
        param_values=p_values,
        fixed_params=fixed,
        n_simulations=n_simulations,
        model_builders=model_builders,
        selection_threshold=selection_threshold,
    )
    plot_comparison(
        x_values=x_vals,
        results=res_p,
        model_names=model_names,
        xlabel='Number of Features (p)',
        title='Number of Features',
        save_path=os.path.join(save_dir, 'p_comparison_rf_{}.png'.format(timestamp)),
        colors=colors,
        marker_map=marker_map,
    )

    n_values = np.linspace(40, 1000, 20, dtype=int)
    fixed = {'p': 10, 'alpha': 0.1, 's': 5, 'sigma': 1.0, 'sigma_u': 0.5}
    x_vals, res_n = run_parameter_test(
        test_name='样本量变化',
        param_name='n',
        param_values=n_values,
        fixed_params=fixed,
        n_simulations=n_simulations,
        model_builders=model_builders,
        selection_threshold=selection_threshold,
    )
    plot_comparison(
        x_values=x_vals,
        results=res_n,
        model_names=model_names,
        xlabel='Number of Samples (n)',
        title='Number of Samples',
        save_path=os.path.join(save_dir, 'n_comparison_rf_{}.png'.format(timestamp)),
        colors=colors,
        marker_map=marker_map,
    )

    sigma_u_values = np.linspace(0.1, 1.0, 20)
    fixed = {'n': 100, 'p': 10, 's': 5, 'alpha': 0.1, 'sigma': 1.0}
    x_vals, res_sigma_u = run_parameter_test(
        test_name='测量误差强度变化',
        param_name='sigma_u',
        param_values=sigma_u_values,
        fixed_params=fixed,
        n_simulations=n_simulations,
        model_builders=model_builders,
        selection_threshold=selection_threshold,
    )
    plot_comparison(
        x_values=x_vals,
        results=res_sigma_u,
        model_names=model_names,
        xlabel='Measurement Error Std (sigma_u)',
        title='Measurement Error',
        save_path=os.path.join(save_dir, 'sigma_u_comparison_rf_{}.png'.format(timestamp)),
        colors=colors,
        marker_map=marker_map,
    )

    default_results = monte_carlo_evaluation(
        n_simulations=n_simulations,
        n=100,
        p=10,
        s=5,
        alpha=0.1,
        sigma=1.0,
        sigma_u=0.5,
        model_builders=model_builders,
        selection_threshold=selection_threshold,
    )

    all_results = {
        'config': {
            'n_simulations': n_simulations,
            'selection_threshold': selection_threshold,
            'weight_method': weight_method,
        },
        'alpha': {'x': alphas, 'results': res_alpha},
        'p': {'x': p_values, 'results': res_p},
        'n': {'x': n_values, 'results': res_n},
        'sigma_u': {'x': sigma_u_values, 'results': res_sigma_u},
        'default': default_results,
    }

    with open(os.path.join(save_dir, 'all_results_rf_{}.pkl'.format(timestamp)), 'wb') as f:
        pickle.dump(all_results, f)

    print('\n' + '=' * 80)
    print('所有测试完成！结果已保存。')
    print('=' * 80)


if __name__ == '__main__':
    main()
