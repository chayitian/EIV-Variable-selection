import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, RandomizedSearchCV

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation import selection_accuracy
from src.models.eiv.adaptive import ACoCoLasso
from src.models.eiv.canonical import CLasso, CoCoLasso

warnings.filterwarnings('ignore', category=ConvergenceWarning)


def build_covariance_matrix(p, structure):
    if structure == 'ar':
        idx = np.arange(p)
        return 0.5 ** np.abs(idx[:, None] - idx[None, :])
    if structure == 'cs':
        return 0.3 * np.ones((p, p)) + 0.7 * np.eye(p)
    raise ValueError("structure must be 'ar' or 'cs'")


def generate_dataset(n, p, beta_true, sigma, tau, sigma_x, rng):
    x_true = rng.multivariate_normal(mean=np.zeros(p), cov=sigma_x, size=n)
    eps = rng.normal(loc=0.0, scale=sigma, size=n)
    y = x_true @ beta_true + eps

    a = rng.normal(loc=0.0, scale=tau, size=(n, p))
    w = x_true + a
    sigma_uu = (tau ** 2) * np.eye(p)
    return x_true, w, y, sigma_uu


def normalize_feature_importance(feature_importance, eps=1e-8):
    importance = np.asarray(feature_importance, dtype=float).reshape(-1)
    importance = np.clip(importance, 0.0, None)

    total = float(np.sum(importance))
    if (not np.isfinite(total)) or total <= 0.0:
        return np.full_like(importance, 1.0 / importance.size)

    normalized = importance / total
    normalized = np.clip(normalized, eps, None)
    normalized = normalized / float(np.sum(normalized))
    return normalized


def tune_xgboost_and_get_importance(w, y, seed, cv_folds=5):
    base_xgb = xgb.XGBRegressor(
        objective='reg:squarederror',
        importance_type='gain',
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )

    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.7, 0.85, 1.0],
        'colsample_bytree': [0.5, 0.7, 1.0],
        'min_child_weight': [1.0, 3.0, 5.0],
        'gamma': [0.0, 0.1, 0.3],
        'reg_alpha': [0.0, 0.1, 1.0],
        'reg_lambda': [1.0, 5.0, 10.0],
    }

    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_distributions,
        n_iter=10,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        cv=cv_folds,
        random_state=seed,
        refit=True,
    )
    search.fit(w, y)

    best_params = dict(search.best_params_)
    best_cv_loss = float(-search.best_score_)
    best_importance = normalize_feature_importance(search.best_estimator_.feature_importances_)
    return best_params, best_cv_loss, best_importance


def fit_xgboost_with_best_params_and_get_importance(w, y, xgb_params, seed):
    params = dict(xgb_params)
    params['objective'] = 'reg:squarederror'
    params['importance_type'] = 'gain'
    params['random_state'] = seed
    params['n_jobs'] = -1
    params['verbosity'] = 0

    model = xgb.XGBRegressor(**params)
    model.fit(w, y)
    return normalize_feature_importance(model.feature_importances_)


def build_model(method_name, alpha, sigma_uu, init_coef=None):
    if method_name == 'CLasso':
        return CLasso(alpha=alpha, Sigma_uu=sigma_uu)
    if method_name == 'CoCoLasso':
        return CoCoLasso(alpha=alpha, Sigma_uu=sigma_uu)
    if method_name in ('ACoCoLasso', 'XGBoostACoCoLasso'):
        if init_coef is None:
            raise ValueError(f'{method_name} requires init_coef')
        return ACoCoLasso(alpha=alpha, Sigma_uu=sigma_uu, init_coef=init_coef)
    raise ValueError(f'Unknown method: {method_name}')


def select_alpha_with_cv(method_name, w, y, sigma_uu, alpha_grid, cv_folds=10, seed=0, init_coef=None):
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    mean_losses = []

    for alpha in alpha_grid:
        fold_losses = []
        for train_idx, val_idx in kf.split(w):
            w_train, w_val = w[train_idx], w[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = build_model(method_name, alpha, sigma_uu, init_coef=init_coef)
            try:
                model.fit(w_train, y_train)
                y_pred = model.predict(w_val)
                loss = float(np.mean((y_val - y_pred) ** 2))
            except Exception:
                loss = np.inf
            fold_losses.append(loss)

        mean_losses.append(float(np.mean(fold_losses)))

    best_idx = int(np.argmin(mean_losses))
    return float(alpha_grid[best_idx]), float(mean_losses[best_idx])


def fit_init_model_and_get_coef_acoco(w, y, sigma_uu, alpha_grid, cv_folds, seed):
    init_alpha, init_cv_loss = select_alpha_with_cv(
        method_name='CoCoLasso',
        w=w,
        y=y,
        sigma_uu=sigma_uu,
        alpha_grid=alpha_grid,
        cv_folds=cv_folds,
        seed=seed,
    )
    init_model = CoCoLasso(alpha=init_alpha, Sigma_uu=sigma_uu)
    init_model.fit(w, y)
    return init_model.coef_.copy(), init_alpha, init_cv_loss


def summarize_metric(values):
    if len(values) == 0:
        return np.nan, np.nan
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def write_csv(path, headers, rows):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_value(mean_val, std_val, decimals=2):
    return f"{mean_val:.{decimals}f} ({std_val:.{decimals}f})"


def build_measure_table_rows(taus, method_columns, summary_store):
    rows = []
    metric_specs = [
        ('C', 'TP', 2),
        ('IC', 'FP', 2),
        ('PE', 'PE', 2),
        ('MSE', 'MSE', 2),
    ]

    for tau in taus:
        rows.append({'Tau': f'tau = {tau:.2f}', 'Measure': ''})
        for metric_name, metric_key, decimals in metric_specs:
            row = {'Tau': '', 'Measure': metric_name}
            for method in method_columns:
                m, s = summary_store[(tau, method)][metric_key]
                row[method] = format_value(m, s, decimals=decimals)
            rows.append(row)
    return rows


def main():
    cv_folds = 10
    xgb_cv_folds = 5
    alpha_grid = np.logspace(-2, 0.6, 10)
    selection_threshold = 1e-6

    methods = [
        'CLasso',
        'CoCoLasso',
        'ACoCoLasso',
        'XGBoostACoCoLasso',
    ]
    experiments = [
        {
            'name': 'high_dim',
            'display_name': 'Experiment 1 (n=100, p=250)',
            'n': 100,
            'p': 250,
            'sigma': 3.0,
            'taus': [0.75, 1.25],
            'n_simulations': 100,
            'structures': [
                ('ar', 'AR_0.5'),
                ('cs', 'Compound_Symmetry'),
            ],
            'beta_nonzero': [3.0, 1.5, 2.0],
        },
        {
            'name': 'ultra_high_dim',
            'display_name': 'Experiment 2 (n=80, p=1000)',
            'n': 80,
            'p': 1000,
            'sigma': 1.0,
            'taus': [0.25, 0.5],
            'n_simulations': 100,
            'structures': [
                ('ar', 'AR_0.5'),
            ],
            'beta_nonzero': [1.0, -0.5, 0.7, -1.2, -0.9, 0.3, 0.55],
        },
    ]

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 90)
    print('XGBoostACoCoLasso reproduction started')
    print(f'Output directory: {output_dir}')
    print('=' * 90)

    for exp_idx, exp in enumerate(experiments):
        n = exp['n']
        p = exp['p']
        sigma = exp['sigma']
        taus = exp['taus']
        n_simulations = exp['n_simulations']
        structures = exp['structures']

        beta_true = np.zeros(p)
        for idx, coef in enumerate(exp['beta_nonzero']):
            beta_true[idx] = coef
        true_indices = list(range(len(exp['beta_nonzero'])))

        print('\n' + '=' * 90)
        print(f"{exp['display_name']} started")
        print('=' * 90)

        for struct_idx, (struct_key, struct_label) in enumerate(structures):
            sigma_x = build_covariance_matrix(p, struct_key)
            summary_store = {}

            print('\n' + '-' * 90)
            print(f'Covariance structure: {struct_label}')
            print('-' * 90)

            for tau_idx, tau in enumerate(taus):
                for method_idx, method in enumerate(methods):
                    cv_seed = 1000 + exp_idx * 1_000_000 + struct_idx * 100 + tau_idx * 10 + method_idx
                    rng_cv = np.random.default_rng(cv_seed)
                    _, w_cv, y_cv, sigma_uu_cv = generate_dataset(
                        n=n,
                        p=p,
                        beta_true=beta_true,
                        sigma=sigma,
                        tau=tau,
                        sigma_x=sigma_x,
                        rng=rng_cv,
                    )

                    init_coef_cv = None
                    init_alpha = None
                    init_cv_loss = None
                    xgb_best_params = None
                    xgb_cv_loss = None

                    if method == 'ACoCoLasso':
                        init_coef_cv, init_alpha, init_cv_loss = fit_init_model_and_get_coef_acoco(
                            w=w_cv,
                            y=y_cv,
                            sigma_uu=sigma_uu_cv,
                            alpha_grid=alpha_grid,
                            cv_folds=cv_folds,
                            seed=cv_seed,
                        )
                    elif method == 'XGBoostACoCoLasso':
                        xgb_best_params, xgb_cv_loss, init_coef_cv = tune_xgboost_and_get_importance(
                            w=w_cv,
                            y=y_cv,
                            seed=cv_seed,
                            cv_folds=xgb_cv_folds,
                        )

                    best_alpha, cv_loss = select_alpha_with_cv(
                        method_name=method,
                        w=w_cv,
                        y=y_cv,
                        sigma_uu=sigma_uu_cv,
                        alpha_grid=alpha_grid,
                        cv_folds=cv_folds,
                        seed=cv_seed,
                        init_coef=init_coef_cv,
                    )

                    tp_vals = []
                    fp_vals = []
                    mse_vals = []
                    pe_vals = []

                    if method == 'ACoCoLasso':
                        print(
                            f'Running method={method}, tau={tau}, init_model=CoCoLasso, '
                            f'init_alpha={init_alpha:.6f}, init_cv_loss={init_cv_loss:.6f}, '
                            f'best_alpha={best_alpha:.6f}, cv_loss={cv_loss:.6f}'
                        )
                    elif method == 'XGBoostACoCoLasso':
                        print(
                            f'Running method={method}, tau={tau}, xgb_cv_loss={xgb_cv_loss:.6f}, '
                            f'best_alpha={best_alpha:.6f}, cv_loss={cv_loss:.6f}, '
                            f'xgb_best_params={xgb_best_params}'
                        )
                    else:
                        print(
                            f'Running method={method}, tau={tau}, '
                            f'best_alpha={best_alpha:.6f}, cv_loss={cv_loss:.6f}'
                        )

                    for sim in range(n_simulations):
                        sim_seed = (
                            10_000
                            + exp_idx * 100_000_000
                            + struct_idx * 10_000_000
                            + tau_idx * 1_000_000
                            + method_idx * 100_000
                            + sim
                        )
                        rng = np.random.default_rng(sim_seed)

                        x_true, w, y, sigma_uu = generate_dataset(
                            n=n,
                            p=p,
                            beta_true=beta_true,
                            sigma=sigma,
                            tau=tau,
                            sigma_x=sigma_x,
                            rng=rng,
                        )

                        try:
                            init_coef_sim = None
                            if method == 'ACoCoLasso':
                                init_model = CoCoLasso(alpha=init_alpha, Sigma_uu=sigma_uu)
                                init_model.fit(w, y)
                                init_coef_sim = init_model.coef_.copy()
                            elif method == 'XGBoostACoCoLasso':
                                init_coef_sim = fit_xgboost_with_best_params_and_get_importance(
                                    w=w,
                                    y=y,
                                    xgb_params=xgb_best_params,
                                    seed=sim_seed,
                                )

                            model = build_model(method, best_alpha, sigma_uu, init_coef=init_coef_sim)
                            model.fit(w, y)
                            selected = np.where(np.abs(model.coef_) > selection_threshold)[0].tolist()

                            metrics = selection_accuracy(
                                true_indices=true_indices,
                                selected_indices=selected,
                                total_features=p,
                                beta_true=beta_true,
                                beta_hat=model.coef_,
                                x_true=x_true,
                            )

                            tp_vals.append(float(metrics['TP']))
                            fp_vals.append(float(metrics['FP']))
                            mse_vals.append(float(metrics['MSE']))
                            pe_vals.append(float(metrics['PE']))
                        except Exception:
                            continue

                    summary_store[(tau, method)] = {
                        'TP': summarize_metric(tp_vals),
                        'FP': summarize_metric(fp_vals),
                        'MSE': summarize_metric(mse_vals),
                        'PE': summarize_metric(pe_vals),
                        'Best_Lambda': best_alpha,
                        'CV_Loss': cv_loss,
                        'XGB_CV_Loss': xgb_cv_loss,
                        'XGB_Best_Params': xgb_best_params,
                    }

            headers = ['Tau', 'Measure'] + methods
            rows = build_measure_table_rows(taus, methods, summary_store)
            csv_path = output_dir / f"XGBoostACoCoLasso_{exp['name']}_{struct_label}.csv"
            write_csv(csv_path, headers, rows)
            print(f'Saved table: {csv_path}')

    print('\nAll done.')


if __name__ == '__main__':
    main()
