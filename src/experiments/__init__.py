from .comparison_common import (
    generate_data,
    evaluate_model_once,
    monte_carlo_evaluation,
    run_parameter_test,
    plot_comparison,
)


def collect_result_pickles(*args, **kwargs):
    from .results_flatten import collect_result_pickles as _collect_result_pickles
    return _collect_result_pickles(*args, **kwargs)


def flatten_results_pickles(*args, **kwargs):
    from .results_flatten import flatten_results_pickles as _flatten_results_pickles
    return _flatten_results_pickles(*args, **kwargs)


def export_flattened_results(*args, **kwargs):
    from .results_flatten import export_flattened_results as _export_flattened_results
    return _export_flattened_results(*args, **kwargs)


def flatten_results_to_excel(*args, **kwargs):
    from .results_flatten import flatten_results_to_excel as _flatten_results_to_excel
    return _flatten_results_to_excel(*args, **kwargs)

__all__ = [
    'generate_data',
    'evaluate_model_once',
    'monte_carlo_evaluation',
    'run_parameter_test',
    'plot_comparison',
    'collect_result_pickles',
    'flatten_results_pickles',
    'export_flattened_results',
    'flatten_results_to_excel',
]
