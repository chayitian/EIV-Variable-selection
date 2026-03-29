import argparse
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


SECTION_ORDER = ('alpha', 'p', 'n', 'sigma_u')
DEFAULT_PATTERN = 'all_results_*.pkl'


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _to_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _infer_experiment_name(file_name: str) -> str:
    match = re.match(r'all_results_(.+?)_\d{8}_\d{6}\.pkl$', file_name)
    if match:
        return match.group(1)

    match = re.match(r'all_results_(.+?)\.pkl$', file_name)
    if match:
        return match.group(1)

    return 'unknown'


def collect_result_pickles(input_root: Path, pattern: str = DEFAULT_PATTERN) -> List[Path]:
    if not input_root.exists():
        raise FileNotFoundError('Input root not found: {}'.format(input_root))

    return sorted([p for p in input_root.rglob(pattern) if p.is_file()])


def _flatten_single_file(pkl_path: Path, input_root: Path) -> List[Dict[str, Any]]:
    with pkl_path.open('rb') as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        return []

    config = data.get('config') if isinstance(data.get('config'), dict) else {}
    experiment = _infer_experiment_name(pkl_path.name)
    run_id = pkl_path.stem
    rel_path = str(pkl_path.relative_to(input_root))

    rows = []
    for section in SECTION_ORDER:
        section_data = data.get(section)
        if not isinstance(section_data, dict):
            continue

        x_values = _as_list(section_data.get('x'))
        results = section_data.get('results')
        if not isinstance(results, dict):
            continue

        for model_name, metric_dict in results.items():
            if not isinstance(metric_dict, dict):
                continue

            for metric_name, metric_series in metric_dict.items():
                series = _as_list(metric_series)
                n_points = min(len(x_values), len(series))
                for idx in range(n_points):
                    rows.append(
                        {
                            'source_file': pkl_path.name,
                            'source_relative_path': rel_path,
                            'experiment': experiment,
                            'run_id': run_id,
                            'section': section,
                            'x_index': idx,
                            'x_value': _to_scalar(x_values[idx]),
                            'model': model_name,
                            'metric': metric_name,
                            'value': _to_scalar(series[idx]),
                            'n_simulations': config.get('n_simulations'),
                            'selection_threshold': config.get('selection_threshold'),
                            'weight_method': config.get('weight_method'),
                        }
                    )

    return rows


def flatten_results_pickles(pkl_files: Sequence[Path], input_root: Path) -> pd.DataFrame:
    records = []
    for pkl_path in pkl_files:
        try:
            records.extend(_flatten_single_file(pkl_path, input_root))
        except Exception as exc:
            print('跳过文件 {}，原因: {}'.format(pkl_path, exc))

    if not records:
        return pd.DataFrame(
            columns=[
                'source_file',
                'source_relative_path',
                'experiment',
                'run_id',
                'section',
                'x_index',
                'x_value',
                'model',
                'metric',
                'value',
                'n_simulations',
                'selection_threshold',
                'weight_method',
            ]
        )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(
        by=['source_relative_path', 'section', 'x_index', 'model', 'metric'],
        kind='stable',
    ).reset_index(drop=True)
    return df


def export_flattened_results(df: pd.DataFrame, output_xlsx: Path, output_csv: Optional[Path] = None) -> None:
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    try:
        with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='long_table', index=False)
            for section in SECTION_ORDER:
                section_df = df[df['section'] == section]
                section_df.to_excel(writer, sheet_name=section, index=False)
    except ImportError as exc:
        raise ImportError('导出 xlsx 需要 openpyxl，请先安装: pip install openpyxl') from exc

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')


def flatten_results_to_excel(
    input_root: Path,
    pattern: str = DEFAULT_PATTERN,
    output_xlsx: Optional[Path] = None,
    with_csv: bool = False,
) -> Dict[str, Any]:
    pkl_files = collect_result_pickles(input_root=input_root, pattern=pattern)
    if not pkl_files:
        raise FileNotFoundError('未找到匹配文件: {}/**/{}'.format(input_root, pattern))

    df = flatten_results_pickles(pkl_files=pkl_files, input_root=input_root)
    if df.empty:
        raise ValueError('找到 pkl 文件，但未解析出可用实验数据。')

    if output_xlsx is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_xlsx = input_root / 'flattened' / 'flattened_results_{}.xlsx'.format(timestamp)

    output_csv = output_xlsx.with_suffix('.csv') if with_csv else None
    export_flattened_results(df=df, output_xlsx=output_xlsx, output_csv=output_csv)

    return {
        'input_root': str(input_root),
        'files_count': len(pkl_files),
        'rows_count': int(len(df)),
        'output_xlsx': str(output_xlsx),
        'output_csv': str(output_csv) if output_csv is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='将 results 子目录下 all_results_*.pkl 拍平并导出为一个 xlsx（长表 + 四个 sheet）。'
    )
    parser.add_argument('--input_root', type=str, default='results', help='输入目录（会递归查找 pkl）')
    parser.add_argument('--pattern', type=str, default=DEFAULT_PATTERN, help='查找文件名模式')
    parser.add_argument('--output_xlsx', type=str, default=None, help='输出 xlsx 路径')
    parser.add_argument('--with_csv', action='store_true', help='同时导出一份长表 csv')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_xlsx = Path(args.output_xlsx) if args.output_xlsx else None

    summary = flatten_results_to_excel(
        input_root=input_root,
        pattern=args.pattern,
        output_xlsx=output_xlsx,
        with_csv=args.with_csv,
    )

    print('=' * 80)
    print('结果拍平完成')
    print('输入目录: {}'.format(summary['input_root']))
    print('处理文件数: {}'.format(summary['files_count']))
    print('输出行数: {}'.format(summary['rows_count']))
    print('XLSX: {}'.format(summary['output_xlsx']))
    if summary['output_csv']:
        print('CSV: {}'.format(summary['output_csv']))
    print('=' * 80)


if __name__ == '__main__':
    main()
