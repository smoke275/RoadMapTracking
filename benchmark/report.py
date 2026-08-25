"""Plain-text report formatting in the paper's Table 3 layout, plus
persistence of results (table + per-seed raw CSV) to benchmark_results/."""
import csv
import os
import time

STRATEGY_LABELS = {
    'minmax-alpha': 'Min-Max α-Guard (Proposed)',
    'geo-follow': 'Geo-Follow',
    'tsp-patrol': 'TSP-Patrol',
    'kernel-control': 'Kernel-Weighted Control (Mandal & Bhattacharya, ICRA25)',
}
EVADER_LABELS = {
    'skeleton': 'Skeleton Evader',
    'adversarial': 'Adversarial Evader',
}


def format_table(results: list) -> str:
    header = (f'{"Environment":<12} {"Strategy":<28} {"Evader":<20} '
              f'{"ᾱ":>12} {"α_max":>12} {"%LOS":>12} {"N_breach":>10}')
    lines = [header, '-' * len(header)]
    for r in results:
        agg = r.aggregate()

        def fmt(key, pct=False):
            mean, std = agg[key]
            suffix = '%' if pct else ''
            return f'{mean:.2f}±{std:.2f}{suffix}'

        mean_b, std_b = agg['n_breach']
        lines.append(
            f'{r.environment:<12} '
            f'{STRATEGY_LABELS.get(r.strategy, r.strategy):<28} '
            f'{EVADER_LABELS.get(r.evader, r.evader):<20} '
            f'{fmt("mean_alpha"):>12} {fmt("peak_alpha"):>12} '
            f'{fmt("los_pct", pct=True):>12} '
            f'{mean_b:>7.1f}±{std_b:.1f}')
    return '\n'.join(lines)


def save_results(results: list, run_args: dict = None,
                 out_dir: str = None) -> tuple:
    """Persist a benchmark run: the aggregate table (plus the CLI settings
    that produced it) as .txt and every per-seed metric row as .csv.
    Returns (table_path, csv_path)."""
    if out_dir is None:
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'benchmark_results')
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')

    table_path = os.path.join(out_dir, f'{stamp}_table.txt')
    with open(table_path, 'w') as f:
        if run_args:
            f.write('# ' + ' '.join(f'{k}={v}' for k, v in run_args.items())
                    + '\n')
        f.write(format_table(results) + '\n')

    csv_path = os.path.join(out_dir, f'{stamp}_raw.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = None
        for r in results:
            for seed, metrics in enumerate(r.per_seed):
                row = {'environment': r.environment, 'strategy': r.strategy,
                       'evader': r.evader, 'seed': seed, **metrics}
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)

    return table_path, csv_path
