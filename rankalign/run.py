from __future__ import annotations
import argparse
import json
import os
import secrets
import sys
import time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import features
from rankalign import load_labels, load_language_prototypes, load_vision_embeddings, rankalign_matrices, vision_prototypes
from solvers import SOLVERS, initial_guess, objective, p_dispersion_subset, solve_backend

def rankalign_subset_cost(Px_full, Py):
    (A, B) = rankalign_matrices(Px_full, Py, center=True, representation='full')
    C = (A - B) ** 2
    np.fill_diagonal(C, 0.0)
    return C + C.T

def load_dataset(emb_root, dataset, vision, language):
    V = load_vision_embeddings(features.vision_path(emb_root, dataset, vision))
    L = load_language_prototypes(features.language_path(emb_root, dataset, language))
    y = load_labels(features.labels_path(emb_root, dataset))
    classes = np.unique(y)
    if len(classes) != len(L):
        raise ValueError(f'{dataset}: labels contain {len(classes)} classes but language file has {len(L)} class rows')
    return (V, L, y)
CSV_COLUMNS = ['dataset', 'vision', 'language', 'method', 'solver', 'Problem size', 'seed', 'Matching accuracy', 'Time elapsed', 'objective', 'Optimal', 'Bound', 'timelimit']

def write_csv_row(path, row):
    import csv
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in CSV_COLUMNS})

def _random_seeds(count):
    count = int(count)
    if count < 1:
        raise ValueError('--seeds must be at least 1')
    rng = secrets.SystemRandom()
    (values, seen) = ([], set())
    while len(values) < count:
        value = rng.randrange(1, 2 ** 31)
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values

def _resolve_seeds(args):
    if args.seed_values is not None:
        values = [int(x) for x in args.seed_values]
        if not values:
            raise ValueError('At least one seed value is required')
        return values
    return _random_seeds(args.seeds)

def build_instances(args, seeds):
    tasks = []
    for dataset in args.dataset:
        for vision in args.vision:
            (V, L, y) = load_dataset(args.emb_root, dataset, vision, args.language)
            Pfull = vision_prototypes(V, y, frac=1.0)
            cost = rankalign_subset_cost(Pfull, L)
            total_classes = len(L)
            for N in args.sizes:
                if N > total_classes:
                    raise ValueError(f'Requested N={N}, but {dataset} has only {total_classes} classes')
                subset = p_dispersion_subset(cost, N)
                for seed in seeds:
                    Px = vision_prototypes(V, y, seed=seed, frac=0.5)[subset]
                    Py = L[subset]
                    tasks.append((dataset, vision, args.language, N, seed, Px, Py))
    return tasks

def run_instance(task, solvers, timelimit):
    (dataset, vision, language, N, seed, Px, Py) = task
    (A, B) = rankalign_matrices(Px, Py, center=True, representation='full')
    (R_x, R_y) = rankalign_matrices(Px, Py, center=True, representation='rank')
    (p0, _) = initial_guess([(R_x, R_y), (A, B)], seed=seed)
    rows = []
    for solver in solvers:
        t1 = time.perf_counter()
        (perm, extra) = solve_backend(solver, A, B, p0, timelimit)
        elapsed = time.perf_counter() - t1
        rows.append({'dataset': dataset, 'vision': vision, 'language': language, 'method': 'RankAlign', 'solver': solver.upper(), 'Problem size': N, 'seed': seed, 'Matching accuracy': float((perm == np.arange(N)).mean()), 'Time elapsed': elapsed, 'objective': objective(A, B, perm), 'Optimal': extra.get('Optimal', False), 'Cost': extra.get('Cost'), 'Bound': extra.get('Bound'), 'timelimit': timelimit})
    return rows

def main():
    ap = argparse.ArgumentParser(description='RankAlign from precomputed frozen representations')
    ap.add_argument('--dataset', nargs='+', default=['cifar100'], choices=list(features.DATASET_DIR))
    ap.add_argument('--vision', nargs='+', default=['clip'], choices=list(features.VISION_FILES))
    ap.add_argument('--language', default='bge', choices=list(features.LANGUAGE_FILES))
    ap.add_argument('--solver', nargs='+', default=['faq'], choices=SOLVERS, help='faq (default), ot, fhg (optional, adapted from itsamatch)')
    ap.add_argument('--sizes', nargs='+', type=int, default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed_values', nargs='+', type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument('--timelimit', type=float, default=300.0)
    ap.add_argument('--emb_root', default=features.EMB_ROOT, help=argparse.SUPPRESS)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)), help=argparse.SUPPRESS)
    ap.add_argument('--data_root', default=None, help='CIFAR-100 location (only needed if representations are missing)')
    ap.add_argument('--imagenet_val', default=None, help='ImageNet val/ folder (only needed if ImageNet-100 representations are missing)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    features.ensure_representations(args.dataset, args.vision, args.language, args.emb_root, data_root=args.data_root, imagenet_val=args.imagenet_val)
    seeds = _resolve_seeds(args)
    print('run seeds:', ' '.join((str(s) for s in seeds)), flush=True)
    tasks = build_instances(args, seeds)
    tasks.sort(key=lambda t: -t[3])
    print(f'{len(tasks)} instances x {len(args.solver)} solver(s) -> {args.out}', flush=True)
    if 'ot' in args.solver:
        import ot

    def write(rows):
        for row in rows:
            if args.out.endswith('.csv'):
                write_csv_row(args.out, row)
            else:
                with open(args.out, 'a') as f:
                    f.write(json.dumps(row) + '\n')
            print(f"{row['dataset']:11s} {row['vision']:8s} {row['solver']:4s} N={row['Problem size']:3d} seed={row['seed']} acc={row['Matching accuracy']:.2f} t={row['Time elapsed']:.2f}s", flush=True)
    if args.workers <= 1:
        for task in tasks:
            write(run_instance(task, args.solver, args.timelimit))
    else:
        import multiprocessing as mp
        payload = [(t, args.solver, args.timelimit) for t in tasks]
        with mp.get_context('fork').Pool(args.workers) as pool:
            for rows in pool.starmap(run_instance, payload):
                write(rows)
if __name__ == '__main__':
    main()
