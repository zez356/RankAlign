from __future__ import annotations
import argparse
import csv
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import features
import run as R
from solvers import SOLVERS
COLUMNS = ['dataset', 'vision', 'language', 'solver', 'corruption', 'relabelled', 'Problem size', 'seed', 'Matching accuracy', 'Time elapsed', 'objective']

def corrupted_prototypes(V, y, seed, rate, frac=0.5):
    g = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(len(y), generator=g)[:int(len(y) * frac)].numpy()
    (X, ys) = (V[idx], y[idx].copy())
    classes = np.unique(y)
    C = len(classes)
    rng = np.random.default_rng(10000 + int(seed) * 101 + int(round(rate * 1000)))
    yt = ys.copy()
    for c in classes:
        Sc = np.flatnonzero(ys == c)
        m = int(np.floor(rate * len(Sc)))
        if m == 0:
            continue
        pick = rng.choice(Sc, m, replace=False)
        new = rng.integers(0, C - 1, size=m)
        yt[pick] = classes[new + (new >= np.searchsorted(classes, c))]
    P = np.stack([X[yt == c].mean(0) for c in classes])
    return (P / np.linalg.norm(P, axis=1, keepdims=True), float((yt != ys).mean()))

def instance(t, solvers, timelimit):
    (dataset, vision, language, seed, rate, V, y, L) = t
    (Px, noise) = corrupted_prototypes(V, y, seed, rate)
    rows = R.run_instance((dataset, vision, language, len(L), seed, Px, L), solvers, timelimit)
    for r in rows:
        r.update(corruption=rate, relabelled=noise)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', nargs='+', default=['cifar100', 'imagenet100'], choices=list(features.DATASET_DIR))
    ap.add_argument('--vision', nargs='+', default=['clip'], choices=list(features.VISION_FILES))
    ap.add_argument('--language', default='bge', choices=list(features.LANGUAGE_FILES))
    ap.add_argument('--solver', nargs='+', default=['faq'], choices=SOLVERS)
    ap.add_argument('--rates', nargs='+', type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--seed_values', nargs='+', type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument('--timelimit', type=float, default=300.0)
    ap.add_argument('--emb_root', default=features.EMB_ROOT, help=argparse.SUPPRESS)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)), help=argparse.SUPPRESS)
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--imagenet_val', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    features.ensure_representations(a.dataset, a.vision, a.language, a.emb_root, data_root=a.data_root, imagenet_val=a.imagenet_val)
    seeds = R._resolve_seeds(a)
    print('run seeds:', ' '.join((str(s) for s in seeds)), flush=True)
    tasks = []
    for ds in a.dataset:
        for v in a.vision:
            (V, L, y) = R.load_dataset(a.emb_root, ds, v, a.language)
            tasks += [(ds, v, a.language, s, r, V, y, L) for r in a.rates for s in seeds]
    if 'ot' in a.solver:
        import ot
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    with open(a.out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        if new:
            w.writeheader()
        if a.workers <= 1:
            it = (instance(t, a.solver, a.timelimit) for t in tasks)
        else:
            import multiprocessing as mp
            pool = mp.get_context('fork').Pool(a.workers)
            it = pool.starmap(instance, [(t, a.solver, a.timelimit) for t in tasks])
        for rows in it:
            for r in rows:
                w.writerow(r)
                print(f"{r['dataset']:11s} {r['vision']:8s} {r['solver']:4s} r={r['corruption']:.2f} seed={r['seed']} acc={r['Matching accuracy']:.2f}", flush=True)
            f.flush()
if __name__ == '__main__':
    main()
