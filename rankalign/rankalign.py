from __future__ import annotations
import numpy as np
import torch
from scipy.stats import rankdata

def load_vision_embeddings(path: str) -> np.ndarray:
    x = torch.load(path, map_location='cpu')
    if isinstance(x, dict):
        for key in ('embeddings', 'features', 'x'):
            if key in x:
                x = x[key]
                break
    x = torch.as_tensor(x).double().cpu().numpy()
    if x.ndim != 2:
        raise ValueError(f'Vision file must contain a 2-D tensor [samples, dim], got {x.shape} from {path}')
    return x

def load_language_prototypes(path: str) -> np.ndarray:
    x = torch.load(path, map_location='cpu')
    if isinstance(x, dict):
        for key in ('prototypes', 'embeddings', 'features', 'x'):
            if key in x:
                x = x[key]
                break
    x = torch.as_tensor(x).double().cpu().numpy()
    if x.ndim != 2:
        raise ValueError(f'Language file must contain the final 2-D class representation [classes, dim], got {x.shape} from {path}')
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError(f'Language representation contains zero-norm rows: {path}')
    return x / norm

def load_labels(path: str) -> np.ndarray:
    y = torch.load(path, map_location='cpu')
    if isinstance(y, dict):
        for key in ('labels', 'y', 'targets'):
            if key in y:
                y = y[key]
                break
    y = torch.as_tensor(y).long().cpu().numpy().reshape(-1)
    return y

def vision_prototypes(V: np.ndarray, y: np.ndarray, seed: int | None=None, frac: float=0.5) -> np.ndarray:
    if len(V) != len(y):
        raise ValueError(f'Vision samples ({len(V)}) and labels ({len(y)}) have different lengths')
    if not 0 < frac <= 1:
        raise ValueError('frac must lie in (0, 1]')
    if frac < 1:
        if seed is None:
            raise ValueError('A seed is required when frac < 1')
        g = torch.Generator().manual_seed(int(seed))
        idx = torch.randperm(len(y), generator=g)[:int(len(y) * frac)].numpy()
        (X, ys) = (V[idx], y[idx])
    else:
        (X, ys) = (V, y)
    classes = np.unique(y)
    P = np.stack([X[ys == c].mean(0) for c in classes])
    norm = np.linalg.norm(P, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError('A vision class prototype has zero norm')
    return P / norm

def _standardize_offdiag(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    out = np.zeros_like(K, dtype=np.float64)
    sd = K[mask].std()
    if sd <= 0:
        return out
    out[mask] = (K[mask] - K[mask].mean()) / sd
    return out

def rankalign_matrices(Px, Py, center: bool=True, representation: str='full'):

    def transform(P):
        X = np.asarray(P, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f'Expected class prototypes [N, d], got {X.shape}')
        if center:
            X = X - X.mean(axis=0, keepdims=True)
            norm = np.linalg.norm(X, axis=1, keepdims=True)
            if np.any(norm == 0):
                raise ValueError('Centered prototype contains a zero-norm row')
            X = X / norm
        K = X @ X.T
        n = K.shape[0]
        if n < 2:
            raise ValueError('RankAlign requires at least two classes')
        off = ~np.eye(n, dtype=bool)
        R = np.zeros_like(K, dtype=np.float64)
        R[off] = rankdata(K[off], method='max') / off.sum()
        if representation == 'rank':
            C = np.zeros_like(R)
            C[off] = R[off] - R[off].mean()
            scale = C.std()
            return C if scale <= 0 else C / scale
        if representation != 'full':
            raise ValueError("representation must be 'full' or 'rank'")
        M = n * (n - 1) // 2
        G = np.zeros_like(R)
        G[off] = -np.log1p(-(R[off] - 0.5 / M))
        G = _standardize_offdiag(G)
        Rrow = np.zeros_like(K, dtype=np.float64)
        for i in range(n):
            row_mask = np.arange(n) != i
            Rrow[i, row_mask] = rankdata(K[i, row_mask], method='max') / (n - 1)
        L = np.zeros_like(Rrow)
        L[off] = -np.log1p(-(Rrow[off] - 0.5 / (n - 1)))
        L = _standardize_offdiag((L + L.T) / 2.0)
        return _standardize_offdiag(G + L)

    def side(P):
        if isinstance(P, (list, tuple)):
            return _standardize_offdiag(np.sum([transform(p) for p in P], axis=0))
        return transform(P)
    return (side(Px), side(Py))
