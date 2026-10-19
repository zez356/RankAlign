from __future__ import annotations
import numpy as np
from scipy.optimize import quadratic_assignment
try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
if HAVE_NUMBA:

    @njit(cache=True)
    def _refine_core(A, B, p0, iters, seed):
        np.random.seed(seed)
        n = p0.shape[0]
        p = p0.copy()
        Bp = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                Bp[i, j] = B[p[i], p[j]]
        f = 0.0
        for i in range(n):
            for j in range(n):
                f += A[i, j] * Bp[i, j]
        D = np.zeros((n, n))
        for r in range(n):
            for s in range(r + 1, n):
                t = 0.0
                for k in range(n):
                    if k != r and k != s:
                        t += (A[r, k] - A[s, k]) * (Bp[s, k] - Bp[r, k])
                d = 2.0 * t
                D[r, s] = d
                D[s, r] = d
        best_f = f
        best_p = p.copy()
        memory = np.zeros((n, n), dtype=np.int64)
        lo = int(0.9 * n)
        hi = int(1.1 * n) + 1
        for it in range(iters):
            bv = -1e+300
            br = -1
            bs = -1
            for r in range(n):
                for s in range(r + 1, n):
                    dv = D[r, s]
                    blocked = memory[r, p[s]] > it and memory[s, p[r]] > it
                    if blocked and (not f + dv > best_f + 1e-12):
                        continue
                    if dv > bv:
                        bv = dv
                        br = r
                        bs = s
            if br < 0:
                continue
            r = br
            s = bs
            memory[r, p[r]] = it + np.random.randint(lo, hi)
            memory[s, p[s]] = it + np.random.randint(lo, hi)
            f += bv
            tmp = p[r]
            p[r] = p[s]
            p[s] = tmp
            for k in range(n):
                tmp2 = Bp[r, k]
                Bp[r, k] = Bp[s, k]
                Bp[s, k] = tmp2
            for k in range(n):
                tmp2 = Bp[k, r]
                Bp[k, r] = Bp[k, s]
                Bp[k, s] = tmp2
            for u in range(n):
                if u == r or u == s:
                    continue
                xu = A[r, u] - A[s, u]
                yu = Bp[s, u] - Bp[r, u]
                for v in range(u + 1, n):
                    if v == r or v == s:
                        continue
                    xv = A[r, v] - A[s, v]
                    yv = Bp[s, v] - Bp[r, v]
                    inc = 2.0 * (xu - xv) * (yu - yv)
                    D[u, v] += inc
                    D[v, u] += inc
            for q in (r, s):
                for v in range(n):
                    if v == q:
                        continue
                    t = 0.0
                    for k in range(n):
                        if k != q and k != v:
                            t += (A[q, k] - A[v, k]) * (Bp[v, k] - Bp[q, k])
                    d = 2.0 * t
                    D[q, v] = d
                    D[v, q] = d
            if f > best_f + 1e-12:
                best_f = f
                best_p = p.copy()
        return best_p

def refine_nb(A, B, p0, iters=None, seed=0):
    n = len(p0)
    iters = iters if iters is not None else 40 * n
    p = _refine_core(np.ascontiguousarray(A, dtype=np.float64), np.ascontiguousarray(B, dtype=np.float64), np.asarray(p0, dtype=np.int64), int(iters), int(seed) % 2 ** 31)
    return (p, float((A * B[np.ix_(p, p)]).sum()))
_STARTS = 64
_KEEP = 16
_REFINE_MULT = 20
_ROUNDS = 40

def objective(A, B, p):
    p = np.asarray(p, dtype=int)
    return float((A * B[np.ix_(p, p)]).sum())

def _swap_deltas(A, B, p):
    Bp = B[np.ix_(p, p)]
    M1 = A @ Bp.T
    M2 = A.T @ Bp
    (d1, d2) = (np.diag(M1), np.diag(M2))
    D = M1 + M1.T - d1[:, None] - d1[None, :] + M2 + M2.T - d2[:, None] - d2[None, :] + A * Bp + A.T * Bp.T + A * Bp.T + A.T * Bp
    np.fill_diagonal(D, -np.inf)
    return D

def _row_deltas(A, Bp, r):
    T = ((A - A[r]) * (Bp[r] - Bp)).sum(1)
    d = 2.0 * (T + 2.0 * A[r] * Bp[r])
    d[r] = -np.inf
    return d

def _refine_numpy(A, B, p0, iters, seed):
    n = len(p0)
    rng = np.random.default_rng(seed)
    p = np.array(p0, copy=True)
    Bp = B[np.ix_(p, p)].copy()
    D = _swap_deltas(A, B, p)
    f = float((A * Bp).sum())
    (best_p, best_f) = (p.copy(), f)
    memory = np.zeros((n, n), dtype=np.int64)
    (lo, hi) = (int(0.9 * n), int(1.1 * n) + 1)
    iu = np.triu_indices(n, 1)
    (rr, ss) = iu
    for it in range(iters):
        dv = D[iu]
        blocked = (memory[rr, p[ss]] > it) & (memory[ss, p[rr]] > it)
        allowed = ~blocked | (f + dv > best_f + 1e-12)
        if not allowed.any():
            continue
        k = int(np.argmax(np.where(allowed, dv, -np.inf)))
        (r, s) = (rr[k], ss[k])
        memory[r, p[r]] = it + rng.integers(lo, hi)
        memory[s, p[s]] = it + rng.integers(lo, hi)
        f += dv[k]
        (p[r], p[s]) = (p[s], p[r])
        Bp[[r, s]] = Bp[[s, r]]
        Bp[:, [r, s]] = Bp[:, [s, r]]
        X = A[r] - A[s]
        Y = Bp[s] - Bp[r]
        D += 2.0 * np.subtract.outer(X, X) * np.subtract.outer(Y, Y)
        (dr, ds) = (_row_deltas(A, Bp, r), _row_deltas(A, Bp, s))
        D[r] = dr
        D[:, r] = dr
        D[s] = ds
        D[:, s] = ds
        np.fill_diagonal(D, -np.inf)
        if f > best_f + 1e-12:
            (best_f, best_p) = (f, p.copy())
    return (best_p, objective(A, B, best_p))

def _refine(A, B, p0, iters, seed):
    if HAVE_NUMBA:
        return refine_nb(A, B, p0, iters=iters, seed=seed)
    return _refine_numpy(A, B, p0, iters=iters, seed=seed)

def _faq_pool(A, B, count, rng):
    pool = []
    for t in range(count):
        res = quadratic_assignment(A, B, method='faq', options=dict(maximize=True, P0='barycenter' if t == 0 else 'randomized', rng=rng, maxiter=100, tol=1e-06))
        pool.append(res.col_ind)
    return pool

def initial_guess(stages, seed=0):
    rng = np.random.default_rng(seed)
    n = stages[0][0].shape[0]
    refine_iters = _REFINE_MULT * n
    pool = []
    for (si, (A, B)) in enumerate(stages):
        count = _STARTS if si == 0 else max(1, _STARTS // 4)
        starts = pool + _faq_pool(A, B, count, rng)
        scored = sorted(((objective(A, B, p), tuple(p)) for p in starts), reverse=True)
        (uniq, seen) = ([], set())
        for (_, p) in scored:
            if p not in seen:
                seen.add(p)
                uniq.append(np.asarray(p, dtype=int))
            if len(uniq) >= _KEEP:
                break
        pool = [_refine(A, B, p, refine_iters, seed + 7 * si + k)[0] for (k, p) in enumerate(uniq)]
    (A, B) = stages[-1]
    scores = [objective(A, B, p) for p in pool]
    best = pool[int(np.argmax(scores))]
    best_f = float(max(scores))
    for r in range(_ROUNDS):
        p = best.copy()
        k = max(3, n // 8)
        idx = rng.choice(n, k, replace=False)
        p[idx] = p[rng.permutation(idx)]
        (q, g) = _refine(A, B, p, refine_iters, seed + 1000 + r)
        if g > best_f + 1e-12:
            (best, best_f) = (q, g)
    return (best, best_f)
SOLVERS = ('faq', 'ot', 'fhg')

def _perm_matrix(p):
    n = len(p)
    P = np.zeros((n, n), dtype=np.float64)
    P[np.arange(n), p] = 1.0
    return P

def _solve_faq(A, B, p0):
    from scipy.optimize import quadratic_assignment
    res = quadratic_assignment(-A, B, method='faq', options={'maximize': False, 'P0': _perm_matrix(p0)})
    return (np.asarray(res.col_ind, dtype=int), {})

def _solve_ot(A, B, p0):
    from ot import gromov
    from scipy.optimize import linear_sum_assignment
    (T, _) = gromov.entropic_gromov_wasserstein(A, B, loss_fun='square_loss', symmetric=True, verbose=False, log=True, G0=_perm_matrix(p0) / len(p0), amijo=True, max_iter=1000.0, tol_rel=1e-10, tol_abs=1e-10, numItermax=100)
    (_, perm) = linear_sum_assignment(T, maximize=True)
    return (np.asarray(perm, dtype=int), {})

def _solve_fhg(A, B, p0, timelimit):
    import torch
    from fhg import FactorizedHahnGrant
    A_t = -torch.as_tensor(A)
    (perm, optimal, cost, bound) = FactorizedHahnGrant().solve(cost1=A_t, cost2=torch.as_tensor(B).clone(), constant=torch.zeros((), dtype=A_t.dtype), timelimit=timelimit, init_permutation=torch.as_tensor(np.asarray(p0)))
    perm = np.asarray(perm.cpu().numpy() if hasattr(perm, 'cpu') else perm, dtype=int)
    return (perm, {'Optimal': bool(optimal), 'Cost': float(cost), 'Bound': float(bound) if bound is not None else None})

def solve_backend(name, A, B, p0, timelimit):
    if name == 'faq':
        return _solve_faq(A, B, p0)
    if name == 'ot':
        return _solve_ot(A, B, p0)
    if name == 'fhg':
        return _solve_fhg(A, B, p0, timelimit)
    raise ValueError(name)

def _local_search(C, S):
    n = C.shape[0]
    in_set = np.zeros(n, dtype=bool)
    in_set[S] = True
    while True:
        contrib = C[:, in_set].sum(1)
        (inside, outside) = (np.where(in_set)[0], np.where(~in_set)[0])
        delta = contrib[outside][None, :] - C[np.ix_(inside, outside)] - contrib[inside][:, None]
        k = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[k] >= -1e-12:
            idx = np.sort(inside)
            return (idx, float(C[np.ix_(idx, idx)].sum()))
        in_set[inside[k[0]]] = False
        in_set[outside[k[1]]] = True

def p_dispersion_subset(C, N, restarts=50, seed=0):
    n = C.shape[0]
    if N >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    (best, best_value) = (None, np.inf)
    for r in range(restarts):
        if r == 0:
            S = [int(np.argmin(C.sum(1)))]
            while len(S) < N:
                c = C[:, S].sum(1)
                c[S] = np.inf
                S.append(int(np.argmin(c)))
            S = np.asarray(S)
        else:
            S = rng.choice(n, N, replace=False)
        (S, value) = _local_search(C, S)
        if value < best_value - 1e-12:
            (best, best_value) = (S, value)
    return best
