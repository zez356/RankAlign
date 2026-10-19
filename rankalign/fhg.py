# Adapted from itsamatch (https://github.com/dominik-schnaus/itsamatch),
# the code accompanying Schnaus et al., "It's a (Blind) Match!", CVPR 2025.
# Original license:
#
# MIT License
#
# Copyright (c) 2025 Dominik Schnaus
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import itertools
import operator
from time import time
import numpy as np
import torch
from scipy._lib._util import check_random_state
from scipy.optimize import OptimizeResult, quadratic_assignment
from scipy.optimize._optimize import _check_unknown_options
from torch import Tensor
try:
    from lapjv import lapjv
    HAVE_LAPJV = True
except ImportError:
    HAVE_LAPJV = False

def _calc_score_faster(A, B, perm, lin, const):
    out = np.sum(A * B[perm][:, perm]) + const
    if lin is not None:
        out += lin[np.arange(len(A)), perm].sum()
    return out

def rank(a):
    b = np.empty_like(a, dtype=np.int64)
    b[a.argsort()] = np.arange(a.shape[0], dtype=np.int64)
    return b

def _quadratic_assignment_2opt(A, B, linear, constant, maximize=False, rng=None, partial_guess=None, **unknown_options):
    _check_unknown_options(unknown_options)
    rng = check_random_state(rng)
    N = A.shape[0]
    if partial_guess is None:
        partial_guess = np.array([[], []]).T
    partial_guess = np.atleast_2d(partial_guess).astype(int)
    msg = None
    if partial_guess.shape[0] > A.shape[0]:
        msg = '`partial_guess` can have only as many entries as there are nodes'
    elif partial_guess.shape[1] != 2:
        msg = '`partial_guess` must have two columns'
    elif partial_guess.ndim != 2:
        msg = '`partial_guess` must have exactly two dimensions'
    elif (partial_guess < 0).any():
        msg = '`partial_guess` must contain only positive indices'
    elif (partial_guess >= len(A)).any():
        msg = '`partial_guess` entries must be less than number of nodes'
    elif not len(set(partial_guess[:, 0])) == len(partial_guess[:, 0]) or not len(set(partial_guess[:, 1])) == len(partial_guess[:, 1]):
        msg = '`partial_guess` column entries must be unique'
    if msg is not None:
        raise ValueError(msg)
    if partial_guess.size:
        guess_rows = np.zeros(N, dtype=bool)
        guess_cols = np.zeros(N, dtype=bool)
        perm = np.zeros(N, dtype=int)
        (rg, cg) = partial_guess.T
        guess_rows[rg] = True
        guess_cols[cg] = True
        perm[guess_rows] = cg
        random_rows = ~guess_rows
        random_cols = ~guess_cols
        perm[random_rows] = rng.permutation(np.arange(N)[random_cols])
    else:
        perm = rng.permutation(np.arange(N))
    best_score = _calc_score_faster(A, B, perm, linear, constant)
    i_free = np.arange(perm.shape[0])
    better = operator.gt if maximize else operator.lt
    n_iter = 0
    done = False
    while not done:
        for (i, j) in itertools.combinations(i_free, 2):
            n_iter += 1
            (perm[i], perm[j]) = (perm[j], perm[i])
            score = _calc_score_faster(A, B, perm, linear, constant)
            if better(score, best_score):
                best_score = score
                break
            (perm[i], perm[j]) = (perm[j], perm[i])
        else:
            done = True
    res = {'col_ind': perm, 'fun': best_score, 'nit': n_iter}
    return OptimizeResult(res)
if __name__ == '__main__':
    import pytest
    from scipy.optimize import quadratic_assignment
    num_seeds = 1000
    dim = 10
    np.random.seed(0)
    seeds = np.random.randint(0, 2 ** 32 - 1, num_seeds)
    for seed in seeds:
        np.random.seed(seed)
        A = np.random.rand(dim, dim)
        B = np.random.rand(dim, dim)
        partial_match_size = np.random.randint(0, dim)
        partial_match_0 = np.random.permutation(dim)[:partial_match_size]
        sorting = np.argsort(partial_match_0)
        partial_match_1 = np.random.permutation(dim)[:partial_match_size]
        partial_match = np.stack([partial_match_0, partial_match_1], axis=1)[sorting]
        maximize = bool(np.random.randint(0, 2))
        res = _quadratic_assignment_2opt(A, B, maximize=maximize, partial_match=partial_match, rng=seed)
        res2 = quadratic_assignment(A, B, method='2opt', options=dict(maximize=maximize, partial_match=partial_match, rng=seed))
        assert res['fun'] == pytest.approx(res2['fun']), (res['fun'], res2['fun'])

def lapjv_interface(cost_matrix: Tensor):
    (col_ind, _, (_, u, v)) = lapjv(cost_matrix.cpu().numpy(), force_doubles=False)
    col_ind = torch.as_tensor(col_ind, dtype=torch.long, device=cost_matrix.device)
    row_ind = torch.arange(cost_matrix.shape[0], dtype=col_ind.dtype, device=cost_matrix.device)
    u = torch.as_tensor(u, dtype=cost_matrix.dtype, device=cost_matrix.device)
    v = torch.as_tensor(v, dtype=cost_matrix.dtype, device=cost_matrix.device)
    return (u, v, row_ind, col_ind)

def hungarian_matching(C, **kwargs):
    inf = torch.inf if C.dtype.is_floating_point else torch.iinfo(C.dtype).max
    (m, n) = C.shape
    transpose = False
    if m > n:
        C = C.T
        (m, n) = (n, m)
        transpose = True
    u = torch.zeros(m, dtype=C.dtype, device=C.device)
    v = torch.zeros(n, dtype=C.dtype, device=C.device)
    col4row = -torch.ones(m, dtype=torch.long)
    row4col = -torch.ones(n, dtype=torch.long)
    AC = set(range(n))
    path = torch.zeros(n, dtype=torch.long)
    for cur_row in range(m):
        shortest_path_cost = inf * torch.ones(n, dtype=C.dtype, device=C.device)
        (SR, SC) = (set(), set())
        (sink, min_val, i) = (-1, 0, cur_row)
        while sink == -1:
            SR = SR.union({i})
            (lowest, j_out) = (inf, -1)
            for j in list(AC.difference(SC)):
                if C[i, j] == inf:
                    continue
                reduced_cost = min_val + C[i, j] - u[i] - v[j]
                if reduced_cost < shortest_path_cost[j]:
                    shortest_path_cost[j] = reduced_cost
                    path[j] = i
                if shortest_path_cost[j] < lowest or (shortest_path_cost[j] == lowest and row4col[j] == -1):
                    lowest = shortest_path_cost[j]
                    j_out = j
            min_val = lowest
            j = j_out
            if min_val == inf:
                return None
            SC = SC.union({j})
            if row4col[j] == -1:
                sink = j
            else:
                i = row4col[j].item()
        u[cur_row] += min_val
        for i_sr in SR.difference({cur_row}):
            u[i_sr] += min_val - shortest_path_cost[col4row[i_sr]]
        for j_sc in SC:
            v[j_sc] -= min_val - shortest_path_cost[j_sc]
        j = sink
        while True:
            i = path[j].item()
            row4col[j] = i
            temp = col4row[i].item()
            col4row[i] = j
            if i == cur_row:
                break
            j = temp
    if transpose:
        return (v, u, col4row[col4row.argsort()], col4row.argsort())
    return (u, v, torch.arange(m).to(col4row), col4row)

def update_best_permutation(best_cost, best_permutation, row_ind, col_ind, source_kernel, target_kernel, first_order_constant, constant, eps=1e-08):
    cost = (source_kernel[row_ind][:, row_ind] * target_kernel[col_ind][:, col_ind]).sum() + first_order_constant[row_ind, col_ind].sum() + constant
    if cost + eps < best_cost:
        return (cost, col_ind[row_ind.argsort()])
    return (best_cost, best_permutation)

class FactorizedHahnGrant:

    def __init__(self, matching_algorithm=None, max_steps_without_improvement=3, epsilon=1e-06, initialize_with_2opt_faq=True, num_repetitions_2opt_faq=100, update_with_lap_solutions=True):
        if matching_algorithm is None:
            matching_algorithm = 'lapjv' if HAVE_LAPJV else 'hungarian'
        self.matching_algorithm = {'lapjv': lapjv_interface, 'hungarian': hungarian_matching}[matching_algorithm]
        self.max_steps_without_improvement = max_steps_without_improvement
        self.epsilon = epsilon
        self.initialize_with_2opt_faq = initialize_with_2opt_faq
        self.num_repetitions_2opt_faq = num_repetitions_2opt_faq
        self.update_with_lap_solutions = update_with_lap_solutions

    def solve(self, cost1: Tensor, cost2: Tensor, linear: Tensor=None, constant: Tensor=None, timelimit: float | None=None, init_permutation: Tensor | None=None):
        if linear is None:
            linear = torch.zeros_like(cost1)
        if constant is None:
            constant = torch.zeros(1, dtype=cost1.dtype, device=cost1.device)
        if timelimit is None:
            timelimit = float('inf')
        n = cost1.shape[0]
        (cost1_min, cost1_sum) = (cost1.min(), cost1.sum())
        (cost2_min, cost2_sum) = (cost2.min(), cost2.sum())
        cost1 -= cost1_min
        cost2 -= cost2_min
        constant -= -cost2_min * cost1_sum - cost1_min * cost2_sum + cost1_min * cost2_min * n ** 2
        leaders = torch.outer(torch.diag(cost1), torch.diag(cost2)) + linear
        start_time = time()
        is_floating_point = cost1.dtype.is_floating_point or cost2.dtype.is_floating_point
        eps = self.epsilon if is_floating_point else 0
        if is_floating_point:
            if cost1.dtype.is_floating_point:
                cost2 = cost2.to(cost1)
            else:
                cost1 = cost1.to(cost2)
        else:
            cost2 = cost2.to(cost1)
        superleader = constant.clone()
        us = torch.zeros(n, n, n - 1, dtype=cost1.dtype, device=cost1.device)
        vs = torch.zeros(n, n, n - 1, dtype=cost1.dtype, device=cost1.device)
        best_permutation = None
        best_cost = torch.inf * torch.ones(1, dtype=cost1.dtype, device=cost1.device)[0]
        if init_permutation is not None:
            (best_cost, best_permutation) = update_best_permutation(best_cost, best_permutation, torch.arange(n), torch.as_tensor(init_permutation, dtype=torch.long), cost1, cost2, linear, constant)
        steps_without_improvement = 0
        optimal = False
        iteration = 0
        while True:
            iteration += 1
            (u, v, row_ind, col_ind) = self.matching_algorithm(leaders)
            if self.update_with_lap_solutions:
                (best_cost, best_permutation) = update_best_permutation(best_cost, best_permutation, row_ind, col_ind, cost1, cost2, linear, constant)
                if time() - start_time > timelimit:
                    break
            if iteration == 1 and self.initialize_with_2opt_faq:
                cost1_numpy = cost1.cpu().numpy()
                cost2_numpy = cost2.cpu().numpy()
                for i in range(self.num_repetitions_2opt_faq):
                    if i == 0 and best_permutation is not None:
                        partial_guess = torch.stack([torch.arange(n, dtype=torch.long, device=cost1.device), best_permutation], dim=1).cpu().numpy()
                    else:
                        partial_guess = None
                    subseed = torch.randint(0, 2 ** 32 - 1, (1,), dtype=torch.long).item()
                    res = quadratic_assignment(cost1_numpy, cost2_numpy, method='faq', options={'maximize': False, 'rng': subseed})
                    (best_cost, best_permutation) = update_best_permutation(best_cost, best_permutation, torch.arange(n, dtype=torch.long, device=cost1.device), torch.as_tensor(res.col_ind, dtype=torch.long, device=cost1.device), cost1, cost2, linear, constant)
                    if time() - start_time > timelimit:
                        break
                    res = _quadratic_assignment_2opt(cost1_numpy, cost2_numpy, linear.cpu().numpy(), constant.cpu().numpy(), maximize=False, rng=subseed, partial_guess=partial_guess)
                    if float(res.fun) < float(best_cost.item()):
                        best_cost = torch.as_tensor([float(res.fun)], dtype=torch.float, device=cost1.device).sum()
                        best_permutation = torch.as_tensor(res.col_ind, dtype=torch.long, device=cost1.device)
                    if time() - start_time > timelimit:
                        break
            if time() - start_time > timelimit:
                break
            improvement = u.sum() + v.sum()
            superleader += improvement
            leaders -= u[:, None] + v[None, :]
            if iteration > 1 and best_cost - superleader < max(eps, eps * best_cost.abs()):
                optimal = True
                break
            if is_floating_point:
                us -= leaders[:, :, None] / (n - 1)
            else:
                us -= leaders[:, :, None] // (n - 1)
                for i in range(n):
                    for j in range(n):
                        us[i, j, :leaders[i, j] % (n - 1)] -= 1
            for index in leaders.view(-1).argsort():
                i = index // n
                j = index % n
                valid_rows = torch.where(torch.arange(n) != i)[0]
                valid_cols = torch.where(torch.arange(n) != j)[0]
                us_selected = us[valid_rows, :][:, valid_cols]
                vs_selected = vs[valid_rows, :][:, valid_cols]
                arange = torch.arange(n - 1)
                row_selected_i = i - (arange < i).long()
                col_selected_j = j - (arange < j).long()
                cost_matrix = 2 * torch.outer(cost1[valid_rows, i], cost2[valid_cols, j]) - us[i, j, :, None] - vs[i, j, None, :] - us_selected[arange, :, row_selected_i] - vs_selected[:, arange, col_selected_j]
                (u, v, row_ind, col_ind) = self.matching_algorithm(cost_matrix)
                row_ind[row_ind >= i] += 1
                col_ind[col_ind >= j] += 1
                row_ind_full = torch.cat([row_ind, torch.tensor([i], dtype=row_ind.dtype, device=row_ind.device)])
                col_ind_full = torch.cat([col_ind, torch.tensor([j], dtype=col_ind.dtype, device=col_ind.device)])
                if self.update_with_lap_solutions:
                    (best_cost, best_permutation) = update_best_permutation(best_cost, best_permutation, row_ind_full, col_ind_full, cost1, cost2, linear, constant)
                us[i, j] += u
                vs[i, j] += v
                if time() - start_time > timelimit:
                    break
                leaders[i, j] = u.sum() + v.sum()
            if improvement < min(eps, eps * superleader.abs(), eps * best_cost.abs()):
                steps_without_improvement += 1
            else:
                steps_without_improvement = 0
            if steps_without_improvement > self.max_steps_without_improvement or time() - start_time > timelimit:
                break
        (u, v, row_ind, col_ind) = self.matching_algorithm(leaders)
        (best_cost, best_permutation) = update_best_permutation(best_cost, best_permutation, row_ind, col_ind, cost1, cost2, linear, constant)
        return (best_permutation, optimal, best_cost.item(), superleader.item())
