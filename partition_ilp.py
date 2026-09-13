"""
Exact corner partition for a k-pursuer team, as a mixed-integer program.

Decision: assign every reflex corner to one of k pursuers, and for every
evader grid position choose where each pursuer stands. Two things matter:

  speed   s_k  — the team critical speed ratio: the largest alpha any
                 pursuer must accept for one of its own corners, over all
                 evader positions (smaller is better; corner_groups.py)
  sight   Lambda — line of sight at the position the pursuer actually uses:
                 the nearness-weighted fraction of (evader position, corner)
                 pairs for which the responsible pursuer, parked at its
                 chosen spot, keeps sight of the evader for the whole escape
                 to that corner (the paper's LOS score with sigma_c = 1 for a
                 stationary pursuer; larger is better)

Both are linear in binaries once the pursuer's candidate positions are
discretised to a finite set Q on the roadmap (vertices plus samples along
every edge). Constants, precomputed once:
  alpha[e, q, c] = d_G(q, c) / L_c(e)              (tent construction)
  vis[e, c, q]   = 1 iff q lies in V(e, c)         (roadmap_los, exact)
  w[e, c]        = (1/L_c) / sum_c' (1/L_c')       (nearness weights)

Model (epsilon-constraint form):

  x[c,i]   in {0,1}   corner c assigned to pursuer i;   sum_i x[c,i] = 1
  z[e,i,q] in {0,1}   pursuer i stands at q, evader at e;  sum_q z[e,i,q] = 1
  v[e,i,c] in {0,1}   pursuer i is responsible for c AND sees the whole
                      escape to c from where it stands

  speed  for every (e,i,q): with B = {c : alpha[e,q,c] > t},
            sum_{c in B} x[c,i] + |B| z[e,i,q] <= |B|
  sight  v[e,i,c] <= x[c,i],   v[e,i,c] <= sum_{q : vis[e,c,q]} z[e,i,q]
  symmetry: the first corner goes to pursuer 1; no pursuer is idle

  maximise  (1/|E|) sum_{e,i,c} w[e,c] v[e,i,c]     subject to the speed bound t

Bisection on t gives the smallest feasible bound s_k^Q (exact over Q; an
upper bound on the continuous s_k of the returned partition, which
corner_groups.evaluate_groups re-measures on the fine grid). Solving at
larger t traces the speed/sight Pareto front. The heuristic clustering of
corner_groups.py is only used as a comparison point and as a fallback for
instances too large to certify.

Solver: scipy.optimize.milp (HiGHS), already used for the guard set cover.
"""
from dataclasses import dataclass, field
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from shapely.geometry import Point

from corner_groups import grid_path_lengths
from roadmap_los import VisCache, persistent_visible_all_corners

_ON_LINE_TOL = 0.05   # a candidate position counts as inside V(e, c) within this distance


# ---------------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------------
def candidate_positions(data, spacing: float) -> list:
    """Roadmap positions: every dense-graph vertex plus samples every
    `spacing` units along every edge. Returns [(v1, v2, offset, elen)]."""
    Q, seen_edges, seen_vertices = [], set(), set()
    for v1, v2 in data.total_edges:
        key = (min(v1, v2), max(v1, v2))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        elen = data.vertices[v1].distance(data.vertices[v2])
        for v, off in ((v1, 0.0), (v2, elen)):
            if v not in seen_vertices:
                seen_vertices.add(v)
                Q.append((v1, v2, off, elen))
        n = int(elen // spacing)
        for j in range(1, n + 1):
            x = j * spacing
            if 1e-6 < x < elen - 1e-6:
                Q.append((v1, v2, x, elen))
    return Q


def position_xy(data, q) -> tuple:
    v1, v2, x, elen = q
    a, b = data.vertices[v1], data.vertices[v2]
    f = 0.0 if elen < 1e-9 else x / elen
    return (a.x + f * (b.x - a.x), a.y + f * (b.y - a.y))


def alpha_table(data, Q: list, grid_pl: list) -> np.ndarray:
    """alpha[e, q, c] = d_G(q, c) / L_c(e), same construction as the guard
    optimiser's tents (ker_pipeline._edge_tent_store) at fixed offsets."""
    corners = list(data.corners)
    dG = np.full((len(Q), len(corners)), np.inf)
    for qi, (v1, v2, x, elen) in enumerate(Q):
        for ci, c in enumerate(corners):
            dG[qi, ci] = min(data.vectors_org[v1][c] + x, data.vectors_org[v2][c] + elen - x)
    A = np.empty((len(grid_pl), len(Q), len(corners)))
    for ei, (_, _, pl) in enumerate(grid_pl):
        A[ei] = dG / np.array([pl[c] for c in corners])
    return A


def sight_table(data, Q: list, grid_pts: list, cache: VisCache) -> np.ndarray:
    """vis[e, c, q] = True iff candidate position q lies in V(e, c)."""
    corners = list(data.corners)
    pts = [Point(position_xy(data, q)) for q in Q]
    vis = np.zeros((len(grid_pts), len(corners), len(Q)), dtype=bool)
    for ei, (x, y) in enumerate(grid_pts):
        V = persistent_visible_all_corners(x, y, data, cache)
        for ci, c in enumerate(corners):
            g = V[c]
            if g.is_empty:
                continue
            for qi, p in enumerate(pts):
                vis[ei, ci, qi] = g.distance(p) < _ON_LINE_TOL
    return vis


def nearness_weights(data, grid_pl: list) -> np.ndarray:
    """w[e, c] = (1/L_c) / sum_c' (1/L_c'), as in los_score.nearness_weights."""
    corners = list(data.corners)
    W = np.empty((len(grid_pl), len(corners)))
    for ei, (_, _, pl) in enumerate(grid_pl):
        inv = np.array([1.0 / pl[c] for c in corners])
        W[ei] = inv / inv.sum()
    return W


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class ILPResult:
    t: float                     # speed bound used (then overwritten with the achieved s_k^Q)
    feasible: bool
    groups: list = field(default_factory=list)
    sight: float = 0.0           # Lambda in [0, 1]
    solve_time: float = 0.0
    status: str = ''
    positions: dict = field(default_factory=dict)   # (e, i) -> (x, y) chosen position


class PartitionILP:
    def __init__(self, data, k: int, grid_n: int = 8, spacing: float = 15.0, log=print):
        self.data, self.k, self.log = data, k, log
        t0 = time.time()
        self.grid_pl = grid_path_lengths(data, grid_n)
        self.pts = [(x, y) for x, y, _ in self.grid_pl]
        self.Q = candidate_positions(data, spacing)
        self.A = alpha_table(data, self.Q, self.grid_pl)
        self.vis = sight_table(data, self.Q, self.pts, VisCache(data))
        self.W = nearness_weights(data, self.grid_pl)
        self.nC, self.nE, self.nQ = len(data.corners), len(self.pts), len(self.Q)
        log(f'[ILP] grid {self.nE} evader pts, {self.nQ} candidate positions, '
            f'{self.nC} corners, k={k}; setup {time.time() - t0:.1f}s')

    # variable indexing
    def _ix(self, c, i):    return c * self.k + i
    def _iz(self, e, i, q): return self.nC * self.k + (e * self.k + i) * self.nQ + q
    def _iv(self, e, i, c): return self.nC * self.k + self.nE * self.k * self.nQ + (e * self.k + i) * self.nC + c
    @property
    def n_vars(self):       return self.nC * self.k + self.nE * self.k * (self.nQ + self.nC)

    def solve(self, t: float, use_sight: bool = True, time_limit: float = 600.0) -> ILPResult:
        k, nC, nE, nQ = self.k, self.nC, self.nE, self.nQ
        rows, cols, vals, lb, ub = [], [], [], [], []
        r = 0

        def add_row(idx, coef, lo, hi):
            nonlocal r
            rows.extend([r] * len(idx)); cols.extend(idx); vals.extend(coef)
            lb.append(lo); ub.append(hi); r += 1

        for c in range(nC):                                   # one pursuer per corner
            add_row([self._ix(c, i) for i in range(k)], [1.0] * k, 1.0, 1.0)
        for i in range(k):                                    # no idle pursuer
            add_row([self._ix(c, i) for c in range(nC)], [1.0] * nC, 1.0, np.inf)
        add_row([self._ix(0, 0)], [1.0], 1.0, 1.0)            # symmetry
        for e in range(nE):                                   # one position per (e, i)
            for i in range(k):
                add_row([self._iz(e, i, q) for q in range(nQ)], [1.0] * nQ, 1.0, 1.0)
        for e in range(nE):                                   # speed bound
            bad = self.A[e] > t                               # [q, c]
            for q in range(nQ):
                B = np.nonzero(bad[q])[0]
                if len(B) == 0:
                    continue
                for i in range(k):
                    if len(B) == nC:
                        add_row([self._iz(e, i, q)], [1.0], 0.0, 0.0)
                    else:
                        add_row([self._ix(int(c), i) for c in B] + [self._iz(e, i, q)],
                                [1.0] * len(B) + [float(len(B))], -np.inf, float(len(B)))
        obj = np.zeros(self.n_vars)
        if use_sight:
            for e in range(nE):
                for i in range(k):
                    for c in range(nC):
                        iv = self._iv(e, i, c)
                        obj[iv] = -self.W[e, c] / nE
                        add_row([iv, self._ix(c, i)], [1.0, -1.0], -np.inf, 0.0)
                        good = np.nonzero(self.vis[e, c])[0]
                        add_row([iv] + [self._iz(e, i, int(q)) for q in good],
                                [1.0] + [-1.0] * len(good), -np.inf, 0.0)
        Acon = coo_matrix((vals, (rows, cols)), shape=(r, self.n_vars)).tocsr()
        hi = np.ones(self.n_vars)
        if not use_sight:
            hi[self.nC * k + nE * k * nQ:] = 0.0              # pin unused v
        t0 = time.time()
        res = milp(c=obj, constraints=LinearConstraint(Acon, lb, ub),
                   integrality=np.ones(self.n_vars), bounds=Bounds(np.zeros(self.n_vars), hi),
                   options={'time_limit': time_limit, 'disp': False})
        dt = time.time() - t0
        if res.status != 0 or res.x is None:
            return ILPResult(t=t, feasible=False, solve_time=dt, status=res.message)
        x = res.x
        groups = [[] for _ in range(k)]
        for ci, c in enumerate(self.data.corners):
            groups[int(np.argmax([x[self._ix(ci, j)] for j in range(k)]))].append(c)
        positions = {}
        for e in range(nE):
            for i in range(k):
                q = int(np.argmax([x[self._iz(e, i, qq)] for qq in range(nQ)]))
                positions[(e, i)] = position_xy(self.data, self.Q[q])
        out = ILPResult(t=t, feasible=True, groups=[sorted(g) for g in groups],
                        sight=(-float(res.fun) if use_sight else 0.0),
                        solve_time=dt, status=res.message, positions=positions)
        if not use_sight:
            out.sight = self.sight_of(out.groups, t)
        return out

    # ---- evaluation on the same discretisation -----------------------------
    def speed_of(self, groups: list) -> float:
        """s_k^Q of a partition: max_e max_i min_q max_{c in C_i} alpha[e,q,c]."""
        cidx = {c: i for i, c in enumerate(self.data.corners)}
        return max(float(self.A[e][:, [cidx[c] for c in g]].max(axis=1).min())
                   for e in range(self.nE) for g in groups)

    def sight_of(self, groups: list, t: float = None) -> float:
        """Best Lambda a fixed partition can reach: each pursuer picks, per e,
        the position within the speed bound t (default: its own speed-optimal
        value at e) that sees the most nearness-weighted escapes."""
        cidx = {c: i for i, c in enumerate(self.data.corners)}
        total = 0.0
        for e in range(self.nE):
            for g in groups:
                sel = [cidx[c] for c in g]
                worst = self.A[e][:, sel].max(axis=1)                      # [q]
                bound = worst.min() + 1e-9 if t is None else t
                ok = np.nonzero(worst <= bound)[0]
                gain = (self.vis[e][sel][:, ok] * self.W[e, sel][:, None]).sum(axis=0)
                total += float(gain.max()) if len(ok) else 0.0
        return total / self.nE

    def min_speed(self, tol: float = 0.005, time_limit: float = 600.0) -> ILPResult:
        """Bisection on t (feasibility model only) for the smallest bound."""
        finite = self.A[np.isfinite(self.A)]
        lo, hi, best = 0.0, float(finite.max()), None
        while hi - lo > tol:
            mid = (lo + hi) / 2
            res = self.solve(mid, use_sight=False, time_limit=time_limit)
            self.log(f'[ILP]   t={mid:.3f}: {"feasible" if res.feasible else "infeasible"} '
                     f'({res.solve_time:.1f}s)')
            if res.feasible:
                best, hi = res, mid
            else:
                lo = mid
        return best


def optimise_partition(data, k: int, grid_n: int = 8, spacing: float = 15.0,
                       relax: tuple = (), log=print, time_limit: float = 600.0) -> dict:
    """Lexicographic optimum (min speed, then max sight at that bound) plus,
    for r in `relax`, the max-sight partition under the bound s_k (1 + r)."""
    m = PartitionILP(data, k, grid_n=grid_n, spacing=spacing, log=log)
    out = {'model': m}
    log(f'[ILP] k={k}: bisection on the speed bound')
    s = m.min_speed(time_limit=time_limit)
    if s is None:
        log('[ILP] no feasible partition at any bound')
        return out
    log(f'[ILP] min feasible bound {s.t:.3f}; maximising sight at that bound')
    lex = m.solve(s.t, use_sight=True, time_limit=time_limit)
    if not lex.feasible:
        log(f'[ILP]   sight model hit the limit ({lex.status}); keeping the feasibility partition')
        lex = s
    lex.t = m.speed_of(lex.groups)
    log(f'[ILP] optimum: s_k={lex.t:.3f}  Lambda={lex.sight:.3f}  groups={lex.groups}  '
        f'({lex.solve_time:.1f}s)')
    out['lex'] = lex
    for r_ in relax:
        res = m.solve(s.t * (1 + r_), use_sight=True, time_limit=time_limit)
        if res.feasible:
            res.t = m.speed_of(res.groups)
            log(f'[ILP] bound +{r_:.0%}: s_k={res.t:.3f}  Lambda={res.sight:.3f}  '
                f'groups={res.groups}  ({res.solve_time:.1f}s)')
        out[f'relax_{r_}'] = res
    return out
