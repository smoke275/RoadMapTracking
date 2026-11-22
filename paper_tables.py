"""LaTeX table rows and the statistics quoted in the experiments section,
computed from the result files of one batch run.

    python paper_tables.py --since 20261122_042800

Only files in benchmark_results whose time stamp is at or after --since (and
before --until, if given) are read, so that an older run cannot leak in.
Significance: paired Wilcoxon signed-rank tests over the shared seeds, with
Holm correction across the environments.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st

import numpy as np
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'benchmark_results')
POLYS = [f'poly{i}' for i in range(1, 10)]
STRATS = ['minmax-alpha', 'minmax-alpha-vis', 'geo-follow', 'greedy-los', 'tsp-patrol',
          'kernel-control']
SHORT = {'minmax-alpha': 'MM', 'minmax-alpha-vis': '+Vis', 'geo-follow': 'GF',
         'greedy-los': 'GL', 'tsp-patrol': 'TSP', 'kernel-control': 'Kernel',
         'watchman-patrol': 'WM-Patrol'}


def files(suffix, since, until):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, f'*{suffix}'))):
        stamp = os.path.basename(f)[:15]
        if re.fullmatch(r'\d{8}_\d{6}', stamp) and since <= stamp and (not until or stamp < until):
            out.append(f)
    return out


def load_runs(since, until):
    """List of (header dict, rows) for every benchmark table in the window."""
    runs = []
    for f in files('_table.txt', since, until):
        head = open(f).readline()
        raw = f.replace('_table.txt', '_raw.csv')
        if not head.startswith('# polygons=') or not os.path.exists(raw):
            continue
        h = dict(sp=float(re.search(r'sp=([0-9.]+)', head).group(1)),
                 roadmap=re.search(r'roadmap=(\w+)', head).group(1),
                 evaders=re.search(r"evaders=\[([^\]]*)\]", head).group(1),
                 strategies=re.search(r"strategies=\[([^\]]*)\]", head).group(1),
                 seeds=int(re.search(r'seeds=(\d+)', head).group(1)), file=os.path.basename(f))
        runs.append((h, list(csv.DictReader(open(raw)))))
    return runs


def col(rows, env, strat, evader, key):
    """Values by seed for one configuration."""
    d = {int(r['seed']): float(r[key]) for r in rows
         if r['environment'] == env and r['strategy'] == strat and r['evader'] == evader}
    return [d[s] for s in sorted(d)]


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def wil(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def ms(v, nd=2):
    return f'{st.mean(v):.{nd}f} \\pm {st.pstdev(v) if len(v) < 2 else st.stdev(v):.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', required=True, help='first time stamp, e.g. 20261122_042800')
    ap.add_argument('--until', default=None)
    a = ap.parse_args()
    runs = load_runs(a.since, a.until)

    # nominal runs: sp = 0.8, KER roadmap, full strategy set (plus the kernel runs)
    nominal = []
    for h, rows in runs:
        if h['sp'] == 0.8 and h['roadmap'] in ('None', 'ker') and 'escaping' not in h['evaders'] \
                and ("'tsp-patrol'" in h['strategies'] or "'kernel-control'" in h['strategies']) \
                and 'watchman' not in h['strategies']:
            nominal += rows
    print(f'% {len(runs)} benchmark tables in the window, {len(nominal)} nominal trial rows')

    def val(env, s, ev, key):
        return col(nominal, env, s, ev, key)

    # ---- Table: tracking against the skeleton evader -----------------------
    print('\n%%%% Table tracking_comparison (skeleton evader)')
    for env in POLYS:
        A = {s: val(env, s, 'skeleton', 'mean_alpha') for s in STRATS}
        L = {s: val(env, s, 'skeleton', 'los_pct') for s in STRATS}
        if not all(A.values()):
            print(f'% {env}: missing strategies {[s for s in STRATS if not A[s]]}')
            continue
        ba = min(STRATS, key=lambda s: st.mean(A[s]))
        bl = max(STRATS, key=lambda s: st.mean(L[s]))
        cells = [(f'$\\mathbf{{{ms(A[s])}}}$' if s == ba else f'${ms(A[s])}$') for s in STRATS]
        cells += [(f'$\\mathbf{{{ms(L[s], 1)}}}$' if s == bl else f'${ms(L[s], 1)}$') for s in STRATS]
        print(f'{env} & ' + ' & '.join(cells) + ' \\\\')
    esc = {s: [st.mean(val(e, s, 'skeleton', 'n_escape')) for e in POLYS
               if val(e, s, 'skeleton', 'n_escape')] for s in STRATS}
    print('% skeleton escapes per trial, range over environments: ' +
          ', '.join(f'{SHORT[s]} {min(v):.2f}-{max(v):.2f}' for s, v in esc.items() if v))

    # ---- Table: adversarial evader -----------------------------------------
    print('\n%%%% Table adversarial')
    five = STRATS[:5]
    for env in POLYS:
        los = [st.mean(val(env, s, 'adversarial', 'los_pct')) for s in five]
        ne = [st.mean(val(env, s, 'adversarial', 'n_escape')) for s in five]
        print(f'{env} & ' + ' & '.join(f'{v:.1f}' for v in los) + ' & ' +
              ' & '.join(f'{v:.2f}' for v in ne) + ' \\\\')
    for s in five:
        los = [st.mean(val(e, s, 'adversarial', 'los_pct')) for e in POLYS]
        ne = [st.mean(val(e, s, 'adversarial', 'n_escape')) for e in POLYS]
        print(f'% adversarial {SHORT[s]}: %LOS {min(los):.1f}-{max(los):.1f}, escapes {min(ne):.2f}-{max(ne):.2f}')

    # ---- Significance (skeleton evader) -------------------------------------
    print('\n%%%% Paired Wilcoxon, Holm across the nine environments (skeleton evader)')
    pairs = [('minmax-alpha-vis', 'geo-follow'), ('minmax-alpha-vis', 'greedy-los'),
             ('minmax-alpha', 'geo-follow'), ('minmax-alpha', 'greedy-los'),
             ('minmax-alpha-vis', 'minmax-alpha'), ('minmax-alpha-vis', 'kernel-control'),
             ('minmax-alpha', 'kernel-control')]
    for key, better in (('mean_alpha', 'lower'), ('los_pct', 'higher')):
        for s1, s2 in pairs:
            p, diff = [], []
            for env in POLYS:
                x, y = val(env, s1, 'skeleton', key), val(env, s2, 'skeleton', key)
                p.append(wil(x, y))
                diff.append(st.mean(x) - st.mean(y))
            adj = holm(p)
            sig = [(env, d) for env, d, q in zip(POLYS, diff, adj) if q < 0.05]
            good = [e for e, d in sig if (d < 0) == (better == 'lower')]
            bad = [e for e, d in sig if (d < 0) != (better == 'lower')]
            print(f'% {key:10s} {SHORT[s1]:5s} vs {SHORT[s2]:6s}: mean diff {min(diff):+.2f}..{max(diff):+.2f}; '
                  f'significantly better on {len(good)} {good}, worse on {len(bad)} {bad}')

    # ---- Derived numbers quoted in the text ---------------------------------
    print('\n%%%% Quoted numbers (skeleton evader)')
    for env in POLYS:
        A = {s: st.mean(val(env, s, 'skeleton', 'mean_alpha')) for s in STRATS}
        L = {s: st.mean(val(env, s, 'skeleton', 'los_pct')) for s in STRATS}
        best_prop = min(A['minmax-alpha'], A['minmax-alpha-vis'])
        print(f'% {env}: reduction of mean alpha vs GF {100 * (1 - best_prop / A["geo-follow"]):5.1f}%, '
              f'vs GL {100 * (1 - best_prop / A["greedy-los"]):5.1f}%, vs Kernel '
              f'{100 * (1 - best_prop / A["kernel-control"]):5.1f}% | +Vis - MM: alpha '
              f'{A["minmax-alpha-vis"] - A["minmax-alpha"]:+.2f}, %LOS '
              f'{L["minmax-alpha-vis"] - L["minmax-alpha"]:+.1f} | %LOS +Vis - GF '
              f'{L["minmax-alpha-vis"] - L["geo-follow"]:+.1f}, +Vis - GL '
              f'{L["minmax-alpha-vis"] - L["greedy-los"]:+.1f}')

    # ---- Environment table ----------------------------------------------------
    print('\n%%%% Table environments')
    stats_f = files('_polygon_stats.csv', a.since, a.until)
    env_f = files('_environments.txt', a.since, a.until)
    K = {}
    if env_f:
        for line in open(env_f[-1]):
            t = line.split()
            if t and t[0] in POLYS:
                K[t[0]] = t[3]
    S = {}
    if stats_f:
        S = {r['polygon']: r for r in csv.DictReader(open(stats_f[-1]))}
        for env in POLYS:
            r = S[env]
            print(f'{env} & {r["n"]} & {r["reflex"]} & {r["essential"]} & {float(r["mean_vis"]):.2f} & '
                  f'{K.get(env, "?")} & {r["guards"]} & {r["edges"]} & {float(r["area_pct"]):.1f} & '
                  f'{float(r["sstar_ker"]):.3f} & {float(r["sstar_watchman"]):.3f} \\\\')
        ss = {e: float(S[e]['sstar_ker']) for e in POLYS}
        print(f'% s* range: {min(ss.values()):.3f} ({min(ss, key=ss.get)}) to {max(ss.values()):.3f} '
              f'({max(ss, key=ss.get)}); |J| range {min(int(S[e]["guards"]) for e in POLYS)}-'
              f'{max(int(S[e]["guards"]) for e in POLYS)}; |K| range {min(map(int, K.values()))}-{max(map(int, K.values()))}')
        ratio = [float(S[e]['sstar_watchman']) / ss[e] for e in POLYS]
        print('% s*_W / s*: ' + ', '.join(f'{e} {r:.2f}' for e, r in zip(POLYS, ratio)))

        print('\n%%%% Trial peak against s* (Min-Max and Min-Max +Vis, skeleton evader)')
        above = 0
        for env in POLYS:
            pk = {s: st.mean(val(env, s, 'skeleton', 'peak_alpha')) for s in STRATS[:2]}
            mean_a = st.mean(val(env, 'minmax-alpha', 'skeleton', 'mean_alpha'))
            above += mean_a > ss[env]
            print(f'% {env}: s* {ss[env]:.2f}, mean trial peak MM {pk["minmax-alpha"]:.2f}, '
                  f'+Vis {pk["minmax-alpha-vis"]:.2f}, mean alpha MM {mean_a:.2f}, '
                  f'guards {S[env]["guards"]}, edges {S[env]["edges"]}')
        print(f'% mean alpha of Min-Max exceeds s* on {above} of 9 environments')

        print('\n%%%% Spearman correlations of polygon descriptors (n = 9)')
        met = {'mean_alpha(+Vis)': [st.mean(val(e, 'minmax-alpha-vis', 'skeleton', 'mean_alpha')) for e in POLYS],
               '%LOS(+Vis)': [st.mean(val(e, 'minmax-alpha-vis', 'skeleton', 'los_pct')) for e in POLYS],
               's*': [ss[e] for e in POLYS]}
        for d in ('mean_vis', 'reflex', 'n', 'essential', 'guards', 'geo_diam_norm', 'passage_norm',
                  'link_diameter', 'ker_norm', 'watchman_norm'):
            x = [float(S[e][d]) for e in POLYS]
            print(f'% {d:14s} ' + '  '.join(f'{m}: {spearmanr(x, y).statistic:+.2f} (p={spearmanr(x, y).pvalue:.3f})'
                                           for m, y in met.items()))

    # ---- Speed sweeps on poly9 --------------------------------------------------
    print('\n%%%% Speed sweep on poly9 (Min-Max, Min-Max +Vis; skeleton and adversarial)')
    sweep = {}
    for h, rows in runs:
        if h['strategies'] == "'minmax-alpha', 'minmax-alpha-vis'" and h['roadmap'] in ('None', 'ker'):
            sweep[h['sp']] = rows
    for sp in sorted(sweep):
        r = sweep[sp]
        print(f'% sp={sp:.1f}: ' + '; '.join(
            f'{SHORT[s]} {ev[:4]} %LOS {st.mean(col(r, "poly9", s, ev, "los_pct")):.1f} esc '
            f'{st.mean(col(r, "poly9", s, ev, "n_escape")):.2f}'
            for s in STRATS[:2] for ev in ('skeleton', 'adversarial')))

    print('\n%%%% Escaping evader sweep on poly9 (escapes per trial)')
    escs = {}
    for h, rows in runs:
        if h['evaders'] == "'escaping'" and h['seeds'] >= 50:
            escs[h['sp']] = rows
    four = STRATS[:4]
    for sp in sorted(escs):
        r = escs[sp]
        print(f'% sp={sp:.1f}: ' + ', '.join(
            f'{SHORT[s]} esc {st.mean(col(r, "poly9", s, "escaping", "n_escape")):.2f} '
            f'%LOS {st.mean(col(r, "poly9", s, "escaping", "los_pct")):.1f}' for s in four))
    for name, sel in (('sp <= 1.0', lambda x: x <= 1.0), ('sp >= 1.4', lambda x: x >= 1.4),
                      ('sp >= 1.2', lambda x: x >= 1.2)):
        pooled = {s: sum((col(escs[sp], 'poly9', s, 'escaping', 'n_escape')
                          for sp in sorted(escs) if sel(sp)), []) for s in four}
        los = {s: sum((col(escs[sp], 'poly9', s, 'escaping', 'los_pct')
                       for sp in sorted(escs) if sel(sp)), []) for s in four}
        if not pooled['minmax-alpha']:
            continue
        ref = 'minmax-alpha-vis'
        print(f'% pooled {name}: ' + ', '.join(f'{SHORT[s]} {st.mean(pooled[s]):.2f}' for s in four) +
              ' | paired p, +Vis vs ' + ', '.join(f'{SHORT[s]} {wil(pooled[ref], pooled[s]):.2g}'
                                                  for s in four if s != ref) +
              ' | %LOS ' + ', '.join(f'{SHORT[s]} {st.mean(los[s]):.1f}' for s in four))

    # ---- Watchman route as the roadmap ---------------------------------------------
    print('\n%%%% Watchman comparison on poly9')
    for h, rows in runs:
        if 'watchman' in h['strategies'] or h['roadmap'] == 'watchman':
            env = rows[0]['environment']          # 'poly9@watchman' on the watchman roadmap
            for s in STRATS[:5] + ['watchman-patrol']:
                for ev in ('skeleton', 'adversarial'):
                    v = col(rows, env, s, ev, 'mean_alpha')
                    if v:
                        print(f'% roadmap={h["roadmap"]:8s} {SHORT[s]:9s} {ev:11s} mean alpha {st.mean(v):.2f} '
                              f'%LOS {st.mean(col(rows, env, s, ev, "los_pct")):.1f} '
                              f'esc {st.mean(col(rows, env, s, ev, "n_escape")):.2f}')

    # ---- Multi-evader table ----------------------------------------------------------
    print('\n%%%% Table multi_evader (k pursuers against m evaders, sp = 0.8)')
    for f in files('_multi_evader.json', a.since, a.until):
        d = json.load(open(f))
        if d['args']['sp'] != 0.8 or len(d['args']['m']) < 3:
            continue
        orc = {(o['k'], o['m']): o for o in d['oracle']}
        for t in sorted(d['trials'], key=lambda t: (t['k'], t['m'])):
            o = orc[(t['k'], t['m'])]
            pm = lambda v, nd: f'{v[0]:.{nd}f} \\pm {v[1]:.{nd}f}'      # stored as [mean, std]
            print(f'{t["k"]} & {t["m"]} & {o["p95"]:.2f} & {o["mean"]:.2f} & ${pm(t["mean_alpha"], 2)}$ & '
                  f'${pm(t["in_view_pct"], 1)}$ & ${pm(t["n_escape"], 1)}$ \\\\'
                  + (f'   % standoff {o["standoff"]:.2f}' if t['k'] == 1 and t['m'] == 1 else ''))
    print('\n%%%% Team speed sweep (one evader), % in view')
    for f in files('_multi_evader.json', a.since, a.until):
        d = json.load(open(f))
        if len(d['args']['m']) == 1:
            print(f'% sp={d["args"]["sp"]:.1f}: ' + ', '.join(
                f'k={t["k"]} {t["in_view_pct"][0]:.1f}% esc {t["n_escape"][0]:.2f}'
                for t in sorted(d['trials'], key=lambda t: t['k'])) + f'   [{os.path.basename(f)}]')

    print('\n%%%% Team partitions')
    for f in files('_groups.txt', a.since, a.until):
        for line in open(f):
            if line.startswith('poly'):
                print('% ' + line[:52].rstrip())


if __name__ == '__main__':
    main()
