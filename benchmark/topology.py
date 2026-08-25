"""Guard-placement topology measurement (paper Table 2, "Proposed" column).

Unlike the tracking benchmark, a guard cover is a deterministic, one-time
property of the KER+ILP pipeline — no randomness, no per-frame simulation,
so each polygon needs exactly one measurement, not multiple seeds.
"""
from dataclasses import dataclass

import ker_pipeline
from benchmark.harness import load_poly
from geometry import clean_polygon

# Table 2's claimed values for the "KER + ILP (Proposed)" column, keyed by
# polygon name: (claimed_vertices, claimed_reflex_corners, |S|, |E|, Length, %Area).
# poly6/poly7 aren't in the paper's Table 2 (only Table 1 lists all 9).
PAPER_CLAIMED = {
    'poly1': (12, 3,  3, 14, 142.5, 100.0),
    'poly2': (18, 5,  4, 22, 231.8, 100.0),
    'poly3': (24, 7,  5, 31, 318.4, 100.0),
    'poly4': (30, 8,  6, 38, 394.2, 100.0),
    'poly5': (36, 10, 7, 47, 482.0, 100.0),
    'poly8': (46, 12, 8, 59, 612.0, 100.0),
    'poly9': (48, 13, 9, 62, 645.3, 100.0),
}


@dataclass
class TopologyResult:
    environment:  str
    n_vertices:   int
    n_corners:    int
    guard_count:  int      # |S|
    edge_count:   int      # |E|
    total_length: float    # sum of patrol-path edge lengths
    coverage_pct: float    # %Area


def measure_topology(polygon_name: str) -> TopologyResult:
    poly = clean_polygon(load_poly(polygon_name))
    data = ker_pipeline.build(poly, renderer=None, force_recompute=False)
    total_length = sum(line.length for line in data.path_lines)
    return TopologyResult(
        environment=polygon_name,
        n_vertices=len(poly),
        n_corners=len(data.corners),
        guard_count=len(data.guards),
        edge_count=len(data.total_edges),
        total_length=total_length,
        coverage_pct=data.coverage_pct,
    )


def format_topology_table(results: list) -> str:
    header = (f'{"Env":<8} {"v":>3} {"c":>3}   {"|S|":>4} {"|E|":>5} {"Length":>9} {"%Area":>7}   '
              f'{"paper|S|":>8} {"paper|E|":>8} {"paperLen":>8} {"paper%":>7}   status')
    lines = [header, '-' * len(header)]
    for r in results:
        claim = PAPER_CLAIMED.get(r.environment)
        if claim is None:
            claim_str = f'{"—":>8} {"—":>8} {"—":>8} {"—":>7}'
            status = 'no paper data for this name'
        else:
            cv, cc, cS, cE, cLen, cArea = claim
            claim_str = f'{cS:>8} {cE:>8} {cLen:>8.1f} {cArea:>6.1f}%'
            if r.n_vertices == cv and r.n_corners == cc:
                status = 'OK — same polygon as the paper'
            else:
                status = (f'POLYGON CHANGED — paper describes {cv}v/{cc}c, '
                          f'this file is now {r.n_vertices}v/{r.n_corners}c '
                          f'(not a valid comparison)')
        lines.append(
            f'{r.environment:<8} {r.n_vertices:>3} {r.n_corners:>3}   '
            f'{r.guard_count:>4} {r.edge_count:>5} {r.total_length:>9.1f} '
            f'{r.coverage_pct:>6.1f}%   {claim_str}   {status}')
    return '\n'.join(lines)
