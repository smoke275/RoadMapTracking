#!/usr/bin/env python3
"""2D segments of the video, rendered headlessly from the benchmark harness
(same pursuer/evader code as the paper's trials, 1 sim frame = 1 video frame
at 30 fps). Run inside the roadmaptracking image with the repo at /app:

  python video/render2d.py problem|roadmap|race|team|all [--out DIR] [--polygon poly9]

Writes JPEG frames to DIR/<segment>/00000.jpg ... ; assemble.py encodes them.
"""
import argparse
import math
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'video'))
os.chdir(REPO)

from shapely.geometry import LineString, Point  # noqa: E402

from common import (BAND, BG, BLUE, GREEN, GRID, GROUP_COLORS, H, MUTED,  # noqa: E402
                    ORANGE, POLY_EDGE, POLY_FILL, PURPLE, RED, ROAD, SS, TEXT,
                    W, YELLOW, Scene, View, Writer, caption_at, chrome, ease)
from benchmark.harness import load_poly  # noqa: E402
from benchmark.evaders import EscapingEvader, SkeletonEvader  # noqa: E402
from benchmark.metrics import ESCAPE_TAU, alphas_at, has_los  # noqa: E402
from benchmark.pursuers import PURSUER_CLASSES  # noqa: E402
from config import GROUP_AFFINITY_GRID, GROUP_TRACE_STEP  # noqa: E402
from corner_groups import compute_frame_multi, load_partition  # noqa: E402
from geometry import find_intersection, get_random_point_in_polygon  # noqa: E402
from ker_pipeline import _build_vis_shape, build, compute_path_lengths  # noqa: E402
from pursuer_motion import StableNodeController  # noqa: E402

S_P, S_E = 0.8, 1.0
TAG = '2D simulation (Python), real time at 30 frames/s'
FULL_BOX = (40 * SS, 120 * SS, (W - 40) * SS, (H - BAND - 24) * SS)


# ----------------------------------------------------------------- helpers
def poly_xy(data):
    return list(zip(data.x, data.y))


def road_edges(data):
    seen, edges = set(), []
    for a, nbrs in data.graph.items():
        for b in nbrs:
            key = tuple(sorted((a, b)))
            if key not in seen:
                seen.add(key)
                edges.append((a, b))
    return edges


def corner_xy(data, c):
    return (data.x[c], data.y[c])


def vis_shape(data, p):
    return _build_vis_shape(p[0], p[1], data.env)


def draw_base(sc, v, data, edges, corner_color=RED, corner_alpha=1.0,
              road_alpha=1.0, guards=True, corner_colors=None):
    sc.polygon(v.pts(poly_xy(data)), fill=POLY_FILL, outline=POLY_EDGE, width=3)
    if road_alpha > 0:
        a = int(255 * road_alpha)
        for p, q in edges:
            sc.line([v.p(*p), v.p(*q)], ROAD, width=3, alpha=a)
        if guards:
            for g in data.guards:
                sc.ring(v.p(*g), 9, BLUE, width=3)
                sc.disc(v.p(*g), 4, BLUE)
    if corner_alpha > 0:
        for c in data.corners:
            col = corner_colors.get(c, corner_color) if corner_colors else corner_color
            sc.disc(v.p(*corner_xy(data, c)), 6, col, alpha=int(255 * corner_alpha))


def draw_vis(sc, v, shape, color=BLUE, alpha=40):
    if shape is None or shape.is_empty:
        return
    polys = list(shape.geoms) if hasattr(shape, 'geoms') else [shape]
    for pg in polys:
        if pg.geom_type == 'Polygon' and len(pg.exterior.coords) >= 3:
            sc.polygon(v.pts(list(pg.exterior.coords)), fill=color, alpha=alpha)


def draw_agent(sc, c, color, r=11):
    sc.disc(c, r + 3, '#ffffff')
    sc.disc(c, r, color)


def draw_los(sc, a, b, ok):
    if ok:
        sc.line([a, b], GREEN, width=3, alpha=220)
    else:
        sc.dashed(a, b, RED, width=3)


class Stats:
    def __init__(self):
        self.n = self.los = self.escapes = 0
        self.alpha_sum = 0.0
        self.run = 0

    def add(self, los, alpha):
        self.n += 1
        if math.isfinite(alpha):
            self.alpha_sum += alpha
        if los:
            self.los += 1
            self.run = 0
        else:
            self.run += 1
            if self.run == ESCAPE_TAU + 1:
                self.escapes += 1

    @property
    def mean_alpha(self):
        return self.alpha_sum / max(1, self.n)

    @property
    def pct(self):
        return 100.0 * self.los / max(1, self.n)


def worst_corner(pos, pl, data):
    al = alphas_at(pos, pl, data)
    c = max(al, key=lambda k: al[k])
    return c, al[c]


def spawn(data, seed):
    """Seed both generators the trial depends on: the `random` module (start
    position) and skeleton._rng, the module-level generator that picks the
    skeleton evader's destinations and is otherwise seeded from OS entropy
    per process."""
    import skeleton
    random.seed(seed)
    skeleton._rng.seed(seed)
    p = get_random_point_in_polygon(data.shapely_env)
    return (p.x, p.y)


# ---------------------------------------------------------------- segment 1
def seg_problem(data, out, seed_hint=None):
    """Escaping evader against a pursuer held still at its guard: the escape
    into a reflex corner's shadow, then the association region."""
    frames = 600
    edges = road_edges(data)
    best = None
    for seed in ([seed_hint] if seed_hint else range(1, 60)):
        start = spawn(data, seed)
        ev = EscapingEvader(data, start, S_E, ratio=S_P / S_E)
        pur = PURSUER_CLASSES['minmax-alpha'](data, start, S_P)
        ppos = tuple(pur.pos)
        traj, los, fb, path = [], [], None, 0.0
        prev = start
        for f in range(frames):
            e = tuple(ev.pos)
            traj.append(e)
            pl = compute_path_lengths(e[0], e[1], data)
            ok = has_los(ppos, e, data)
            los.append(ok)
            if not ok and fb is None and f > 0:
                fb = f
            path += math.hypot(e[0] - prev[0], e[1] - prev[1])
            prev = e
            ev.step(dt=1.0, pursuer_alphas=alphas_at(ppos, pl, data), pursuer_pos=ppos)
        if fb is not None and 120 <= fb <= 330 and path >= 40 and not any(los[fb:fb + 60]):
            best = (seed, traj, los, fb, ppos)
            break
        if seed_hint:
            best = (seed, traj, los, fb or 300, ppos)
    seed, traj, los, fb, ppos = best
    print(f'[problem] seed {seed}, LOS breaks at frame {fb} ({fb / 30:.1f} s)')
    # the corner the evader hid behind: nearest reflex corner at the break
    eb = traj[fb]
    cstar = min(data.corners, key=lambda c: math.hypot(data.x[c] - eb[0], data.y[c] - eb[1]))
    asso = data.asso.get(cstar)
    vshape = vis_shape(data, ppos)
    shadow = data.shapely_env.difference(vshape) if vshape is not None else None
    tb = fb / 30.0
    t3 = min(tb + 5.0, 14.0)
    caps = [(0.0, 'A pursuer confined to a patrol roadmap tracks an evader that moves freely inside a known polygon.'),
            (tb, 'The evader can break line of sight only by crossing into the shadow of a reflex corner.'),
            (t3, "Occupying that corner's association region before the evader arrives is necessary to prevent the escape.")]
    v = View(data.shapely_env.bounds, FULL_BOX)
    wr = Writer(os.path.join(out, 'problem'))
    for f in range(frames):
        t = f / 30.0
        sc = Scene()
        draw_base(sc, v, data, edges)
        if f >= fb:
            k = ease((f - fb) / 20.0)
            draw_vis(sc, v, shadow, color='#555555', alpha=int(70 * k))
        draw_vis(sc, v, vshape, BLUE, 34)
        if t >= t3:
            k = ease((t - t3) / 0.8)
            if asso:
                sc.polygon(v.pts([(p[0], p[1]) for p in asso]), fill=YELLOW, alpha=int(150 * k))
                sc.polygon(v.pts([(p[0], p[1]) for p in asso]), outline='#c98a00', width=3)
                cx, cy = v.p(*corner_xy(data, cstar))
                sc.label((cx, cy), 'association region A(c)', size=22, color='#8a5a00', dx=18, dy=-92, bg='#ffffff')
        if f >= fb:
            cx, cy = v.p(*corner_xy(data, cstar))
            sc.ring((cx, cy), 18, RED, width=4)
            sc.label((cx, cy), 'reflex corner c', size=22, color=RED, dx=18, dy=-14, bg='#ffffff')
        # trail
        if f > 1:
            sc.line(v.pts(traj[max(0, f - 240):f + 1]), ORANGE, width=3, alpha=120)
        e = traj[f]
        draw_los(sc, v.p(*ppos), v.p(*e), los[f])
        draw_agent(sc, v.p(*ppos), BLUE)
        draw_agent(sc, v.p(*e), ORANGE)
        sc.label(v.p(*ppos), 'pursuer', size=20, color=BLUE, dx=16, dy=-34, bg='#ffffff')
        sc.label(v.p(*e), 'evader', size=20, color=ORANGE, dx=16, dy=-34, bg='#ffffff')
        im = sc.finish()
        tiles = [('line of sight', 'yes' if los[f] else 'lost', GREEN if los[f] else RED)]
        leg = [('disc', BLUE, 'pursuer (on the roadmap)'), ('disc', ORANGE, 'evader'),
               ('ring', BLUE, 'patrol node (guard)'), ('line', ROAD, 'patrol roadmap'),
               ('disc', RED, 'reflex corner'), ('fill', BLUE, 'visible from the pursuer')]
        if f >= fb:
            leg.append(('fill', '#555555', 'shadow: hidden from the pursuer'))
        if t >= t3:
            leg.append(('fill', YELLOW, 'association region A(c)'))
        chrome(im, 'The problem', 'poly9, 39 vertices, 17 reflex corners. Pursuer held at its patrol node; escaping evader.',
               caption_at(caps, t), TAG, tiles=tiles, legend_items=leg)
        wr.write(im)
    print(f'[problem] {wr.n} frames')


# ---------------------------------------------------------------- segment 2
def ker_pool(data):
    """Keil's kernel candidates: extended reflex chords and their pairwise
    intersections (same construction as ker_pipeline.build)."""
    pts = poly_xy(data)
    env = data.shapely_env.buffer(1e-6)
    chords = []
    for i in data.corners:
        for j in range(len(pts)):
            if j == i:
                continue
            seg = LineString([pts[j], pts[i]])
            if not env.covers(seg):
                continue
            dx, dy = pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]
            n = math.hypot(dx, dy)
            if n < 1e-9:
                continue
            last, _ = find_intersection(pts, Point(pts[i]), (dx / n, dy / n))
            if last is None or not all(math.isfinite(q) for q in last):
                continue
            chords.append((pts[i], (last[0], last[1])))
    cands = [corner_xy(data, c) for c in data.corners] + [c[1] for c in chords]
    lines = [LineString(c) for c in chords]
    for a in range(len(lines)):
        for b in range(a + 1, len(lines)):
            p = lines[a].intersection(lines[b])
            if not p.is_empty and p.geom_type == 'Point':
                cands.append((p.x, p.y))
    return chords, cands


def seg_roadmap(data, out):
    frames = 660
    edges = road_edges(data)
    chords, cands = ker_pool(data)
    print(f'[roadmap] {len(chords)} chords, {len(cands)} candidates')
    # draw roadmap edges in a growing order from the first guard
    g0 = data.guards[0]
    edges = sorted(edges, key=lambda e: min(math.hypot(e[0][0] - g0[0], e[0][1] - g0[1]),
                                            math.hypot(e[1][0] - g0[0], e[1][1] - g0[1])))
    caps = [(0.0, 'poly9: 39 vertices. The pursuer may only move on a patrol roadmap built once, offline.'),
            (2.0, 'Reflex corners: 17. Each one casts a shadow the evader can hide in.'),
            (5.0, "Keil's kernel candidates: 1,477 points, where extended reflex chords meet."),
            (10.0, 'An exact 0/1 set cover keeps 7 of them, so every association region holds a patrol node.'),
            (14.0, 'Geodesic patrol roadmap between the guards: the pursuer lives on these edges.'),
            (18.0, 'Nine benchmark polygons: 360 to 2,784 candidates become 5 to 13 guards that see 95.6 to 100% of the area.')]
    v = View(data.shapely_env.bounds, FULL_BOX)
    wr = Writer(os.path.join(out, 'roadmap'))
    for f in range(frames):
        t = f / 30.0
        sc = Scene()
        sc.polygon(v.pts(poly_xy(data)), fill=POLY_FILL, outline=POLY_EDGE, width=3)
        # association regions, late phase
        if t >= 18.0:
            k = ease((t - 18.0) / 1.0)
            for c in data.corners:
                a = data.asso.get(c)
                if a:
                    sc.polygon(v.pts([(p[0], p[1]) for p in a]), fill=YELLOW, alpha=int(45 * k))
        # chords + candidates, drawn on one translucent layer (one composite)
        if 5.0 <= t < 14.0:
            k = ease((t - 5.0) / 1.5) * (1.0 if t < 10.0 else max(0.0, 1.0 - (t - 10.0) / 1.5))
            if k > 0:
                from PIL import ImageDraw
                from common import rgb
                lay = sc.layer()
                ld = ImageDraw.Draw(lay)
                for a, b in chords:
                    ld.line([v.p(*a), v.p(*b)], fill=rgb(RED, int(70 * k)), width=SS)
                r = 3 * SS
                for p in cands:
                    x, y = v.p(*p)
                    ld.ellipse([x - r, y - r, x + r, y + r], fill=rgb('#1baf7a', int(230 * k)))
                sc.blend(lay)
        # roadmap
        if t >= 14.0:
            n = int(len(edges) * ease((t - 14.0) / 2.5))
            for p, q in edges[:n]:
                sc.line([v.p(*p), v.p(*q)], ROAD, width=3)
        # guards
        if t >= 10.0:
            k = ease((t - 10.0) / 1.0)
            for g in data.guards:
                sc.ring(v.p(*g), 9 + 10 * (1 - k), BLUE, width=3)
                sc.disc(v.p(*g), 4, BLUE)
        # corners
        if t >= 2.0:
            k = ease((t - 2.0) / 0.8)
            for c in data.corners:
                sc.disc(v.p(*corner_xy(data, c)), 6 + 8 * (1 - k), RED, alpha=int(255 * k))
        im = sc.finish()
        tiles = []
        if t >= 2.0:
            tiles.append(('reflex corners', '17', RED))
        if 5.0 <= t < 11.5:
            tiles.append(('kernel candidates', '1,477', GREEN))
        if t >= 10.0:
            tiles.append(('guards (ILP cover)', '7', BLUE))
        if t >= 14.0:
            tiles.append(('patrol graph edges', str(len(data.total_edges)), MUTED))
        leg = [('disc', RED, 'reflex corner')]
        if 5.0 <= t < 11.5:
            leg += [('dash', RED, 'extended reflex chord'), ('disc', GREEN, 'kernel candidate')]
        if t >= 10.0:
            leg.append(('ring', BLUE, 'selected guard (patrol node)'))
        if t >= 14.0:
            leg.append(('line', ROAD, 'patrol roadmap edge'))
        if t >= 18.0:
            leg.append(('fill', YELLOW, 'association regions A(c)'))
        chrome(im, 'Building the patrol roadmap', 'Keil kernel candidates, exact set cover, geodesic patrol graph',
               caption_at(caps, t), TAG, tiles=tiles, legend_items=leg if t >= 2.0 else None)
        wr.write(im)
    print(f'[roadmap] {wr.n} frames')


# ---------------------------------------------------------------- segment 3
def seg_race(data, out, seed=21, dry=False):
    """dry=True runs the identical loop (same calls, same order) without
    drawing or writing frames and prints the outcome: used to choose a seed
    whose outcome is representative of the paper's averages."""
    frames = 900
    edges = road_edges(data)
    start = spawn(data, seed)
    ev = SkeletonEvader(data.skel_nodes, data.skel_adj, data.shapely_env, start, S_E)
    strategies = [('minmax-alpha-vis', 'Min-Max +Vis (proposed)', BLUE),
                  ('geo-follow', 'Geo-Follow (baseline)', ORANGE)]
    purs = [PURSUER_CLASSES[s](data, start, S_P) for s, _, _ in strategies]
    stats = [Stats() for _ in purs]
    boxes = [(40 * SS, 150 * SS, 950 * SS, (H - BAND - 24) * SS),
             (970 * SS, 150 * SS, (W - 40) * SS, (H - BAND - 24) * SS)]
    views = [View(data.shapely_env.bounds, b, margin=0.035) for b in boxes]
    caps = [(0.0, 'Each corner is a timing race: the ratio α of the pursuer\'s roadmap distance to the association region over the evader\'s distance to the corner.'),
            (7.0, 'Per frame, the optimal guard g* is a dynamically weighted network 1-center, solved exactly by an upper-envelope routine.'),
            (14.0, 'A stable-node roadmap controller turns placement into motion; an ε-slack layer spends spare timing margin on line of sight.'),
            (21.0, 'Paper, nine polygons: lowest ᾱ everywhere, up to 42% below Geo-Follow; +Vis adds 0.7 to 7.7 points of %LOS at no timing cost.')]
    wr = None if dry else Writer(os.path.join(out, 'race'))
    for f in range(frames):
        t = f / 30.0
        e = tuple(ev.pos)
        pl = compute_path_lengths(e[0], e[1], data)
        sc = None if dry else Scene()
        infos = []
        for i, (pur, st, v) in enumerate(zip(purs, stats, views)):
            pur.step(e, pl)
            p = tuple(pur.pos)
            los = has_los(p, e, data)
            c, a = worst_corner(p, pl, data)
            st.add(los, a)
            g = pur.target(e, pl)
            vshape = vis_shape(data, p)
            infos.append((st, strategies[i]))
            if dry:
                continue
            draw_base(sc, v, data, edges)
            draw_vis(sc, v, vshape, BLUE, 34)
            draw_los(sc, v.p(*p), v.p(*e), los)
            sc.ring(v.p(*corner_xy(data, c)), 15, RED, width=3)
            sc.label(v.p(*corner_xy(data, c)), f'α = {a:.2f}', size=19, color=RED, dx=14, dy=-40, bg='#ffffff')
            sc.diamond(v.p(*g), 11, PURPLE)
            sc.label(v.p(*g), 'g*' if i == 0 else 'target', size=18, color=PURPLE, dx=14, dy=14, bg='#ffffff')
            draw_agent(sc, v.p(*p), strategies[i][2])
            draw_agent(sc, v.p(*e), ORANGE)
        if dry:
            ev.step(dt=1.0)
            continue
        im = sc.finish()
        pane_titles = [(60, 112, strategies[0][1], BLUE), (990, 112, strategies[1][1], ORANGE)]
        from PIL import ImageDraw
        d = ImageDraw.Draw(im, 'RGBA')
        for x0, (st, (_, _, col)) in zip((60, 990), infos):
            txt = f'worst-corner α {st.mean_alpha:.2f}   %LOS {st.pct:.0f}   escapes {st.escapes}'
            from common import font, rgb
            fnt = font(20)
            w = d.textlength(txt, font=fnt)
            d.rounded_rectangle([x0 + 400 - 10, 112 - 6, x0 + 400 + w + 10, 112 + 32], radius=8, fill=rgb('#ffffff', 235), outline=rgb(GRID))
            d.text((x0 + 400, 112), txt, font=fnt, fill=rgb(TEXT))
        chrome(im, 'Online placement and tracking', 'Same evader and seed in both panels; skeleton evader, s_p / s_e = 0.8',
               caption_at(caps, t), TAG, pane_titles=pane_titles)
        wr.write(im)
        ev.step(dt=1.0)
    for st, (_, name, _) in infos:
        print(f'[race] seed {seed} {name}: mean worst α {st.mean_alpha:.2f}, %LOS {st.pct:.1f}, escapes {st.escapes}')
    if wr:
        print(f'[race] {wr.n} frames')


# ---------------------------------------------------------------- segment 5
def seg_team(data, out, k=3, m=3, seed=3):
    frames, intro = 600, 150
    edges = road_edges(data)
    method, groups = load_partition(data, GROUP_AFFINITY_GRID, GROUP_TRACE_STEP, k)
    print(f'[team] partition {method}: {groups}')
    corner_colors = {c: GROUP_COLORS[i] for i, g in enumerate(groups) for c in g}
    import skeleton
    random.seed(seed)
    skeleton._rng.seed(seed)
    evs = [get_random_point_in_polygon(data.shapely_env) for _ in range(m)]
    agents = [SkeletonEvader(data.skel_nodes, data.skel_adj, data.shapely_env, (p.x, p.y), S_E, avoid_recent=3) for p in evs]
    evs = [(p.x, p.y) for p in evs]
    mf0 = compute_frame_multi(evs, None, [evs[0]] * len(groups), data, groups)
    ctrls = [StableNodeController((fc.guard.x, fc.guard.y)) for fc in mf0.frames]
    caps = [(0.0, f'A mixed-integer program partitions the corners: team speed ratio 1.682 (one pursuer) → 1.134 (k = 2) → 1.004 (k = 3).'),
            (5.0, 'Each pursuer guards its own corner group; the nearest-evader rule lets every corner race the evader closest to it.'),
            (15.0, 'Paper, poly9: three pursuers keep 89.1% of three evaders in view; one pursuer keeps 48.2%, with 5.0 escapes per trial.')]
    v = View(data.shapely_env.bounds, FULL_BOX)
    wr = Writer(os.path.join(out, 'team'))
    seen_frames, n_frames, escapes = 0.0, 0, 0
    no_los = [0] * m
    for f in range(intro + frames):
        t = f / 30.0
        moving = f >= intro
        positions = [tuple(c.pos) for c in ctrls]
        mf = compute_frame_multi(evs, None, positions, data, groups)
        sc = Scene()
        draw_base(sc, v, data, edges, corner_colors=corner_colors)
        for i, p in enumerate(positions):
            draw_vis(sc, v, vis_shape(data, p), GROUP_COLORS[i], 26)
        seen = 0
        for j, e in enumerate(evs):
            viewers = [p for p in positions if data.shapely_env.covers(LineString([p, e]))]
            if viewers:
                seen += 1
                p = min(viewers, key=lambda q: math.hypot(q[0] - e[0], q[1] - e[1]))
                sc.line([v.p(*p), v.p(*e)], GREEN, width=3, alpha=200)
        if moving:
            n_frames += 1
            seen_frames += seen / m
            for j, e in enumerate(evs):
                if any(data.shapely_env.covers(LineString([p, e])) for p in positions):
                    no_los[j] = 0
                else:
                    no_los[j] += 1
                    if no_los[j] == ESCAPE_TAU + 1:
                        escapes += 1
        for i, (p, fc) in enumerate(zip(positions, mf.frames)):
            sc.diamond(v.p(fc.guard.x, fc.guard.y), 10, GROUP_COLORS[i])
            draw_agent(sc, v.p(*p), GROUP_COLORS[i])
            sc.label(v.p(*p), f'P{i + 1}', size=18, color=GROUP_COLORS[i], dx=14, dy=-30, bg='#ffffff')
        for j, e in enumerate(evs):
            draw_agent(sc, v.p(*e), ORANGE)
            sc.label(v.p(*e), f'E{j + 1}', size=18, color=ORANGE, dx=14, dy=-30, bg='#ffffff')
        im = sc.finish()
        tiles = [('in view now', f'{seen}/{m}', GREEN if seen == m else ORANGE),
                 ('% in view (running)', f'{100 * seen_frames / max(1, n_frames):.0f}', TEXT),
                 ('escapes', str(escapes), RED if escapes else TEXT)]
        leg = [('disc', GROUP_COLORS[0], 'pursuer P1 and its corner group'),
               ('disc', GROUP_COLORS[1], 'pursuer P2 and its corner group'),
               ('disc', GROUP_COLORS[2], 'pursuer P3 and its corner group'),
               ('disc', ORANGE, 'evaders E1 to E3'), ('diamond', GROUP_COLORS[0], 'guard target g* per group'),
               ('ring', BLUE, 'patrol node'), ('line', GREEN, 'line of sight'), ('fill', BLUE, 'visible from a pursuer')]
        chrome(im, 'Pursuer teams', f'poly9, k = {k} pursuers vs. m = {m} skeleton evaders, MILP corner partition, s_p / s_e = 0.8',
               caption_at(caps, t), TAG, tiles=tiles, legend_items=leg)
        wr.write(im)
        if moving:
            for ctrl, fc in zip(ctrls, mf.frames):
                ctrl.step(data.graph, (fc.guard.x, fc.guard.y), S_P)
            evs = [tuple(a.step(1.0)) for a in agents]
    print(f'[team] % in view {100 * seen_frames / max(1, n_frames):.1f}, escapes {escapes}, {wr.n} frames')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('segment', choices=['problem', 'roadmap', 'race', 'team', 'all'])
    ap.add_argument('--out', default=os.path.join(REPO, 'video', 'out', 'frames'))
    ap.add_argument('--polygon', default='poly9')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--dry', action='store_true', help='race only: simulate without rendering, print the outcome')
    a = ap.parse_args()
    data = build(load_poly(a.polygon))
    t0 = time.time()
    segs = ['problem', 'roadmap', 'race', 'team'] if a.segment == 'all' else [a.segment]
    for s in segs:
        if s == 'problem':
            seg_problem(data, a.out, a.seed)
        elif s == 'roadmap':
            seg_roadmap(data, a.out)
        elif s == 'race':
            seg_race(data, a.out, a.seed or 21, dry=a.dry)
        elif s == 'team':
            seg_team(data, a.out, seed=a.seed or 3)
    print(f'done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
