"""
Read-only polygon viewer — no drawing, just a quick look at a saved polygon
(vertex order, reflex corners, area, self-intersection). Pass --cover to
also run the KER+ILP guard-placement pipeline and overlay the minimum guard
set and each guard's visibility coverage.

Run:  python view_polygon.py                 # views the current polygon (config.FILE_NAME)
      python view_polygon.py poly2           # views a preset (resources/sites_poly2.csv)
      python view_polygon.py my_env.csv      # views a specific file
      python view_polygon.py --cover         # also shows the minimum guard cover
"""
import argparse
import colorsys
import math

import pygame
import visilibity as vis
from shapely.geometry import Polygon as ShapelyPolygon

import ker_pipeline
from benchmark.harness import load_poly
from config import EPSILON
from geometry import clean_polygon, poly_to_points


def _shoelace(pts) -> float:
    n = len(pts)
    return sum((pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1])
               for i in range(n)) / 2


def _is_reflex(pts, i) -> bool:
    a, b, c = pts[(i - 1) % len(pts)], pts[i], pts[(i + 1) % len(pts)]
    s = [b[0] - a[0], b[1] - a[1]]
    t = [c[0] - b[0], c[1] - b[1]]
    return s[0] * t[1] - t[0] * s[1] > 1e-9


def _is_simple(pts) -> bool:
    try:
        return ShapelyPolygon(pts).is_valid
    except Exception:
        return False


def _palette(n):
    return [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / max(n, 1), 0.65, 0.95))
           for i in range(n)]


def view_polygon(name: str = 'current', show_cover: bool = False):
    poly_pts = clean_polygon(load_poly(name))
    pts = [(p.x(), p.y()) for p in poly_pts]

    minx, maxx = min(p[0] for p in pts), max(p[0] for p in pts)
    miny, maxy = min(p[1] for p in pts), max(p[1] for p in pts)
    span = max(maxx - minx, maxy - miny, 1.0)
    pad = span * 0.1
    ox, oy = minx - pad, miny - pad
    scale = span + 2 * pad

    pygame.init()
    W, H = 700, 700
    screen = pygame.display.set_mode((W, H))
    caption = f'Polygon viewer — {name} ({len(pts)} vertices)'
    pygame.display.set_caption(f'{caption} — ESC to close')
    font       = pygame.font.SysFont('monospace', 15)
    font_small = pygame.font.SysFont('monospace', 11)

    WHITE, GRAY, GRID = (255, 255, 255), (60, 60, 60), (80, 80, 80)
    RED, CYAN, ORANGE = (220, 70, 70), (0, 220, 255), (240, 140, 40)
    GOLD = (255, 215, 60)

    def to_px(px, py):
        return int((px - ox) * W / scale), int(H - (py - oy) * H / scale)

    reflex = [i for i in range(len(pts)) if _is_reflex(pts, i)]
    area = abs(_shoelace(pts))
    simple = _is_simple(pts)

    guards, guard_polys, coverage_pct = [], [], None
    if show_cover:
        print(f'[view_polygon] running KER+ILP guard placement for {name} ...')
        data = ker_pipeline.build(poly_pts, renderer=None, force_recompute=False)
        guards = list(data.guards)
        coverage_pct = data.coverage_pct
        palette = _palette(len(guards))
        for guard_x, guard_y in guards:
            vp = vis.Visibility_Polygon(vis.Point(guard_x, guard_y), data.env, EPSILON)
            vx, vy = poly_to_points(vp)
            guard_polys.append(list(zip(vx, vy)))

    running = True
    while running:
        screen.fill(GRAY)

        # grid lines at round-number intervals covering the bounding box
        step = 10 ** max(0, len(str(int(span))) - 2)
        gx = math.floor(ox / step) * step
        while gx <= ox + scale:
            x0, _ = to_px(gx, oy)
            pygame.draw.line(screen, GRID, (x0, 0), (x0, H), 1)
            gx += step
        gy = math.floor(oy / step) * step
        while gy <= oy + scale:
            _, y0 = to_px(ox, gy)
            pygame.draw.line(screen, GRID, (0, y0), (W, y0), 1)
            gy += step

        if show_cover:
            for gpoly, color in zip(guard_polys, palette):
                gpx = [to_px(x, y) for x, y in gpoly]
                if len(gpx) >= 3:
                    surf = pygame.Surface((W, H), pygame.SRCALPHA)
                    pygame.draw.polygon(surf, (*color, 60), gpx)
                    screen.blit(surf, (0, 0))
                    pygame.draw.polygon(screen, (*color, 200), gpx, 1)

        px_pts = [to_px(x, y) for x, y in pts]
        pygame.draw.lines(screen, RED, True, px_pts, 2)
        for i, p in enumerate(px_pts):
            color = CYAN if i in reflex else WHITE
            r = 5 if i in reflex else 3
            pygame.draw.circle(screen, color, p, r)
            screen.blit(font_small.render(str(i), True, (170, 170, 170)),
                       (p[0] + 6, p[1] - 14))

        if show_cover:
            for gi, (guard_x, guard_y) in enumerate(guards):
                gp = to_px(guard_x, guard_y)
                pygame.draw.circle(screen, GOLD, gp, 8)
                pygame.draw.circle(screen, (0, 0, 0), gp, 8, 1)
                screen.blit(font_small.render(str(gi + 1), True, (0, 0, 0)),
                           (gp[0] - 3, gp[1] - 5))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        status_color = WHITE if simple else ORANGE
        lines = [
            (f'{name} — {len(pts)} vertices, {len(reflex)} reflex (cyan)', WHITE),
            (f'area = {area:.1f}', WHITE),
            ('simple polygon' if simple else 'WARNING: self-intersecting',
             status_color),
        ]
        if show_cover:
            cov_str = f'{coverage_pct:.1f}%' if coverage_pct is not None else 'n/a'
            lines.append((f'minimum guard cover: {len(guards)} guards (gold), '
                          f'{cov_str} area coverage', GOLD))
        for i, (text, color) in enumerate(lines):
            screen.blit(font.render(text, True, color), (10, 8 + i * 18))

        pygame.display.flip()

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Quick read-only polygon viewer')
    parser.add_argument('polygon', nargs='?', default='current',
                        help="'current' (default), a preset name (polyN), or a CSV path")
    parser.add_argument('--cover', action='store_true',
                        help='Also run KER+ILP guard placement and overlay the '
                             'minimum guard cover and each guard\'s visibility area')
    args = parser.parse_args()
    view_polygon(args.polygon, show_cover=args.cover)


if __name__ == '__main__':
    main()
