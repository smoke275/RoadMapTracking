"""
Read-only polygon viewer — no drawing, no KER pipeline build, just a quick
look at a saved polygon (vertex order, reflex corners, area, self-intersection).

Run:  python view_polygon.py               # views the current polygon (config.FILE_NAME)
      python view_polygon.py poly2         # views a preset (resources/sites_poly2.csv)
      python view_polygon.py my_env.csv    # views a specific file
"""
import argparse
import math

import pygame
from shapely.geometry import Polygon as ShapelyPolygon

from benchmark.harness import load_poly
from geometry import clean_polygon


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


def view_polygon(name: str = 'current'):
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
    pygame.display.set_caption(f'Polygon viewer — {name} ({len(pts)} vertices) — ESC to close')
    font       = pygame.font.SysFont('monospace', 15)
    font_small = pygame.font.SysFont('monospace', 11)

    WHITE, GRAY, GRID = (255, 255, 255), (60, 60, 60), (80, 80, 80)
    RED, CYAN, ORANGE = (220, 70, 70), (0, 220, 255), (240, 140, 40)

    def to_px(px, py):
        return int((px - ox) * W / scale), int(H - (py - oy) * H / scale)

    reflex = [i for i in range(len(pts)) if _is_reflex(pts, i)]
    area = abs(_shoelace(pts))
    simple = _is_simple(pts)

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

        px_pts = [to_px(x, y) for x, y in pts]
        pygame.draw.lines(screen, RED, True, px_pts, 2)
        for i, p in enumerate(px_pts):
            color = CYAN if i in reflex else WHITE
            r = 5 if i in reflex else 3
            pygame.draw.circle(screen, color, p, r)
            screen.blit(font_small.render(str(i), True, (170, 170, 170)),
                       (p[0] + 6, p[1] - 14))

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
        for i, (text, color) in enumerate(lines):
            screen.blit(font.render(text, True, color), (10, 8 + i * 18))

        pygame.display.flip()

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Quick read-only polygon viewer')
    parser.add_argument('polygon', nargs='?', default='current',
                        help="'current' (default), a preset name (polyN), or a CSV path")
    args = parser.parse_args()
    view_polygon(args.polygon)


if __name__ == '__main__':
    main()
