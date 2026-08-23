"""Pygame-based polygon drawing tool — loads/saves to FILE_NAME."""
import csv
import math

import pygame
import visilibity as vis
from shapely.geometry import Polygon as ShapelyPolygon

from config import FILE_NAME

CLOSE_SNAP_PX = 12   # click within this many pixels of the start point to close


def load_polygon() -> list:
    """Load the polygon straight from FILE_NAME, skipping the interactive
    pygame drawing tool. Returns None if the file has fewer than 3 points."""
    points = []
    with open(FILE_NAME) as f:
        for row in csv.reader(f):
            points.append((float(row[0]), float(row[1])))
    if len(points) < 3:
        return None
    return [vis.Point(p[0], p[1]) for p in points]


def _shoelace(pts) -> float:
    n = len(pts)
    return sum((pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1])
               for i in range(n)) / 2


def _is_simple(pts) -> bool:
    """True if the closed polygon has no self-intersections."""
    if len(pts) < 3:
        return True
    try:
        return ShapelyPolygon(pts).is_valid
    except Exception:
        return False


def draw_polygon() -> list:
    pygame.init()
    W, H   = 700, 700
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Draw Polygon — click to close, SPACE to close, "
                               "BACKSPACE to undo, ESC to quit")

    WHITE    = (255, 255, 255)
    RED      = (220, 50, 50)
    GRAY     = (60, 60, 60)
    GRID     = (80, 80, 80)
    GREEN    = (80, 220, 120)
    YELLOW   = (230, 200, 60)
    ORANGE   = (240, 140, 40)
    START_PT = (100, 200, 255)

    def to_px(px, py):
        return int((px + 500) * W / 1000), int(H - (py + 500) * H / 1000)

    def from_px(px, py):
        return px * 1000 / W - 500, 500 - py * 1000 / H

    points = []
    closed = False
    font       = pygame.font.SysFont("monospace", 15)
    font_small = pygame.font.SysFont("monospace", 12)

    try:
        with open(FILE_NAME) as f:
            for row in csv.reader(f):
                points.append((float(row[0]), float(row[1])))
    except FileNotFoundError:
        pass   # FILE_NAME doesn't exist yet — start from a blank canvas
    loaded_existing = len(points) > 0
    if loaded_existing:
        closed = True   # loaded polygon is already a finished shape

    def draw_grid():
        for wx in range(-500, 501, 100):
            x0, y0 = to_px(wx, -500)
            x1, y1 = to_px(wx, 500)
            pygame.draw.line(screen, GRID, (x0, y0), (x1, y1), 1)
        for wy in range(-500, 501, 100):
            x0, y0 = to_px(-500, wy)
            x1, y1 = to_px(500, wy)
            pygame.draw.line(screen, GRID, (x0, y0), (x1, y1), 1)
        # origin marker
        ox, oy = to_px(0, 0)
        pygame.draw.line(screen, (110, 110, 110), (ox - 8, oy), (ox + 8, oy), 1)
        pygame.draw.line(screen, (110, 110, 110), (ox, oy - 8), (ox, oy + 8), 1)

    def blit_lines(lines, x, y, size_font):
        for i, (text, color) in enumerate(lines):
            screen.blit(size_font.render(text, True, color), (x, y + i * 18))

    running = True
    while running:
        screen.fill(GRAY)
        draw_grid()

        mx, my = pygame.mouse.get_pos()
        near_start = (not closed and len(points) >= 3
                     and math.hypot(mx - to_px(*points[0])[0],
                                    my - to_px(*points[0])[1]) < CLOSE_SNAP_PX)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif not closed and event.type == pygame.MOUSEBUTTONDOWN:
                cx, cy = event.pos
                click_near_start = (len(points) >= 3 and math.hypot(
                    cx - to_px(*points[0])[0], cy - to_px(*points[0])[1]) < CLOSE_SNAP_PX)
                if click_near_start:
                    closed = True
                else:
                    points.append(from_px(cx, cy))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and len(points) > 2:
                    closed = True
                elif event.key == pygame.K_p:
                    print(points)
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_n:
                    points, closed = [], False
                elif event.key == pygame.K_BACKSPACE and points:
                    if closed:
                        closed = False   # reopen for editing instead of popping the last vertex
                    else:
                        points.pop()

        # ---- draw ------------------------------------------------------
        px_pts = [to_px(px, py) for px, py in points]
        if len(px_pts) > 1:
            pygame.draw.lines(screen, RED, closed, px_pts, 2)

        if not closed and px_pts:
            # rubber-band preview of the next segment
            preview_color = GREEN if near_start else YELLOW
            pygame.draw.line(screen, preview_color, px_pts[-1], (mx, my), 1)

        for i, p in enumerate(px_pts):
            if i == 0 and not closed and len(points) >= 3:
                color, r = (GREEN if near_start else START_PT), (7 if near_start else 5)
            else:
                color, r = WHITE, 3
            pygame.draw.circle(screen, color, p, r)

        # ---- HUD ---------------------------------------------------------
        simple_ok = closed and _is_simple(points)
        status = []
        if closed:
            area = abs(_shoelace(points))
            status.append((f'Polygon CLOSED — {len(points)} pts, area {area:.0f}',
                          WHITE if simple_ok else ORANGE))
            if not simple_ok:
                status.append(('WARNING: self-intersecting polygon — fix before saving',
                              ORANGE))
        else:
            area = abs(_shoelace(points)) if len(points) > 2 else 0.0
            status.append((f'Drawing — {len(points)} pts, area so far {area:.0f}',
                          WHITE))
            if near_start:
                status.append(('Click to CLOSE here', GREEN))
        status.append((f'cursor: ({from_px(mx, my)[0]:.0f}, {from_px(mx, my)[1]:.0f})',
                       (170, 170, 170)))
        blit_lines(status, 10, 8, font)

        hints = [
            ('click: add point   click near start / SPACE: close', (170, 170, 170)),
            ('BACKSPACE: undo point (or reopen if closed)   N: new   P: print   ESC: done',
             (170, 170, 170)),
        ]
        blit_lines(hints, 10, H - 44, font_small)

        pygame.display.flip()

    pygame.quit()

    if not closed or len(points) < 3:
        return None
    if not _is_simple(points):
        print('[WARNING] Saved polygon is self-intersecting — the simulation '
              'pipeline will likely fail or misbehave on it. Redraw before using.')

    if _shoelace(points) > 0:
        points = list(reversed(points))
    print(f'Area = {abs(_shoelace(points)):.1f}')

    poly = [vis.Point(p[0], p[1]) for p in points]
    with open(FILE_NAME, 'w') as f:
        for p in points:
            f.write(f'{p[0]:.2f}, {p[1]:.2f}\n')
    return poly
