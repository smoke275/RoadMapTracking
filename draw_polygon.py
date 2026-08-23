"""Pygame-based polygon drawing tool — loads/saves to FILE_NAME."""
import csv

import pygame
import visilibity as vis

from config import FILE_NAME


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


def draw_polygon() -> list:
    pygame.init()
    W, H   = 700, 700
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Draw Polygon — SPACE to close, BACKSPACE to undo, ESC to quit")

    WHITE = (255, 255, 255)
    RED   = (220, 50, 50)
    GRAY  = (60, 60, 60)

    def to_px(px, py):
        return int((px + 500) * W / 1000), int(H - (py + 500) * H / 1000)

    def from_px(px, py):
        return px * 1000 / W - 500, 500 - py * 1000 / H

    def shoelace(pts):
        n = len(pts)
        return sum((pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1])
                   for i in range(n)) / 2

    points = []
    closed = False
    font   = pygame.font.SysFont("monospace", 16)

    with open(FILE_NAME) as f:
        for row in csv.reader(f):
            points.append((float(row[0]), float(row[1])))
    if points:
        points.append(points[0])

    running = True
    while running:
        screen.fill(GRAY)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif not closed and event.type == pygame.MOUSEBUTTONDOWN:
                points.append(from_px(*pygame.mouse.get_pos()))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and len(points) > 2:
                    points.append(points[0])
                    closed = True
                    if shoelace(points[:-1]) > 0:
                        points = list(reversed(points))
                    print(f'Area = {shoelace(points[:-1]):.1f}')
                elif event.key == pygame.K_p:
                    print(points)
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_n:
                    points, closed = [], False
                elif event.key == pygame.K_BACKSPACE and points:
                    points.pop()

        px_pts = [to_px(px, py) for px, py in points]
        if len(px_pts) > 1:
            pygame.draw.lines(screen, RED, False, px_pts, 2)

        hint = "SPACE=close  BACKSPACE=undo  ESC=done"
        screen.blit(font.render(hint, True, WHITE), (10, 10))
        pygame.display.flip()

    pygame.quit()

    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
        poly = [vis.Point(p[0], p[1]) for p in points]
        with open(FILE_NAME, 'w') as f:
            for p in points:
                f.write(f'{p[0]:.2f}, {p[1]:.2f}\n')
        return poly
    return None
