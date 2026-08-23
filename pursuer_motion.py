"""Stable-node roadmap motion controller — pure logic, no Qt/rendering.

Shared by the interactive window (window.py) and the headless benchmark
harness (benchmark/). Replans a Dijkstra path to the guard every frame but
commits to intermediate roadmap nodes, stripping the two replanning
artifacts (AT-POS and ARTIFACT nodes) that make naive per-frame Dijkstra
oscillate around mid-edge virtual endpoints.
"""
import math

from shapely.geometry import Point

from graph import dijkstra


def _node_eq(a, b, tol=0.5):
    return math.hypot(a[0] - b[0], a[1] - b[1]) < tol


class StableNodeController:
    """Drives a point along a roadmap graph toward a per-frame moving guard."""

    def __init__(self, pos=None):
        self.pos = list(pos) if pos is not None else None
        self.base_path: list = []
        self.base_idx: int = 0
        self.guard_pos = None
        self.edge_behind = None
        self.direct: bool = False

    def reset(self, pos):
        self.pos = list(pos)
        self.base_path = []
        self.base_idx = 0
        self.guard_pos = None
        self.edge_behind = None
        self.direct = False

    def step(self, graph: dict, guard, speed: float, log=None):
        """Advance up to `speed` world units toward `guard` along `graph`.

        graph — dict adjacency {(x,y): {(x,y): weight}} (SimulationData.graph)
        guard — (x, y) target position on the roadmap
        speed — distance budget for this frame (already dt-scaled)
        log   — optional callable(str) for debug tracing

        Returns the new (x, y) position (also stored in self.pos).
        """
        gx, gy = guard[0], guard[1]
        px, py = self.pos

        # Replan every frame (stable-node approach)
        plan_dist, wp = dijkstra(graph, (px, py), Point(gx, gy))
        if len(wp) < 2:
            nearest = min(graph.keys(),
                          key=lambda n: math.hypot(n[0] - gx, n[1] - gy))
            plan_dist, wp = dijkstra(graph, (px, py), Point(*nearest))

        new_base = wp[1:-1]

        while new_base and math.hypot(new_base[0][0] - px, new_base[0][1] - py) < 0.5:
            at_node = new_base[0]
            if log:
                log(f'STRIP AT-POS node={at_node} (pursuer already here)')
            self.edge_behind = at_node
            if (self.base_idx < len(self.base_path) and
                    _node_eq(self.base_path[self.base_idx], at_node)):
                self.base_idx += 1
            new_base = new_base[1:]

        if len(new_base) == 0:
            if self.base_path and log:
                log(f'DIRECT (same edge) pursuer=({px:.1f},{py:.1f}) '
                    f'guard=({gx:.1f},{gy:.1f})')
            self.base_path = []
            self.base_idx = 0
            self.direct = True
        else:
            cur_target = (self.base_path[self.base_idx]
                          if self.base_idx < len(self.base_path)
                          else None)
            _new_base = new_base[:]
            if (self.edge_behind is not None and
                    _new_base and
                    _node_eq(_new_base[0], self.edge_behind) and
                    (
                        (cur_target is not None and len(_new_base) >= 2 and
                         _node_eq(_new_base[1], cur_target))
                        or
                        (cur_target is None and self.direct)
                    )):
                if log:
                    log(f'STRIP ARTIFACT node={_new_base[0]}')
                _new_base.pop(0)

            if not _new_base:
                self.base_path = []
                self.base_idx = 0
                self.direct = True
                if log:
                    log('SAME EDGE — direct to guard')
            elif cur_target is not None and _node_eq(_new_base[0], cur_target):
                self.base_path = self.base_path[:self.base_idx] + _new_base
                self.direct = False
                if log:
                    log(f'TAIL-UPDATE next={cur_target} new_tail_len={len(_new_base)}')
            else:
                self.direct = False
                if log:
                    log(f'NODE-CHANGE old_next={cur_target} new_next={_new_base[0]} '
                        f'pursuer=({px:.1f},{py:.1f}) guard=({gx:.1f},{gy:.1f}) '
                        f'path_len={plan_dist:.1f} base_nodes={_new_base}')
                self.base_path = _new_base
                self.base_idx = 0

        self.guard_pos = (gx, gy)

        # Walk toward current committed target
        rem = speed
        while rem > 0:
            if self.base_idx < len(self.base_path):
                tx, ty = self.base_path[self.base_idx]
            elif self.guard_pos is not None:
                tx, ty = self.guard_pos
            else:
                break
            dx, dy = tx - px, ty - py
            d_wp = math.hypot(dx, dy)
            if d_wp < 1e-4:
                if self.base_idx < len(self.base_path):
                    self.edge_behind = (tx, ty)
                    self.base_idx += 1
                else:
                    break
                continue
            if d_wp <= rem:
                px, py = tx, ty
                rem -= d_wp
                if self.base_idx < len(self.base_path):
                    self.edge_behind = (tx, ty)
                    self.base_idx += 1
                else:
                    break
            else:
                if self.base_idx < len(self.base_path) or self.direct:
                    px += (dx / d_wp) * rem
                    py += (dy / d_wp) * rem
                rem = 0

        self.pos = [px, py]
        return px, py
