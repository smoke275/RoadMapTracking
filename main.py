"""
Entry point for the KER pursuit-evasion simulation.

Run:  python main.py
      python main.py --skip-draw   # skip the drawing tool, load the saved polygon

Controls:
  Drag RED dot     — move evader anywhere inside polygon
  Drag GREEN dot   — slide observer/pursuer along the patrol path
  Drag CYAN dot    — cycle active corner
  A                — toggle auto-evader (Voronoi skeleton)
  P                — toggle roadmap pursuer
  L                — toggle line-of-sight forecast (Φ score + escape path colouring)
  S                — cycle pursuer strategy (incl. 'watchman-patrol', which walks
                     the shortest watchman route, see watchman_route.py)
  W                — toggle the shortest watchman route overlay
  Esc              — quit

With config.NUM_PURSUERS > 1 the k-pursuer window (multi_window.py) opens
instead; see its docstring for that mode's controls.
"""
import sys
import argparse
import threading

from PyQt5.QtWidgets import QApplication

from config import NUM_PURSUERS
from draw_polygon import draw_polygon, load_polygon
from window import Window


def startup():
    parser = argparse.ArgumentParser(description='KER pursuit-evasion simulation')
    parser.add_argument('-r', '--recompute', action='store_true',
                        help='Ignore cached results and recompute from scratch')
    parser.add_argument('-s', '--skip-draw', action='store_true',
                        help='Skip the interactive polygon-drawing tool and load '
                             'the polygon straight from resources/ (config.FILE_NAME)')
    parser.add_argument('-w', '--watchman', action='store_true',
                        help='Start with the shortest watchman route overlay on '
                             '(same as pressing W)')
    parser.add_argument('--roadmap', choices=['ker', 'watchman'], default=None,
                        help='Patrol roadmap the pursuer is confined to: the KER '
                             'guard graph or the shortest watchman route '
                             '(default: config.ROADMAP_MODE)')
    args, qt_args = parser.parse_known_args()
    if args.roadmap:
        import config
        config.ROADMAP_MODE = args.roadmap
    if args.recompute:
        print('[INFO] --recompute flag set: ignoring cache.')

    if args.skip_draw:
        poly = load_polygon()
        if poly is None:
            print('[ERROR] --skip-draw: polygon file has fewer than 3 points — exiting.')
            return
    else:
        poly = draw_polygon()
        if poly is None:
            print('[ERROR] No polygon drawn — exiting.')
            return

    app    = QApplication([sys.argv[0]] + qt_args)
    if NUM_PURSUERS > 1:
        # k-pursuer tracking: corners partitioned into NUM_PURSUERS groups,
        # one roadmap pursuer per group (see corner_groups.py).
        from multi_window import MultiPursuitWindow
        window = MultiPursuitWindow(NUM_PURSUERS)
    else:
        window = Window()
        window._show_watchman = args.watchman
    thread = threading.Thread(
        target=window.run, args=(poly,),
        kwargs={'force_recompute': args.recompute},
        daemon=True)
    thread.start()
    sys.exit(app.exec())


if __name__ == '__main__':
    startup()
