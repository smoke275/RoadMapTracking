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
    args, qt_args = parser.parse_known_args()
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
    thread = threading.Thread(
        target=window.run, args=(poly,),
        kwargs={'force_recompute': args.recompute},
        daemon=True)
    thread.start()
    sys.exit(app.exec())


if __name__ == '__main__':
    startup()
