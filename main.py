"""
Entry point for the KER pursuit-evasion simulation.

Run:  python main.py

Controls:
  Drag RED dot     — move evader anywhere inside polygon
  Drag GREEN dot   — slide observer/pursuer along the patrol path
  Drag CYAN dot    — cycle active corner
  A                — toggle auto-evader (Voronoi skeleton)
  P                — toggle roadmap pursuer
  Esc              — quit
"""
import sys
import argparse
import threading

from PyQt5.QtWidgets import QApplication

from draw_polygon import draw_polygon
from window import Window


def startup():
    parser = argparse.ArgumentParser(description='KER pursuit-evasion simulation')
    parser.add_argument('-r', '--recompute', action='store_true',
                        help='Ignore cached results and recompute from scratch')
    args, qt_args = parser.parse_known_args()
    if args.recompute:
        print('[INFO] --recompute flag set: ignoring cache.')

    poly = draw_polygon()
    if poly is None:
        print('[ERROR] No polygon drawn — exiting.')
        return

    app    = QApplication([sys.argv[0]] + qt_args)
    window = Window()
    thread = threading.Thread(
        target=window.run, args=(poly,),
        kwargs={'force_recompute': args.recompute},
        daemon=True)
    thread.start()
    sys.exit(app.exec())


if __name__ == '__main__':
    startup()
