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
import threading

from PyQt5.QtWidgets import QApplication

from draw_polygon import draw_polygon
from window import Window


def startup():
    poly = draw_polygon()
    if poly is None:
        print('[ERROR] No polygon drawn — exiting.')
        return

    app    = QApplication(sys.argv)
    window = Window()
    thread = threading.Thread(target=window.run, args=(poly,), daemon=True)
    thread.start()
    sys.exit(app.exec())


if __name__ == '__main__':
    startup()
