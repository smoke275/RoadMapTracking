#!/usr/bin/env python3
"""Title, animated result cards and summary -> mp4 segments (1920x1080, 30 fps).
Runs on the host (matplotlib + PIL + ffmpeg). Numbers are the paper's
(RoadmapBasedTracking/sections/07_experiments.tex, Tables I-V and Sec. VII).

  python3 video/cards.py video/out
"""
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BG, BLUE, FPS, GREEN, GRID, H, MUTED, ORANGE, PURPLE,  # noqa: E402
                    RED, TEAL, TEXT, W, YELLOW, ease, font, rgb)

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': BG, 'axes.facecolor': BG})

# ------------------------------------------------------------- paper data
POLYS = ['poly1', 'poly2', 'poly3', 'poly4', 'poly5', 'poly6', 'poly7', 'poly8', 'poly9']
STRATS = ['Min-Max', 'Min-Max +Vis', 'Geo-Follow', 'Greedy-LOS', 'TSP-Patrol', 'Kernel-Weighted']
ALPHA = {  # Table I, mean worst-corner ratio per polygon
    'Min-Max': [2.49, 1.39, 1.56, 1.54, 2.25, 3.10, 1.73, 4.58, 1.51],
    'Min-Max +Vis': [2.48, 1.39, 1.56, 1.59, 2.09, 3.07, 1.73, 4.74, 1.52],
    'Geo-Follow': [3.19, 2.41, 1.69, 2.47, 2.93, 3.11, 2.16, 4.81, 1.93],
    'Greedy-LOS': [3.15, 2.33, 1.60, 2.47, 2.89, 3.12, 2.19, 4.66, 1.95],
    'TSP-Patrol': [7.31, 5.11, 4.90, 5.71, 4.51, 4.66, 5.83, 10.54, 5.14],
    'Kernel-Weighted': [4.36, 3.62, 2.47, 2.75, 3.04, 3.07, 2.57, 6.49, 2.37]}
LOS = {  # Table I, %LOS per polygon
    'Min-Max': [85.8, 95.6, 88.8, 88.4, 86.0, 82.4, 83.3, 64.8, 86.8],
    'Min-Max +Vis': [87.9, 98.8, 90.8, 90.0, 93.7, 84.3, 86.2, 65.6, 87.5],
    'Geo-Follow': [86.5, 85.8, 93.0, 97.0, 92.5, 87.5, 89.2, 61.9, 88.7],
    'Greedy-LOS': [87.8, 92.0, 98.7, 97.7, 93.0, 87.2, 92.0, 63.1, 94.4],
    'TSP-Patrol': [42.2, 64.5, 30.8, 48.8, 56.4, 57.8, 62.1, 10.9, 25.2],
    'Kernel-Weighted': [57.3, 82.0, 77.2, 91.7, 90.6, 83.9, 93.7, 45.9, 81.3]}
SSTAR = [1.907, 1.677, 1.833, 2.713, 1.945, 2.168, 1.418, 1.738, 1.682]  # Table II
COLORS = {'Min-Max': BLUE, 'Min-Max +Vis': TEAL, 'Geo-Follow': ORANGE, 'Greedy-LOS': YELLOW,
          'TSP-Patrol': '#9a9a9a', 'Kernel-Weighted': PURPLE}

ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                       '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium',
                       '-crf', '17', '-pix_fmt', 'yuv420p', f'{OUT}/seg4_results.mp4'], stdin=subprocess.PIPE)


def emit(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    ff.stdin.write(np.ascontiguousarray(buf).tobytes())


def frame_chrome(fig, title, sub, caption):
    fig.text(40 / W, 1 - 46 / H, title, fontsize=25, fontweight='bold', color=TEXT, va='center')
    fig.text(40 / W, 1 - 86 / H, sub, fontsize=15.5, color=MUTED, va='center')
    fig.patches.append(plt.Rectangle((0, 0), 1, 120 / H, transform=fig.transFigure, color=BG, zorder=5))
    fig.add_artist(plt.Line2D([0, 1], [120 / H, 120 / H], transform=fig.transFigure, color=GRID, lw=1.5, zorder=6))
    lines = caption.split('\n')
    y0 = 66 / H + (len(lines) - 1) * 17 / H
    for i, ln in enumerate(lines):
        fig.text(40 / W, y0 - i * 34 / H, ln, fontsize=21 if len(lines) == 1 else 19, color=TEXT, va='center', zorder=7)
    fig.text(1 - 40 / W, 13 / H, 'Numbers from the paper, 50 paired seeds per configuration', fontsize=11, color=MUTED, ha='right', va='center', zorder=7)


def style(ax, ylabel):
    ax.set_ylabel(ylabel, color=TEXT, fontsize=15)
    ax.grid(True, axis='y', color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=12)


def run_card(build, seconds, anim=1.6):
    n, na = int(seconds * FPS), int(anim * FPS)
    for i in range(n):
        t = ease(i / max(1, na - 1)) if i < na else 1.0
        fig = build(t)
        emit(fig)
        plt.close(fig)


def card_tracking(t):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    axs = [fig.add_axes([0.07, 0.24, 0.40, 0.58]), fig.add_axes([0.57, 0.24, 0.40, 0.58])]
    x = np.arange(len(STRATS))
    for ax, table, ylabel, better in zip(axs, (ALPHA, LOS), ('mean worst-corner ratio ᾱ (lower is better)', '%LOS (higher is better)'), ('min', 'max')):
        vals = [np.mean(table[s]) for s in STRATS]
        bars = ax.bar(x, [v * t for v in vals], color=[COLORS[s] for s in STRATS], width=0.66, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * t + (0.08 if table is ALPHA else 1.2), f'{v:.2f}' if table is ALPHA else f'{v:.0f}', ha='center', va='bottom', fontsize=13, color=TEXT)
        ax.set_xticks(x)
        ax.set_xticklabels(STRATS, rotation=18, ha='right')
        style(ax, ylabel)
        ax.set_ylim(0, (max(vals) * 1.18) if table is ALPHA else 100)
        for lbl in ax.get_xticklabels()[:2]:
            lbl.set_fontweight('bold')
            lbl.set_color(BLUE)
    frame_chrome(fig, 'Single pursuer, nine polygons, five baselines', 'Skeleton evader, 600 frames, s_p / s_e = 0.8; bars are means over the nine environments (Table I)',
                 'Both proposed methods give the lowest ᾱ on all nine environments, up to 42% below Geo-Follow.\n'
                 'The ε-slack layer (+Vis) adds 0.7 to 7.7 points of %LOS over Min-Max while changing ᾱ by at most 0.16.')
    return fig


def card_speed(t):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0.07, 0.24, 0.56, 0.58])
    x = np.arange(len(POLYS))
    bars = ax.bar(x, [s * t for s in SSTAR], color=BLUE, width=0.62, zorder=3)
    for b, s in zip(bars, SSTAR):
        ax.text(b.get_x() + b.get_width() / 2, s * t + 0.04, f'{s:.2f}', ha='center', va='bottom', fontsize=13, color=TEXT)
    ax.axhline(1.0, color=RED, lw=2, ls='--', zorder=4)
    ax.text(len(POLYS) - 0.5, 1.04, 'evader speed', color=RED, ha='right', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(POLYS)
    style(ax, 'critical speed ratio s*  (pursuer / evader)')
    ax.set_ylim(0, 3.1)
    # side tiles
    tiles = [('1.42 to 2.71×', 'speed a single pursuer needs to win every corner race'),
             ('5 to 13', 'guards cover 23 to 60 vertices'),
             ('0.8', 'nominal ratio in the trials: below s* everywhere')]
    for i, (big, small) in enumerate(tiles):
        y = 0.76 - i * 0.19
        fig.text(0.68, y, big, fontsize=34, fontweight='bold', color=BLUE if i == 0 else TEXT, va='center')
        fig.text(0.68, y - 0.065, small, fontsize=14.5, color=MUTED, va='center')
    frame_chrome(fig, 'Critical speed ratio s*', 'Necessary speed bound for single-pursuer cutoff, certified on a 25 × 25 evader grid (Table II)',
                 'On no benchmark can a pursuer at or below the evader\'s speed satisfy the cutoff condition everywhere.\n'
                 'The bound is set by visibility bottlenecks, not size: one deep recess makes poly4 the hardest with only five guards.')
    return fig


def card_sweep(t):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    tiles = [('81% → 96%', '%LOS against the skeleton evader as s_p / s_e rises from 0.6 to 2.0', TEAL),
             ('÷ 3', 'escapes against the skeleton evader over the same sweep', TEAL),
             ('0.56 / trial', 'escapes against the escaping adversary at s_p / s_e ≥ 1.4: the fewest of all methods\n(Greedy-LOS 0.76, Geo-Follow 0.89, plain Min-Max 0.77; paired p ≤ 0.013)', TEAL),
             ('crossover ≈ 1.2', 'below the evader\'s speed, shadowing it concedes fewer escapes;\nwith speed margin, timing-optimal placement wins', MUTED)]
    for i, (big, small, col) in enumerate(tiles):
        y = 0.78 - i * 0.165
        k = ease(min(1.0, max(0.0, t * 1.6 - i * 0.18)))
        fig.text(0.06, y, big, fontsize=40, fontweight='bold', color=col, va='center', alpha=k)
        fig.text(0.40, y + 0.012, small, fontsize=16, color=TEXT, va='center', alpha=k)
    frame_chrome(fig, 'Speed sweep on poly9', 's_p / s_e from 0.6 to 2.0, 50 seeds per ratio, s* = 1.682 (Fig. 5)',
                 'With speed margin, timing-optimal placement wins the corner races and the slack layer turns won races into sight.')
    return fig


run_card(card_tracking, 8.0)
run_card(card_speed, 8.0)
run_card(card_sweep, 8.0, anim=2.4)
ff.stdin.close()
ff.wait()


# ------------------------------------------------------------- PIL cards
def still(lines, out, seconds, accent=True):
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    y = 0
    heights = []
    for text, size, bold, color, gap in lines:
        heights.append(size + gap)
    total = sum(heights)
    y = (H - total) / 2
    if accent:
        d.rectangle([W / 2 - 110, y - 46, W / 2 + 110, y - 38], fill=rgb(BLUE))
    for text, size, bold, color, gap in lines:
        f = font(size, bold)
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=rgb(color))
        y += size + gap
    im.save(out.replace('.mp4', '.png'))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-loop', '1', '-framerate', str(FPS), '-i', out.replace('.mp4', '.png'),
                    '-t', str(seconds), '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p', out], check=True)


still([('From Optimal Placement to Team Tracking', 60, True, TEXT, 8),
       ('on Patrol Roadmaps in Polygons', 60, True, TEXT, 34),
       ('Shashwata Mandal and Sourabh Bhattacharya', 32, False, TEXT, 10),
       ('Iowa State University', 26, False, MUTED, 60),
       ('1.  Why reflex corners decide tracking, and how a patrol roadmap is built', 25, False, TEXT, 14),
       ('2.  Exact per-frame placement, pursuer teams, and results on nine polygons', 25, False, TEXT, 14),
       ('3.  Closed-loop flight of a hexacopter in ROS 2 and Gazebo', 25, False, TEXT, 0)],
      f'{OUT}/seg0_title.mp4', 6)

still([('Summary', 52, True, BLUE, 40),
       ('A patrol roadmap from Keil kernel candidates and an exact set cover puts a node in every association region.', 27, False, TEXT, 14),
       ('Per-frame placement is a weighted network 1-center, solved exactly; a stable-node controller and an ε-slack layer track.', 27, False, TEXT, 14),
       ('The critical speed ratio s* = 1.42 to 2.71× bounds one pursuer; a MILP corner partition brings a team of three to 1.004.', 27, False, TEXT, 14),
       ('Validated against five baselines and three evader models on nine polygons, and in closed-loop flight in Gazebo.', 27, False, TEXT, 60),
       ('github.com/smoke275/RoadMapTracking', 26, True, MUTED, 0)],
      f'{OUT}/seg7_summary.mp4', 8)
print('cards done')
