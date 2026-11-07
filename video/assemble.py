#!/usr/bin/env python3
"""Encode the 2D frame sequences, cut and caption the Gazebo takes, concatenate
every segment in storyboard order and produce the final H.264 file.

  python3 video/assemble.py video/out [--target-mb 90]
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402

from common import BG, FPS, GRID, H, MUTED, TEXT, W  # noqa: E402

FONT = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans.ttf')
FONTB = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans-Bold.ttf')
GZ_TITLE = 'Closed loop in ROS 2 and Gazebo'
GZ_TAG = 'ROS 2 Humble, Gazebo Fortress, real time'
TOOLBAR = 50  # px of Gazebo GUI toolbar to crop from the top of the capture


def run(cmd):
    subprocess.run(cmd, check=True)


def dur(p):
    return float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p]))


def frames_to_mp4(frames_dir, out):
    run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', os.path.join(frames_dir, '%05d.jpg'),
         '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p', out])


def gz_cut(src, a, b, caption, out, subtitle):
    """Crop the toolbar, fit the 3D view above the caption band, draw the chrome."""
    capf = out + '.caption.txt'
    open(capf, 'w').write(caption.replace('%', '%%'))   # drawtext expands a bare %
    band_y = H - 120
    vf = (f'crop=in_w:in_h-{TOOLBAR}:0:{TOOLBAR},scale=-2:{band_y}:flags=lanczos,'
          f'pad={W}:{H}:(ow-iw)/2:0:color={BG},'
          f'drawbox=x=0:y={band_y}:w={W}:h=2:color={GRID}:t=fill,'
          f'drawbox=x=28:y=18:w=1060:h=86:color=white@0.88:t=fill,'
          f'drawtext=fontfile={FONTB}:text=\'{GZ_TITLE}\':fontsize=34:fontcolor={TEXT}:x=40:y=28,'
          f'drawtext=fontfile={FONT}:text=\'{subtitle}\':fontsize=19:fontcolor={MUTED}:x=40:y=72,'
          f'drawtext=fontfile={FONT}:textfile={capf}:fontsize=27:line_spacing=8:fontcolor={TEXT}:x=40:y={band_y}+(120-th)/2,'
          f'drawtext=fontfile={FONT}:text=\'{GZ_TAG}\':fontsize=17:fontcolor={MUTED}:x=w-tw-40:y=h-34')
    run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{a:.2f}', '-to', f'{b:.2f}', '-i', src, '-vf', vf, '-r', str(FPS),
         '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p', '-an', out])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--target-mb', type=float, default=90.0)
    ap.add_argument('--final', default='RoadmapTracking_video.mp4')
    a = ap.parse_args()
    S = a.out
    seq = [f'{S}/seg0_title.mp4']
    for seg in ('problem', 'roadmap', 'race'):
        frames_to_mp4(f'{S}/frames/{seg}', f'{S}/seg_{seg}.mp4')
        seq.append(f'{S}/seg_{seg}.mp4')
    seq.append(f'{S}/seg4_results.mp4')
    frames_to_mp4(f'{S}/frames/team', f'{S}/seg_team.mp4')
    seq.append(f'{S}/seg_team.mp4')

    # Gazebo: proposed take, then a short baseline take under the same dynamics
    gza = f'{S}/gz_alpha/gz_raw.mp4'
    sub_a = 'Min-Max +Vis at 5 Hz; hexacopter with per-rotor thrust dynamics; Spot evader; poly9 at 0.05 m per unit'
    cuts = [(gza, 0, 9, 'The planner runs unmodified as a 5 Hz ROS 2 node: envelope optimizer, ε-slack layer and stable-node controller.\n'
                        'poly9 at 0.05 m per unit with 3 m walls; a hexacopter tracks the roadmap state overhead, a walking Spot plays the evader.'),
            (gza, 10, 21, 'The loop closes both ways: the planner reads the vehicle\'s odometry and holds its target whenever the hexacopter lags\n'
                          'more than 2 m behind the planned state.'),
            (gza, 24, 34, 'Five 180 s flights per strategy: acceleration RMS 1.9 to 2.0 m/s² against the 2.5 m/s² limit,\n'
                          'about six commanded direction reversals per minute, mean tracking lag 1.2 to 1.3 m.'),
            (gza, 36, 46, 'Physical line of sight 62 to 63%, within ten points of the kinematic planner.')]
    parts = [gz_cut(src, s, e, cap, f'{S}/gzcut_{i}.mp4', sub_a) for i, (src, s, e, cap) in enumerate(cuts)]
    gzg = f'{S}/gz_geo/gz_raw.mp4'
    if os.path.exists(gzg):
        sub_g = 'Geo-Follow baseline, identical dynamics: chases the roadmap vertex nearest the evader'
        parts.append(gz_cut(gzg, 6, 14, 'Geo-Follow under the same dynamics jumps between roadmap vertices: 40 direction reversals per minute,\n'
                                        'outrunning the vehicle on 55% of ticks and raising the realized acceleration RMS to 10 m/s².',
                            f'{S}/gzcut_geo.mp4', sub_g))
    seq += parts + [f'{S}/seg7_summary.mp4']

    with open(f'{S}/concat.txt', 'w') as f:
        for p in seq:
            f.write(f"file '{os.path.abspath(p)}'\n")
    total = sum(dur(p) for p in seq)
    print(f'total duration {total:.1f} s over {len(seq)} segments')
    kbps = int(a.target_mb * 8 * 1024 * 1024 / total / 1000 * 0.96)
    print(f'target bitrate {kbps} kbps')
    common = ['-c:v', 'libx264', '-preset', 'slow', '-b:v', f'{kbps}k', '-maxrate', f'{int(kbps * 1.6)}k',
              '-bufsize', f'{kbps * 3}k', '-pix_fmt', 'yuv420p', '-r', str(FPS), '-an']
    run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', f'{S}/concat.txt', *common,
         '-pass', '1', '-passlogfile', f'{S}/x264', '-f', 'null', '/dev/null'])
    final = os.path.join(S, a.final)
    run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', f'{S}/concat.txt', *common,
         '-pass', '2', '-passlogfile', f'{S}/x264', '-movflags', '+faststart', final])
    print(f'{final}: {os.path.getsize(final) / 1e6:.1f} MB, {dur(final):.1f} s')


if __name__ == '__main__':
    main()
