#!/usr/bin/env python3
"""Record the Gazebo 3D view while steering its camera.

  python3 video/gz_record.py OUT_DIR --strategy alpha-guard --seconds 70 [--shots full|drone]

Starts the sim container with the bare GUI (video/gui.config, Spot evader),
raises the window, records it with ffmpeg x11grab at 30 fps (gz_raw.mp4), and
drives the camera through /gui/move_to/pose and /gui/follow at the times in
SHOTS, logging them to events.json. Keep other windows off the Gazebo window
while it records (x11grab reads the on-screen window)."""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CT = 'ros2_sim_video'
DISPLAY = os.environ.get('DISPLAY', ':1')

SHOTS = {
    'full': [(0, 'overview'), (10, 'drone'), (24, 'spot'), (36, 'drone_wide'), (52, 'overview')],
    'drone': [(0, 'overview'), (4, 'drone')],
}
# camera presets: overview pose (x, y, z, pitch, yaw); follow offsets (target frame)
OVERVIEW = (0.0, -30.0, 36.0, 0.9, 1.5708)
OFFSETS = {'drone': (-6.0, -6.0, 3.5), 'drone_wide': (-12.0, -10.0, 9.0), 'spot': (-3.2, -3.2, 1.6)}


def sh(cmd, check=False):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)


def ign(cmd):
    return sh(f"docker exec {CT} bash -c 'source /opt/ros/humble/setup.bash; {cmd}'")


def quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def move_to(x, y, z, pitch, yaw):
    qx, qy, qz, qw = quat(0.0, pitch, yaw)
    ign('ign service -s /gui/follow --reqtype ignition.msgs.StringMsg --reptype ignition.msgs.Boolean --timeout 2000 --req "data: \\"\\""')
    return ign(f'ign service -s /gui/move_to/pose --reqtype ignition.msgs.GUICamera --reptype ignition.msgs.Boolean --timeout 2000 '
               f'--req "pose: {{position: {{x: {x}, y: {y}, z: {z}}}, orientation: {{x: {qx:.5f}, y: {qy:.5f}, z: {qz:.5f}, w: {qw:.5f}}}}}"')


def follow(name, off):
    ign(f'ign service -s /gui/follow/offset --reqtype ignition.msgs.Vector3d --reptype ignition.msgs.Boolean --timeout 2000 --req "x: {off[0]}, y: {off[1]}, z: {off[2]}"')
    return ign(f'ign service -s /gui/follow --reqtype ignition.msgs.StringMsg --reptype ignition.msgs.Boolean --timeout 2000 --req "data: \\"{name}\\""')


def shot(kind):
    if kind == 'overview':
        return move_to(*OVERVIEW)
    if kind.startswith('drone'):
        return follow('simple_drone', OFFSETS[kind])
    if kind == 'spot':
        return follow('spot_evader', OFFSETS['spot'])


def window_ids():
    tree = sh(f'DISPLAY={DISPLAY} xwininfo -root -tree').stdout
    gui = re.search(r'(0x[0-9a-f]+) "Gazebo": \("ign-gazebo-gui"', tree)
    frame = re.search(r'(0x[0-9a-f]+) "Gazebo": \("mutter-x11-frames"', tree)
    return (gui.group(1) if gui else None), (frame.group(1) if frame else None)


def geometry(wid):
    out = sh(f'DISPLAY={DISPLAY} xwininfo -id {wid}').stdout
    w = int(re.search(r'Width:\s+(\d+)', out).group(1))
    h = int(re.search(r'Height:\s+(\d+)', out).group(1))
    return w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--strategy', default='alpha-guard')
    ap.add_argument('--seconds', type=float, default=70)
    ap.add_argument('--shots', default='full', choices=list(SHOTS))
    ap.add_argument('--evader', default='spot')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--settle', type=float, default=6.0, help='seconds after the planner starts before recording')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sh(f'docker rm -f {CT}')
    sh('xhost +local:docker')
    run = (f'docker run -d --name {CT} --net=host -e DISPLAY={DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix:rw '
           f'-e QT_X11_NO_MITSHM=1 -e ROS_DOMAIN_ID=48 -e IGN_PARTITION=video_{a.strategy} '
           f'--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all --device /dev/dri -v "{REPO}":/app ros2_sim bash -c "'
           f'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && '
           f'ros2 launch ros2_sim sim.launch.py rviz:=false evader_type:={a.evader} strategy:={a.strategy} '
           f'evader_seed:={a.seed} gui_config:=/app/video/gui.config log_csv:=/tmp/video.csv"')
    sh(run, check=True)
    print('container started; waiting for the planner', flush=True)
    t0 = time.time()
    while time.time() - t0 < 150:
        log = sh(f'docker logs {CT} 2>&1').stdout
        if 'process has died' in log or 'ModuleNotFound' in log:
            print(log[-3000:])
            sys.exit('launch failed')
        if 'strategy=' in log and 'OK creation of entity' in log:
            break
        time.sleep(1)
    else:
        print(sh(f'docker logs {CT} 2>&1').stdout[-3000:])
        sys.exit('planner never started (Gazebo hung at startup? rerun)')
    print(f'planner up after {time.time() - t0:.0f} s', flush=True)
    time.sleep(a.settle)

    gui, frame = window_ids()
    if not gui:
        sys.exit('no Gazebo window found')
    sh(f'docker run --rm -e DISPLAY={DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix:rw xdo-helper bash -c '
       f'"xdotool windowmove {frame} 0 0; xdotool windowactivate --sync {frame}; xdotool windowraise {gui}"')
    time.sleep(1.0)
    w, h = geometry(gui)
    print(f'window {gui} {w}x{h}', flush=True)
    shot('overview')
    time.sleep(1.5)
    # x11grab reads the on-screen window: make sure it is really in front
    for attempt in range(5):
        probe = f'{a.out}/probe.png'
        if os.path.exists(probe):
            os.remove(probe)
        w, h = geometry(gui)
        r = sh(f'ffmpeg -y -loglevel error -f x11grab -draw_mouse 0 -window_id {gui} -video_size {w}x{h} -i {DISPLAY} '
               f'-vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" -frames:v 1 {probe}')
        dark = 1.0
        if os.path.exists(probe):
            from PIL import Image
            im = Image.open(probe).convert('L').resize((64, 36))
            dark = sum(1 for v in im.getdata() if v < 12) / (64 * 36)
            if dark < 0.5:
                break
        print(f'probe: {r.stderr.strip()[:120] or f"{dark:.0%} black"}; raising the window again', flush=True)
        sh(f'docker run --rm -e DISPLAY={DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix:rw xdo-helper bash -c '
           f'"xdotool windowactivate --sync {frame}; xdotool windowraise {gui}"')
        time.sleep(1.5)
    else:
        sys.exit('the Gazebo window stays obscured or black; clear the desktop and rerun')

    rec = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'x11grab', '-draw_mouse', '0', '-window_id', gui, '-framerate', '30',
                            '-video_size', f'{w}x{h}', '-i', DISPLAY, '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2',
                            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '16',
                            '-pix_fmt', 'yuv420p', f'{a.out}/gz_raw.mp4'], stdin=subprocess.PIPE)
    t0 = time.time()
    events = []
    schedule = list(SHOTS[a.shots])
    while time.time() - t0 < a.seconds:
        now = time.time() - t0
        if schedule and now >= schedule[0][0]:
            _, kind = schedule.pop(0)
            r = shot(kind)
            events.append({'t': round(now, 2), 'shot': kind, 'ok': 'true' in (r.stdout or '')})
            print(events[-1], flush=True)
        time.sleep(0.2)
    rec.stdin.write(b'q')
    rec.stdin.flush()
    rec.wait()
    json.dump({'window': [w, h], 'strategy': a.strategy, 'events': events}, open(f'{a.out}/events.json', 'w'), indent=1)
    sh(f'docker rm -f {CT}')
    print(f'recorded {a.seconds:.0f} s to {a.out}/gz_raw.mp4')


if __name__ == '__main__':
    main()
