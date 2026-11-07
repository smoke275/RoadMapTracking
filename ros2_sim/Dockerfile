# Verified against the actual pulled image (2026-08): Gazebo (Ignition
# Fortress / Gazebo 6), RViz2, and ros_gz (ros-humble-ros-gz-bridge,
# ros-humble-ros-gz-sim, etc.) are ALL already present in desktop-full —
# confirmed via `dpkg -l` inside the pulled image, not assumed. Only
# shapely (for scripts/generate_world.py, matching the main repo's
# geometry tooling) needs adding.
FROM osrf/ros:humble-desktop-full

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    build-essential \
    swig \
    && rm -rf /var/lib/apt/lists/*

# shapely for scripts/generate_world.py, the rest for running the actual
# KER / min-max-alpha planner (ker_pipeline.py and friends) inside this
# image as a ROS 2 node, with the main repo bind-mounted at /app and
# PYTHONPATH=/app — same dependency set as the main roadmaptracking image
# (visilibity needs build-essential + swig to compile; PyQt5 is imported
# by ker_pipeline at module level but never opens a display here).
# numpy<2 keeps ROS Humble's dist-packages ecosystem happy; scipy is
# force-upgraded past Ubuntu's 1.8 because the guard-set ILP needs
# scipy.optimize.milp (scipy >= 1.9).
RUN pip3 install --no-cache-dir --upgrade \
    'numpy<2' 'scipy>=1.10' shapely pyclipper pyvisgraph tqdm visilibity \
    PyQt5 matplotlib seaborn bidict pygame

ENV ROS_DOMAIN_ID=42

WORKDIR /ros2_ws
RUN mkdir -p /ros2_ws/src

COPY . /ros2_ws/src/ros2_sim

# colcon does not descend into a package dir, so the vendored Spot/CHAMP
# packages (spot_stack/, used by evader_type:=spot) must live beside
# ros2_sim, not inside it. Same arrangement as CodeICRA25/icra25_ros2.
RUN mv /ros2_ws/src/ros2_sim/spot_stack /ros2_ws/src/spot_stack

SHELL ["/bin/bash", "-c"]
# NOT --symlink-install: for ament_python packages that breaks `ros2
# launch`'s Node action, which expects console_scripts under
# install/<pkg>/lib/<pkg>/ — --symlink-install (a pip -e install under the
# hood) puts them in install/<pkg>/bin/ instead. Confirmed by actually
# running the launch file and hitting exactly this failure.
# Builds champ, champ_msgs, champ_base (C++), champ_config, spot_description
# and ros2_sim; all of CHAMP's ROS dependencies are already in desktop-full.
RUN source /opt/ros/humble/setup.bash && \
    colcon build

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

CMD ["bash"]
