import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'ros2_sim'


def model_data_files(models_root='models'):
    """Recurse into models/ so nested asset dirs (meshes/, materials/
    textures/, from Fuel-downloaded models) install with their structure
    intact — a flat glob() only catches one level."""
    entries = []
    for dirpath, _dirnames, filenames in os.walk(models_root):
        if not filenames:
            continue
        dest = os.path.join('share', package_name, dirpath)
        srcs = [os.path.join(dirpath, f) for f in filenames]
        entries.append((dest, srcs))
    return entries


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         [os.path.join('resource', package_name)]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        *model_data_files(),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Shashwata Mandal',
    maintainer_email='smandal@iastate.edu',
    description='Case-study Gazebo/RViz2 visualization of the KER pursuit-evasion simulation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'pursuit_controller = ros2_sim.pursuit_controller:main',
            'evader_wanderer = ros2_sim.evader_wanderer:main',
        ],
    },
)
