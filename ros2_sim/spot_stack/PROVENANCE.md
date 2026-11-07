# spot_stack — vendored Spot + CHAMP packages

Copied unmodified from
`~/Documents/Repository/LandmarkBasedNavigation/tracking-sim/src/spot_stack`
(itself vendored from gazebo-isr-simulator, see that repo's PROVENANCE),
on 2026-10-12.

| Package | Role |
|---|---|
| `champ`, `champ_base`, `champ_msgs` | CHAMP quadruped controller (BSD-3, see LICENSE) |
| `champ_config` | Spot gait / joint / link parameters |
| `spot_description` | Spot model: meshes, `models/spot/model.{sdf,urdf}` |

The Gazebo-side model the simulation spawns is an adapted copy at
`../models/spot_evader/model.sdf`, produced by
`../scripts/adapt_spot_model.py` (model renamed to "evader", sensors
stripped, collision bitmask, Fortress "ignition" plugin names kept); it
still references this package's meshes via package:// URIs.

## RoadmapTracking copy

Copied unmodified again, on 2026-11-01, from
`~/Documents/Repository/CodeICRA25/icra25_ros2/spot_stack` (which carries
the chain above). Nothing in this directory is edited here either: the
Gazebo-side evader model this package spawns is the adapted copy at
`../models/spot_evader/model.sdf`, written by
`../scripts/adapt_spot_model.py` (model renamed `spot_evader`, topics and
odom frame matched to this package's bridge, sensors stripped, collision
bitmask added, Fortress "ignition" plugin names kept).
