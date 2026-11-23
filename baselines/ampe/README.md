# Minimax baseline (Antonio, Becerra, Murrieta-Cid)

Solver of "Approximate Methods for Visibility-Based Pursuit-Evasion",
IEEE Transactions on Robotics, 2024, doi 10.1109/TRO.2024.3474948.
The files in `orig/` are the authors' code, unchanged, used with their
permission as a benchmark.

What is ours:

- `Environment.hpp`: same interface as the authors' file, but the free space is
  the interior of one simple polygon, and two points see each other if their
  segment stays in the closed polygon.
- `build.sh`: writes `Minmax_patched.hpp` from `orig/Minmax.hpp`. Three lines
  differ: a hook that takes the pursuer neighbors from outside, its
  declaration, and a quieter progress print.
- `ampe_poly.cpp`: driver. Mode `free` samples the pursuer anywhere in the
  polygon, as in their paper. Mode `roadmap` puts the pursuer samples on our
  patrol roadmap and connects them by distance along the roadmap. The roadmap
  mode is our adaptation and is not part of their paper.

Pipeline: `ampe_solve.py` (offline tables in `resources/ampe/`), then the
strategies `ampe-roadmap` and `ampe-free` of `run_benchmark.py`
(`benchmark/ampe_pursuer.py`).

The solver is defined for the game in which the pursuer wins if it never loses
sight of the evader. At speed ratios where the pursuer loses from every state
(for example 0.8 on poly9), the value function gives little guidance and the
baseline tracks poorly; at ratio 1.5 on poly9 the free pursuer keeps the evader
in sight in 100% of the frames.
