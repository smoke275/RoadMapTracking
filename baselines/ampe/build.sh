#!/usr/bin/env bash
# Build the minimax baseline. Run inside the roadmaptracking image or on a host with g++ and OpenMP:
#   bash baselines/ampe/build.sh
# The authors' solver (orig/Minmax.hpp) is not edited. This script writes Minmax_patched.hpp,
# which differs from it in three lines: a hook for pursuer neighbors given from outside
# (roadmap mode), its declaration, and a quieter progress print.
set -e
cd "$(dirname "$0")"
tr -d '\r' < orig/Minmax.hpp | sed \
  -e 's|    neighbors(pose_p, ngbs_p, Rprm_p);|    if (g_ext_ngbs_p) ngbs_p = g_ext_ngbs_p(Rprm_p); else neighbors(pose_p, ngbs_p, Rprm_p);|' \
  -e '0,/^class MinMaxGame/s//std::function<vector<vector<pair<int,double>>>(double)> g_ext_ngbs_p;\nclass MinMaxGame/' \
  -e 's|        cout << "Step : " << cont << ", queue size : " << state_queue.size() << endl;|        if (cont % 20 == 1) cerr << "Step : " << cont << ", queue size : " << state_queue.size() << endl;|' \
  > Minmax_patched.hpp
cp orig/utils.hpp orig/quasirand.hpp .
g++ -O2 -fopenmp -std=c++17 -o ampe_poly ampe_poly.cpp
echo "built $(pwd)/ampe_poly"
