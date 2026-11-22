#!/usr/bin/env bash
# Build the native core. Run inside the roadmaptracking image (it has g++):
#   docker run --rm -v "$PWD":/app -w /app roadmaptracking bash native/build.sh
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -shared -fPIC -o libkercore.so kercore.cpp
echo "built $(pwd)/libkercore.so"
