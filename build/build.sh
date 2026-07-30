#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"
mojo build --emit shared-lib -O3 \
  "$repo_dir/src/noise.mojo" \
  -o "$repo_dir/dist/libmojo-perlin-noise.so"
