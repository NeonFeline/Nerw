#!/usr/bin/env bash
# Mesh, solve, and turn the displacement seismograms into a DAS strain gather.
#
#   SPECFEM2D_BIN=/path/to/specfem2d/bin ./run.sh [das_strain.py options]
#
# Set PYTHON="uv run python" to use the project environment.
set -euo pipefail
cd "$(dirname "$0")"
: "${SPECFEM2D_BIN:?set SPECFEM2D_BIN to the bin/ directory of a SPECFEM2D build}"
PYTHON=${PYTHON:-python3}

$PYTHON make_stations.py
rm -rf OUTPUT_FILES
mkdir -p OUTPUT_FILES
"$SPECFEM2D_BIN/xmeshfem2D" | tee OUTPUT_FILES/output_mesher.txt
"$SPECFEM2D_BIN/xspecfem2D" | tee OUTPUT_FILES/output_solver.txt
$PYTHON das_strain.py "$@"
