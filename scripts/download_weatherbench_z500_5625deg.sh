#!/usr/bin/env bash
# Download WeatherBench v1 geopotential @ 500 hPa, 5.625deg (~2 GB zip).
#
# AFTER:
#   <OPENSTL_DATA_ROOT>/weather_5_625deg/geopotential_500/*.nc
#   <OPENSTL_DATA_ROOT>/weather_5_625deg/geopotential -> geopotential_500  (symlink for OpenSTL)
#
# Usage:
#   bash scripts/download_weatherbench_z500_5625deg.sh [OPENSTL_DATA_ROOT]
set -euo pipefail

OPENSTL_DATA_ROOT="${1:-/root/autodl-tmp/OpenSTL/data}"
BASE="${OPENSTL_DATA_ROOT}/weather_5_625deg"
DST="${BASE}/geopotential_500"
ZIP_NAME="geopotential_500_5.625deg.zip"
RSYNC_REMOTE="rsync://m1524895@dataserv.ub.tum.de/m1524895/5.625deg/geopotential_500/${ZIP_NAME}"

export RSYNC_PASSWORD="${RSYNC_PASSWORD:-m1524895}"

mkdir -p "${DST}"
cd "${DST}"
echo "Fetching ${ZIP_NAME} via rsync (public WeatherBench mirror)..."
rsync -av --progress "${RSYNC_REMOTE}" .
echo "Unzipping..."
unzip -o -q "${ZIP_NAME}"
rm -f "${ZIP_NAME}"

# OpenSTL loader expects `<weather_root>/geopotential/geopotential*.nc`.
ln -sfn geopotential_500 "${BASE}/geopotential"

echo "Done. NetCDF files in: ${DST}"
ls -1 "${DST}"/*.nc | head -3
echo "... ($(ls -1 "${DST}"/*.nc | wc -l) files)"
echo "Symlink: ${BASE}/geopotential -> geopotential_500"
