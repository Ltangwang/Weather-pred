#!/usr/bin/env bash
# Download WeatherBench v1 ERA5 preprocessing: 2m_temperature @ 5.625deg (~2.2GB zip).
#
# WHY rsync:
# Plain HTTPS wget against dataserv often returns HTTP 404/401 or 0-byte files on some
# networks. TUM mediaTUM publishes a public FTP/rsync credential for this mirror:
#   https://mediatum.ub.tum.de/1524895
#
# AFTER:
#   <OPENSTL_DATA_ROOT>/weather_5_625deg/2m_temperature/*.nc
#
# Usage:
#   bash scripts/download_weatherbench_t2m_5625deg.sh [OPENSTL_DATA_ROOT]
set -euo pipefail

OPENSTL_DATA_ROOT="${1:-/root/autodl-tmp/OpenSTL/data}"
DST="${OPENSTL_DATA_ROOT}/weather_5_625deg/2m_temperature"
ZIP_NAME="2m_temperature_5.625deg.zip"
RSYNC_REMOTE="rsync://m1524895@dataserv.ub.tum.de/m1524895/5.625deg/2m_temperature/${ZIP_NAME}"

export RSYNC_PASSWORD="${RSYNC_PASSWORD:-m1524895}"

mkdir -p "${DST}"
cd "${DST}"
echo "Fetching ${ZIP_NAME} via rsync (public WeatherBench mirror)..."
rsync -av --progress "${RSYNC_REMOTE}" .
echo "Unzipping..."
unzip -o -q "${ZIP_NAME}"
rm -f "${ZIP_NAME}"
echo "Done. NetCDF files in: ${DST}"
ls -1 "${DST}"/*.nc | head -3
echo "... ($(ls -1 "${DST}"/*.nc | wc -l) files)"
