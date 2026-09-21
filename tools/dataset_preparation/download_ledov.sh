#!/usr/bin/env bash
# ==============================================================================
# PAVEN: LEDOV Dataset Downloader
# Paper: "LEDOV: A Large-Scale Database for Eye-Tracking-While-Detecting Over Videos"
#        (Lai et al., IEEE TMM 2019)
# Repository: https://github.com/remega/LEDOV
# ==============================================================================

set -e

DEST_DIR="${1:-./data/LEDOV}"
echo "[PAVEN] Preparing download of LEDOV dataset into: ${DEST_DIR}"
mkdir -p "${DEST_DIR}"
cd "${DEST_DIR}"

echo "[PAVEN] Please consult the official LEDOV repository for dataset access and licenses:"
echo "       https://github.com/remega/LEDOV"
echo "Dataset contains 538 annotated videos across humans, animals, and objects."

if command -v git &> /dev/null; then
    echo "[PAVEN] Cloning repository metadata..."
    if [ ! -d "LEDOV_repo" ]; then
        git clone https://github.com/remega/LEDOV.git LEDOV_repo
    fi
fi

echo "[PAVEN] LEDOV downloader script complete."
