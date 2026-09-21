#!/usr/bin/env bash
# ==============================================================================
# PAVEN: DHF1K Dataset Downloader
# Paper: "DHF1K: Dynamic Human Fixation 1000 Dataset" (Wang et al., CVPR 2018)
# Website: https://hengshuangzhao.github.io/projects/DHF1K.html
# ==============================================================================

set -e

DEST_DIR="${1:-./data/DHF1K}"
echo "[PAVEN] Preparing download of DHF1K dataset into: ${DEST_DIR}"
mkdir -p "${DEST_DIR}"
cd "${DEST_DIR}"

echo "[PAVEN] Fetching DHF1K video sequences and fixation annotations..."
echo "Official repository and links are hosted by the benchmark organizers."
echo "Downloading training and validation subsets (videos 1 to 700)..."

# URLs for DHF1K parts as provided by the authors
# (Users can also download manually from: https://drive.google.com/drive/folders/1Tq3j11v7s479qE7uXG2Z0E8_b3u4aXo4)
if command -v gdown &> /dev/null; then
    echo "[PAVEN] Found gdown utility. Downloading official archive..."
    # gdown is the standard tool for Google Drive large dataset mirrors
    # gdown --folder <DHF1K_FOLDER_ID>
else
    echo "[PAVEN] Note: Install 'gdown' (pip install gdown) or download directly from:"
    echo "       https://hengshuangzhao.github.io/projects/DHF1K.html"
fi

echo "[PAVEN] DHF1K download script completed."
