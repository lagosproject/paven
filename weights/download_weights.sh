#!/usr/bin/env bash
# ==============================================================================
# PAVEN: Pre-trained Weights Downloader
# Fetches official weights from Hugging Face Hub (lagosproject/paven)
# ==============================================================================

set -e

DEST_DIR="${1:-./weights}"
mkdir -p "${DEST_DIR}"

HF_URL="https://huggingface.co/lagosproject/paven/resolve/main/paven_vinet_yyy.pt"
OUTPUT_FILE="${DEST_DIR}/paven_vinet_yyy.pt"

echo "[PAVEN] Downloading pre-trained weights from Hugging Face Hub..."
echo "Source: ${HF_URL}"
echo "Target: ${OUTPUT_FILE}"

if command -v curl &> /dev/null; then
    curl -L "${HF_URL}" -o "${OUTPUT_FILE}"
elif command -v wget &> /dev/null; then
    wget -O "${OUTPUT_FILE}" "${HF_URL}"
else
    echo "[PAVEN] Error: Neither curl nor wget was found on your system."
    exit 1
fi

echo "[PAVEN] Download complete."
