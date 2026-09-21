#!/usr/bin/env bash
# ==============================================================================
# PAVEN - Automated VTM 18.2 Builder & Patcher
# Clones official JVET reference software (VTM-18.2), applies the PAVEN Delta-QP
# patch, and compiles the EncoderApp binary using CMake.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${1:-$SCRIPT_DIR/build_vtm}"
PATCH_FILE="$SCRIPT_DIR/vtm_18_2_paven.patch"

echo "=================================================================="
echo " Building PAVEN-Modified VTM 18.2 (JVET Reference Software)"
echo "=================================================================="
echo "[*] Working directory: $WORK_DIR"
echo "[*] Patch file:        $PATCH_FILE"

if [ ! -f "$PATCH_FILE" ]; then
    echo "[-] Error: Patch file not found at $PATCH_FILE"
    exit 1
fi

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ ! -d "VVCSoftware_VTM" ]; then
    echo "[*] Cloning official VTM 18.2 from Fraunhofer HHI..."
    git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git -b VTM-18.2 VVCSoftware_VTM
else
    echo "[*] Existing VVCSoftware_VTM directory found. Updating..."
fi

cd VVCSoftware_VTM

echo "[*] Applying PAVEN Delta-QP patch..."
git checkout -f VTM-18.2
git apply "$PATCH_FILE"

echo "[*] Compiling VTM with CMake (Release mode)..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

ENCODER_BIN=""
if [ -f "bin/EncoderAppStatic" ]; then
    ENCODER_BIN="$PWD/bin/EncoderAppStatic"
elif [ -f "bin/EncoderApp" ]; then
    ENCODER_BIN="$PWD/bin/EncoderApp"
fi

echo "=================================================================="
if [ -n "$ENCODER_BIN" ]; then
    echo "[+] SUCCESS: PAVEN VTM Encoder binary built successfully!"
    echo "[+] Path: $ENCODER_BIN"
    echo ""
    echo "Usage with PAVEN Delta-QP matrix:"
    echo "  $ENCODER_BIN -c ../cfg/encoder_randomaccess_vtm.cfg \\"
    echo "               -c ../cfg/per-sequence/BQMall.cfg \\"
    echo "               -i BQMall_832x480_60.yuv \\"
    echo "               --QP=27 \\"
    echo "               --QPFilePath=/path/to/paven_matrix.qp \\"
    echo "               -b output.bin"
else
    echo "[-] Warning: Encoder binary not found in expected output directories."
fi
echo "=================================================================="
