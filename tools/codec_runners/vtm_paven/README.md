# VTM 18.2 Integration for PAVEN

This directory contains the modifications made to the **Versatile Video Coding (VVC) Test Model 18.2 (VTM-18.2)** developed by the Joint Video Experts Team (JVET) of ITU-T and ISO/IEC.

---

## 💡 Why a Patch instead of Full VTM Source?

Official VTM is a massive C++ codebase (~150 MB, 500,000+ lines). To maintain academic hygiene, avoid repository bloat, and respect upstream Fraunhofer HHI licensing:
1. We provide the **exact 399-line patch** ([`vtm_18_2_paven.patch`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/codec_runners/vtm_paven/vtm_18_2_paven.patch)) that modifies only 9 files in VTM 18.2.
2. We provide an automated 1-click script ([`build_vtm_paven.sh`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/codec_runners/vtm_paven/build_vtm_paven.sh)) that clones clean VTM 18.2 from the official JVET repository, applies the patch, and builds the encoder binary.

---

## 🛠️ Modified VTM Files Overview

| File | Purpose of Modification |
|---|---|
| `source/App/EncoderApp/EncAppCfg.h` | Declares `--QPFilePath` CLI string argument and `m_qp_matrix_values` buffer pointer. |
| `source/App/EncoderApp/EncAppCfg.cpp` | Registers the `--QPFilePath` command line option in the VTM option parser. |
| `source/App/EncoderApp/EncApp.cpp` | Parses the comma-separated `.qp` matrix file, calculates `QP_final = QP_base + Delta_QP` per CTU, and passes the matrix to `EncLib`. |
| `source/Lib/EncoderLib/EncCfg.h` | Adds getter/setter `setQPMatrixValues()` and `getQPMatrixValues()`. |
| `source/Lib/EncoderLib/EncCu.cpp` | Overrides the default CTU quantization parameter with `the_qp_val` during rate-distortion optimization and mode search. |
| `source/Lib/EncoderLib/EncModeCtrl.h` | Extended interface for mode exploration with custom partition control. |
| `source/Lib/EncoderLib/EncModeCtrl.cpp` | Implements conditional mode exploration and partition pruning. |

---

## 🚀 1-Click Compilation Guide

Run the automated script:
```bash
chmod +x tools/codec_runners/vtm_paven/build_vtm_paven.sh
./tools/codec_runners/vtm_paven/build_vtm_paven.sh /path/to/build_directory
```

### Manual Compilation
```bash
# 1. Clone official JVET VTM-18.2
git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git -b VTM-18.2
cd VVCSoftware_VTM

# 2. Apply PAVEN patch
git apply /path/to/vtm_18_2_paven.patch

# 3. Build with CMake
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## 🎥 Running VTM with PAVEN Matrices

```bash
./bin/EncoderAppStatic \
    -c ../cfg/encoder_randomaccess_vtm.cfg \
    -c ../cfg/per-sequence/BQMall.cfg \
    -i /path/to/BQMall_832x480_60.yuv \
    --QP=27 \
    --QPFilePath=/path/to/BQMall_832x480_60.qp \
    -b /path/to/output_paven_qp27.bin \
    -o /path/to/recon_paven_qp27.yuv
```
