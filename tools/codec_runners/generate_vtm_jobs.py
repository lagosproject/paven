"""
Batch Encoding Job Generator for VTM / VVC
Generates Slurm/Bash jobs to encode test sequences with PAVEN Delta-QP matrices.
Note: This script generates runner scripts only; it does not contain or distribute VTM binaries.
"""

import os
import argparse
from pathlib import Path

DEFAULT_SEQUENCES = [
    "BasketballDrill_832x480_50.yuv",
    "BasketballDrive_1920x1080_50.yuv",
    "BasketballPass_416x240_50.yuv",
    "BlowingBubbles_416x240_50.yuv",
    "BQMall_832x480_60.yuv",
    "BQSquare_416x240_60.yuv",
    "BQTerrace_1920x1080_60.yuv",
    "Cactus_1920x1080_50.yuv",
    "Campfire_3840x2160_30fps_10bit_420_bt709_videoRange.yuv",
    "CatRobot_3840x2160_60fps_10bit_420_jvet.yuv",
    "DaylightRoad2_3840x2160_60fps_10bit_420.yuv",
    "Drums_3840x2160_100fps_10bit_420_jvet.yuv",
    "Kimono1_1920x1080_24.yuv",
    "ParkScene_1920x1080_24.yuv",
    "PartyScene_832x480_50.yuv",
    "RaceHorses_416x240_30.yuv",
    "RaceHorses_832x480_30.yuv",
    "RollerCoaster2_3840x2160_60fps_10bit_420.yuv",
    "Tango2_3840x2160_60fps_10bit_420.yuv",
    "ToddlerFountain2_3840x2160_60fps_10bit_420.yuv",
    "TrafficFlow_3840x2160_30fps_10bit_420_jvet.yuv",
]

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={video_name}_qp{base_qp}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=18:00:00
#SBATCH --output=slurm_{video_name}_qp{base_qp}.out

VTM_BIN="{vtm_bin}"
CFG_MAIN="{vtm_cfg_dir}/encoder_randomaccess_vtm.cfg"
CFG_SEQ="{seq_cfg_dir}/{config_name}.cfg"
INPUT_YUV="{input_yuv_dir}/{video_name}"
QP_FILE="{qp_dir}/{video_name}.qp"
OUTPUT_BIN="{output_dir}/{video_name}_qp{base_qp}.bin"
RECON_YUV="{output_dir}/recon/{video_name}_qp{base_qp}_recon.yuv"

mkdir -p "{output_dir}/recon"

$VTM_BIN \\
    -c "$CFG_MAIN" \\
    -c "$CFG_SEQ" \\
    -i "$INPUT_YUV" \\
    --QP={base_qp} \\
    --IntraPeriod=32 \\
    --QPFilePath="$QP_FILE" \\
    --BitstreamFile="$OUTPUT_BIN" \\
    --ReconFile="$RECON_YUV"
"""


def generate_jobs(args):
    out_dir = Path(args.output_scripts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for vid in DEFAULT_SEQUENCES:
        cfg_name = vid.split("_")[0]
        script_content = SLURM_TEMPLATE.format(
            video_name=vid,
            config_name=cfg_name,
            base_qp=args.base_qp,
            vtm_bin=args.vtm_bin,
            vtm_cfg_dir=args.vtm_cfg_dir,
            seq_cfg_dir=args.seq_cfg_dir,
            input_yuv_dir=args.input_yuv_dir,
            qp_dir=args.qp_dir,
            output_dir=args.output_dir,
        )

        script_path = out_dir / f"encode_{vid}_qp{args.base_qp}.sh"
        with open(script_path, "w") as f:
            f.write(script_content)

    print(f"Generated {len(DEFAULT_SEQUENCES)} SLURM encoding scripts in: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate VTM batch encoding scripts.")
    parser.add_argument("--base-qp", type=int, default=27, choices=[22, 27, 32, 37], help="Base quantization parameter.")
    parser.add_argument("--vtm-bin", type=str, default="/path/to/EncoderAppStatic", help="Path to VTM EncoderApp binary.")
    parser.add_argument("--vtm-cfg-dir", type=str, default="/path/to/vtm/cfg", help="Path to VTM main config folder.")
    parser.add_argument("--seq-cfg-dir", type=str, default="configs/sequences", help="Path to per-sequence configs.")
    parser.add_argument("--input-yuv-dir", type=str, default="/path/to/YUVvideos", help="Path to raw YUV test videos.")
    parser.add_argument("--qp-dir", type=str, default="output/qp_matrices", help="Path to generated PAVEN .qp files.")
    parser.add_argument("--output-dir", type=str, default="output/encoded_bits", help="Path to save output bitstreams.")
    parser.add_argument("--output-scripts-dir", type=str, default="scripts/vtm_batches", help="Where to write .sh jobs.")
    args = parser.parse_args()

    generate_jobs(args)


if __name__ == "__main__":
    main()
