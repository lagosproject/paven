"""
PAVEN: Bitrate Reduction and Statistical Analysis Tool
Paper: "PAVEN: A Perceptual Algorithm for Versatile video Encoding using Neural networks"
       (Elsevier EAAI, 2025, DOI: 10.1016/j.engappai.2025.111664)
Author: Pablo Fernández Lagos (UPM Master Thesis, Chapter 6, 2025)

Computes bitrate differences between standard baseline VVC encoding and PAVEN perceptual encoding,
derives statistics (mean, median, variance, std, min, max), and exports publication-ready LaTeX tables.
"""

import os
import io
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


def compute_bitrate_from_bin(bin_path: str, duration_sec: float) -> float:
    """Calculate bitrate in kbps from encoded .bin bitstream file size and duration."""
    size_bytes = os.path.getsize(bin_path)
    bitrate_kbps = (size_bytes * 8.0) / (duration_sec * 1000.0)
    return bitrate_kbps


def compute_reduction_statistics(df: pd.DataFrame, col_name: str = "Reduction Percentage (%)") -> Dict[str, float]:
    """Calculate descriptive statistics on the reduction percentage column."""
    values = df[col_name].values
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "variance": float(np.var(values, ddof=1)),
        "std_dev": float(np.std(values, ddof=1)),
        "min_val": float(np.min(values)),
        "max_val": float(np.max(values)),
    }


def generate_latex_table(stats: Dict[str, float], qp_value: int = 27) -> str:
    """Generate LaTeX table code matching the paper's formatting."""
    latex = f"""\\begin{{table}}[t]
    \\caption{{Bitrate reduction statistics (%) for base QP {qp_value}.}}
    \\label{{tab:bitrate_stats_qp{qp_value}}}
    \\centering
    \\begin{{tabular}}{{|l|r|}}
        \\hline
        \\textbf{{Metric}} & \\textbf{{Value (\\%)}} \\\\
        \\hline
        Mean & {stats['mean']:.2f} \\\\
        Median & {stats['median']:.2f} \\\\
        Variance & {stats['variance']:.2f} \\\\
        Standard Deviation & {stats['std_dev']:.2f} \\\\
        Minimum & {stats['min_val']:.2f} \\\\
        Maximum & {stats['max_val']:.2f} \\\\
        \\hline
    \\end{{tabular}}
\\end{{table}}
"""
    return latex


# Reference sample data from experimental evaluation (TFM Table 6.1 and EAAI paper)
DEFAULT_QP27_DATA = """File,Bitrate Baseline (kbps),Bitrate PAVEN (kbps),Reduction Percentage (%)
BasketballDrill_832x480_50.yuv,353.83,336.93,4.78
BasketballDrive_1920x1080_50.yuv,1089.53,1049.39,3.68
BasketballPass_416x240_50.yuv,151.62,147.73,2.57
BlowingBubbles_416x240_50.yuv,156.81,141.24,9.93
BQMall_832x480_60.yuv,378.36,350.55,7.35
BQSquare_416x240_60.yuv,149.82,129.07,13.85
BQTerrace_1920x1080_60.yuv,1002.51,840.54,16.16
Cactus_1920x1080_50.yuv,1012.71,941.83,7.00
Campfire_3840x2160_30fps.yuv,3494.37,3432.30,1.78
CatRobot_3840x2160_60fps.yuv,2477.57,2341.21,5.50
DaylightRoad2_3840x2160_60fps.yuv,2375.53,2255.06,5.07
Drums_3840x2160_100fps.yuv,5352.57,4954.66,7.43
Kimono1_1920x1080_24.yuv,395.63,361.72,8.57
ParkScene_1920x1080_24.yuv,505.25,439.39,13.04
PartyScene_832x480_50.yuv,653.85,552.32,15.53
RaceHorses_416x240_30.yuv,113.50,109.45,3.57
RaceHorses_832x480_30.yuv,348.86,328.24,5.91
RollerCoaster2_3840x2160_60fps.yuv,1925.07,1862.75,3.24
Tango2_3840x2160_60fps.yuv,2016.24,1955.36,3.02
TrafficFlow_3840x2160_30fps.yuv,675.08,620.03,8.15
"""


def main():
    parser = argparse.ArgumentParser(description="PAVEN: Bitrate Reduction Analysis & LaTeX Table Generator")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file with columns: File, Bitrate Baseline, Bitrate PAVEN, Reduction Percentage (%)")
    parser.add_argument("--qp", type=int, default=27, help="Base QP value evaluated (e.g. 27, 37)")
    parser.add_argument("--out-tex", type=str, default=None, help="Output path for LaTeX table (.tex)")
    args = parser.parse_args()

    if args.csv and os.path.isfile(args.csv):
        df = pd.read_csv(args.csv)
    else:
        print("[PAVEN] Using default experimental test set data (QP 27, 20 standard sequences)...")
        df = pd.read_csv(io.StringIO(DEFAULT_QP27_DATA))

    stats = compute_reduction_statistics(df)
    print("\n" + "="*50)
    print(f" PAVEN Bitrate Reduction Statistics (Base QP {args.qp})")
    print("="*50)
    for k, v in stats.items():
        print(f"  {k:20s}: {v:.2f}%")
    print("="*50 + "\n")

    latex_code = generate_latex_table(stats, qp_value=args.qp)
    print(latex_code)

    if args.out_tex:
        with open(args.out_tex, "w") as f:
            f.write(latex_code)
        print(f"[PAVEN] LaTeX table written to: {args.out_tex}")


if __name__ == "__main__":
    main()
