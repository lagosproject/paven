#!/usr/bin/env python3
"""
PAVEN: Quickstart Inference & Delta-QP Grid Generation Example
Paper: "PAVEN: A Perceptual Algorithm for Versatile video Encoding using Neural networks"
       (Elsevier EAAI, 2025, DOI: 10.1016/j.engappai.2025.111664)
"""

import os
import torch
import numpy as np
from paven import PavenModel, QpGrid

def main():
    print("==================================================================")
    print(" PAVEN: Perceptual Saliency & Delta-QP Grid Quickstart Demo")
    print("==================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")

    # 1. Instantiate the Model
    local_candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../huggingface_paven/paven_vinet_yyy.pt")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../weights/paven_vinet_yyy.pt")),
        "paven_vinet_yyy.pt"
    ]
    local_weights = next((p for p in local_candidates if os.path.isfile(p)), None)

    if local_weights:
        print(f"[*] Found local model weights: {local_weights}")
        model = PavenModel.from_file(local_weights, device=device)
    else:
        print("[*] Downloading model weights directly from Hugging Face Hub (lagosproject/paven)...")
        model = PavenModel.from_pretrained(repo_id="lagosproject/paven", device=device)

    model.eval()

    # 2. Simulate input temporal clip (32 frames of 1080p video, resized to 224x384 in YYY format)
    # Shape: (Batch=1, Channels=3 [YYY], Time=32, Height=224, Width=384)
    print("[*] Creating synthetic 32-frame video clip in YYY format...")
    clip_tensor = torch.randn(1, 3, 32, 224, 384, dtype=torch.float32, device=device)

    # 3. Predict Spatio-temporal Saliency Map
    with torch.no_grad():
        saliency_map = model(clip_tensor) # Shape: (1, 224, 384)
    
    saliency_np = saliency_map.squeeze().cpu().numpy()
    print(f"[*] Predicted Saliency Map shape: {saliency_np.shape}, Range: [{saliency_np.min():.3f}, {saliency_np.max():.3f}]")

    # 4. Generate CTU-level (128x128) Delta-QP matrix for a 1080p frame (1920x1080)
    width, height = 1920, 1080
    qp_gen = QpGrid(img_size=(width, height), frame_count=1)
    delta_qp_matrix = qp_gen.compute_frame_grid(saliency_np)

    print(f"[*] Generated CTU Delta-QP Matrix ({delta_qp_matrix.shape[0]} rows x {delta_qp_matrix.shape[1]} cols):")
    print(delta_qp_matrix)

    # Save to demo .qp file
    qp_gen.process_image(saliency_np, frame_idx=0)
    out_qp = "sample_output.qp"
    qp_gen.save_qp_to_file(out_qp)
    print(f"[*] Delta-QP matrix saved to: {out_qp}")
    print("==================================================================")
    print(" Demo executed successfully!")
    print("==================================================================")

if __name__ == "__main__":
    main()
