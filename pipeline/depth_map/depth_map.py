import torch
import cv2
import numpy as np
import os
from pathlib import Path


def generate_depth_map(
    wrapped_image_path,
    infilled_image_path=None,
    output_dir=None,
    wrap_ratio=0.5,
    target_w=8192,
    target_h=4096,
    depth_scale=5.0,
    min_depth=0.15
):
    """
    Generates wrapped & unwrapped depth maps using MiDaS.

    Parameters
    ----------
    wrapped_image_path : str | Path
        Path to horizontally-extended (wrapped) panorama image
    infilled_image_path : str | Path | None
        Path to original (infilled) panorama image (optional, only used for size reference)
    output_dir : str | Path | None
        Output directory (defaults to pipeline/common files/Final Versions)

    Returns
    -------
    dict
        {
            "depth_wrapped": Path,
            "depth_unwrapped": Path
        }
    """

    # ===============================
    # CONFIG
    # ===============================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    wrapped_image_path = Path(wrapped_image_path)
    if not wrapped_image_path.exists():
        raise FileNotFoundError(f"Wrapped image not found: {wrapped_image_path}")

    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parents[1]
            / "common files"
            / "Final Versions"
        )
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ===============================
    # 1️⃣ Load MiDaS
    # ===============================
    print(f"🚀 Initializing MiDaS on {DEVICE}...")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    midas.to(DEVICE).eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform

    # ===============================
    # 2️⃣ Load & Pre-process Image
    # ===============================
    img_wrapped = cv2.imread(str(wrapped_image_path))
    img_wrapped = cv2.cvtColor(img_wrapped, cv2.COLOR_BGR2RGB)

    img_wrapped = cv2.resize(
        img_wrapped,
        (int(target_w * (1 + 2 * wrap_ratio)), target_h),
        interpolation=cv2.INTER_AREA
    )

    H, W_wrapped, _ = img_wrapped.shape
    pad = int((W_wrapped / (1 + 2 * wrap_ratio)) * wrap_ratio)
    W_original = W_wrapped - 2 * pad

    # ===============================
    # 3️⃣ Depth Prediction
    # ===============================
    input_batch = transform(img_wrapped).to(DEVICE)

    with torch.no_grad():
        depth_wrapped = midas(input_batch)

    depth_wrapped = depth_wrapped.squeeze().cpu().numpy().astype(np.float32)

    # ===============================
    # 4️⃣ Advanced Geometry Mapping
    # ===============================
    depth_wrapped = depth_wrapped.max() - depth_wrapped

    p_low, p_high = np.percentile(depth_wrapped, (1, 99))
    depth_wrapped = np.clip(depth_wrapped, p_low, p_high)
    depth_wrapped -= depth_wrapped.min()
    depth_wrapped /= (depth_wrapped.max() + 1e-8)

    NEAR_LIFT = 0.25
    depth_wrapped = depth_wrapped * (1.0 - NEAR_LIFT) + NEAR_LIFT

    GAMMA = 1.6
    depth_wrapped = np.power(depth_wrapped, GAMMA)

    depth_wrapped = np.maximum(depth_wrapped, min_depth)
    depth_wrapped *= depth_scale

    depth_wrapped = cv2.resize(
        depth_wrapped,
        (W_wrapped, target_h),
        interpolation=cv2.INTER_CUBIC
    )

    # ===============================
    # 5️⃣ Unwrap
    # ===============================
    depth_unwrapped = depth_wrapped[:, pad:pad + W_original]

    # ===============================
    # 6️⃣ Save outputs
    # ===============================
    def save_refined_vis(depth_data, path):
        # Normalize for 8-bit save
        vis = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)

        # 🔁 INVERT (black <-> white)
        vis = 255 - vis

        # Apply CLAHE to help see the mountain detail in the preview
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        vis = clahe.apply(vis.astype(np.uint8))

        cv2.imwrite(str(path), vis)


    depth_wrapped_path = output_dir / "depth_wrapped_final.png"
    depth_unwrapped_path = output_dir / "depth_unwrapped_final.png"

    # save_refined_vis(depth_wrapped, depth_wrapped_path)
    save_refined_vis(depth_unwrapped, depth_unwrapped_path)

    print("✅ Depth map generation completed.")
    print(f"Saved to: {depth_unwrapped_path}")

    return {
        "depth_wrapped": depth_wrapped_path,
        "depth_unwrapped": depth_unwrapped_path
    }

# generate_depth_map(
#     wrapped_image_path=r"F:\TextTo360\pipeline\common files\Edited Images\upscaled_final_extended.png",
#     infilled_image_path=r"F:\TextTo360\pipeline\common files\Final Versions\upscaled_final.png",
#     output_dir=r"F:\TextTo360\pipeline\common files\Final Versions"
# )
