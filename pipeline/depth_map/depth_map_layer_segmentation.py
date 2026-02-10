
import torch
import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans


def generate_depth_layers(
    wrapped_image_path,
    infilled_image_path,
    wrap_ratio=0.5,
    target_w=1024,
    target_h=512,
    max_clusters=8
):
    """
    Generates depth-based cut layers and masks from a wrapped panorama.

    Saves:
    - Cut RGBA layers -> pipeline/common files/Edited Images
    - Binary masks    -> pipeline/common files/Final Versions

    Returns
    -------
    dict
        {
            "layers": [Path, ...],
            "masks": [Path, ...]
        }
    """

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    wrapped_image_path = Path(wrapped_image_path)
    infilled_image_path = Path(infilled_image_path)

    if not wrapped_image_path.exists():
        raise FileNotFoundError(f"Wrapped image not found: {wrapped_image_path}")
    if not infilled_image_path.exists():
        raise FileNotFoundError(f"Infilled image not found: {infilled_image_path}")

    # ===============================
    # Output directories (FIXED)
    # ===============================
    PIPELINE_DIR = Path(__file__).resolve().parents[1]

    CUT_DIR = PIPELINE_DIR / "common files" / "Edited Images"
    MASK_DIR = PIPELINE_DIR / "common files" / "Final Versions"

    CUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    # ===============================
    # Load MiDaS
    # ===============================
    print(f"🚀 Initializing MiDaS on {DEVICE}...")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").to(DEVICE).eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    # ===============================
    # Load images
    # ===============================
    img_wrapped_rgb = cv2.cvtColor(
        cv2.imread(str(wrapped_image_path)),
        cv2.COLOR_BGR2RGB
    )
    img_infilled = cv2.imread(str(infilled_image_path))

    # Prepare wrapped input
    img_work = cv2.resize(
        img_wrapped_rgb,
        (int(target_w * (1 + 2 * wrap_ratio)), target_h)
    )

    H, W_wrapped, _ = img_work.shape
    pad = int((W_wrapped / (1 + 2 * wrap_ratio)) * wrap_ratio)
    W_original = W_wrapped - 2 * pad

    # ===============================
    # Depth inference
    # ===============================
    input_batch = transform(img_work).to(DEVICE)

    with torch.no_grad():
        depth = midas(input_batch).squeeze().cpu().numpy().astype(np.float32)

    # Normalize depth
    depth = depth.max() - depth
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth = cv2.resize(depth, (W_wrapped, target_h), interpolation=cv2.INTER_CUBIC)

    # Unwrap
    depth_unwrapped = depth[:, pad:pad + W_original]
    image = cv2.resize(
        img_infilled,
        (W_original, target_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    # ===============================
    # Depth clustering
    # ===============================
    def compute_depth_clusters(depth_map, k):
        flat = depth_map.reshape(-1, 1)
        flat = flat[np.isfinite(flat).flatten()]

        kmeans = KMeans(
            n_clusters=k,
            n_init="auto",
            random_state=42
        ).fit(flat)

        return np.sort(kmeans.cluster_centers_.flatten())

    def clusters_to_thresholds(centers):
        return [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]

    centers = compute_depth_clusters(depth_unwrapped, max_clusters)
    thresholds = clusters_to_thresholds(centers)

    # ===============================
    # Cut & save layers
    # ===============================
    depth_u8 = (depth_unwrapped * 255).astype(np.uint8)
    bounds = [0] + [int(t * 255) for t in thresholds] + [255]

    saved_layers = []
    saved_masks = []

    for i in range(len(bounds) - 1):
        lower, upper = bounds[i], bounds[i + 1]

        remove_mask = cv2.inRange(depth_u8, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        remove_mask = cv2.morphologyEx(remove_mask, cv2.MORPH_CLOSE, kernel)
        remove_mask = cv2.GaussianBlur(remove_mask, (5, 5), 0)

        keep_alpha = cv2.bitwise_not(remove_mask)

        b, g, r = cv2.split(image)
        rgba = cv2.merge([b, g, r, keep_alpha])

        layer_path = CUT_DIR / f"layer_{i}_cut.png"
        cv2.imwrite(str(layer_path), rgba)
        saved_layers.append(layer_path)

        mask_rgb = np.ones_like(image, dtype=np.uint8) * 255
        mask_rgba = cv2.merge([
            mask_rgb[:, :, 0],
            mask_rgb[:, :, 1],
            mask_rgb[:, :, 2],
            remove_mask
        ])

        mask_path = MASK_DIR / f"mask_{i}.png"
        cv2.imwrite(str(mask_path), mask_rgba)
        saved_masks.append(mask_path)

        print(f"✅ Layer {i} saved | Depth range: {lower}-{upper}")

    return {
        "layers": saved_layers,
        "masks": saved_masks
    }

# generate_depth_layers(
#     wrapped_image_path=r"F:\TextTo360\pipeline\common files\Edited Images\upscaled_final_extended.png",
#     infilled_image_path=r"F:\TextTo360\pipeline\common files\Final Versions\upscaled_final.png"
# )
