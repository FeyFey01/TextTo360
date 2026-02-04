import cv2
import numpy as np
from pathlib import Path


def rect_strip_to_disk(strip, radius=None, invert_radial=False):
    h, w = strip.shape[:2]
    if radius is None:
        radius = w // 2
    R = int(radius)
    out_h = out_w = 2 * R

    ys, xs = np.indices((out_h, out_w), dtype=np.float32)
    dx = xs - R
    dy = ys - R
    r_pixels = np.sqrt(dx * dx + dy * dy)
    mask = r_pixels <= (R - 1e-6)

    r_norm = r_pixels[mask] / (R - 1)
    y_src = (1.0 - r_norm) * (h - 1) if invert_radial else r_norm * (h - 1)

    theta = np.arctan2(dy[mask], dx[mask])
    theta[theta < 0] += 2 * np.pi
    x_src = theta / (2 * np.pi) * (w - 1)

    map_x = np.full((out_h, out_w), -1, dtype=np.float32)
    map_y = np.full((out_h, out_w), -1, dtype=np.float32)
    map_x[mask] = x_src
    map_y[mask] = y_src

    strip_c = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR) if strip.ndim == 2 else strip
    disk = cv2.remap(
        strip_c,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return disk


def create_mask_from_disk(disk, mask_radius_ratio=0.5):
    h, w = disk.shape[:2]
    R = w // 2
    mask_R = int(R * mask_radius_ratio)

    ys, xs = np.indices((h, w))
    dx = xs - R
    dy = ys - R
    r_pixels = np.sqrt(dx**2 + dy**2)

    alpha = np.zeros((h, w), dtype=np.uint8)
    alpha[r_pixels <= mask_R] = 255

    bgra = cv2.cvtColor(disk, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


def top_strip_to_circle_and_mask(
    image,
    strip_height=256,
    mask_radius_ratio=128 / 256,
    invert_radial=False,
):
    """
    image: np.ndarray or image path
    returns: (circle_image_bgra, circle_mask_bgra)
    """

    # Load if path
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Couldn't open image: {image}")
    else:
        img = image

    h, w = img.shape[:2]

    # Take top strip
    strip = img[0:strip_height, :, ...]

    # Wrap to disk
    circle = rect_strip_to_disk(
        strip,
        radius=None,
        invert_radial=invert_radial
    )

    # Ensure BGRA
    circle_bgra = cv2.cvtColor(circle, cv2.COLOR_BGR2BGRA)

    # Create masked circle
    circle_mask = create_mask_from_disk(
        circle,
        mask_radius_ratio=mask_radius_ratio
    )

    # --- SAVE LOGIC ---
    # pipeline\runners\top_circle.py -> .. -> .. -> pipeline\
    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / "common files" / "Edited Images"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define paths
    circle_path = save_dir / "top_circle.png"
    mask_path = save_dir / "top_circle_mask.png"

    # Save using OpenCV (cv2 uses BGR/BGRA)
    cv2.imwrite(str(circle_path), circle_bgra)
    cv2.imwrite(str(mask_path), circle_mask)

    print(f"Saved: {circle_path}")
    print(f"Saved: {mask_path}")

    # Return the objects and the paths (or just the objects as before)
    return circle_path, mask_path

# circle, masked_circle = top_strip_to_circle_and_mask(
#     r"F:/TextTo360/pipeline/common files/Generated Images/abyssorangemix_seed_517337057_infilled.png"
# )
