import cv2
import numpy as np
from pathlib import Path

def disk_to_rect(disk, rect_height):
    """
    Converts a circular disk image to a rectangular strip without stretching/seam issues.
    """
    h, w = disk.shape[:2]
    R = w // 2
    out_h = rect_height
    out_w = w

    xs = np.linspace(0, out_w, out_w, endpoint=False) + 0.5
    ys = np.linspace(0, out_h, out_h, endpoint=False) + 0.5
    xs, ys = np.meshgrid(xs, ys)

    theta = xs / out_w * 2 * np.pi
    r_norm = ys / out_h
    r_pixels = r_norm * (R - 1)

    map_x = R + r_pixels * np.cos(theta)
    map_y = R + r_pixels * np.sin(theta)

    rect = cv2.remap(disk, map_x.astype(np.float32), map_y.astype(np.float32),
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rect


def unwrap_and_merge(
    top_disk_path, main_image_path, bottom_disk_path,
    strip_height=256, cut_height=200,
    top_infilled=True, bottom_infilled=True
):
    """
    Unwraps top/bottom disks and merges with main image.
    
    top_infilled: if True, merge only top cut_height, else use full top strip
    bottom_infilled: if True, merge only bottom cut_height, else use full bottom strip
    """
    # --- load images ---
    top_disk = cv2.imread(top_disk_path, cv2.IMREAD_UNCHANGED)
    bottom_disk = cv2.imread(bottom_disk_path, cv2.IMREAD_UNCHANGED)
    main_img = cv2.imread(main_image_path, cv2.IMREAD_UNCHANGED)

    if top_disk is None or bottom_disk is None or main_img is None:
        raise FileNotFoundError("One of the input images could not be loaded.")

    # --- unwrap top ---
    top_unwrap = disk_to_rect(top_disk, rect_height=strip_height)
    if top_infilled:
        top_part = top_unwrap[:cut_height, :, :3] if top_unwrap.shape[2] == 4 else top_unwrap[:cut_height, :, :]
    else:
        top_part = top_unwrap[:, :, :3] if top_unwrap.shape[2] == 4 else top_unwrap

    # --- unwrap bottom ---
    bottom_unwrap = disk_to_rect(bottom_disk, rect_height=strip_height)
    bottom_unwrap_flipped = cv2.flip(bottom_unwrap, 0)
    if bottom_infilled:
        bottom_part = bottom_unwrap_flipped[-cut_height:, :, :3] if bottom_unwrap_flipped.shape[2] == 4 else bottom_unwrap_flipped[-cut_height:, :, :]
    else:
        bottom_part = bottom_unwrap_flipped[:, :, :3] if bottom_unwrap_flipped.shape[2] == 4 else bottom_unwrap_flipped

    # --- center part from main image ---
    orig_h, orig_w = main_img.shape[:2]
    top_h = top_part.shape[0] if top_infilled else 0
    bottom_h = bottom_part.shape[0] if bottom_infilled else 0
    center_part = main_img[top_h:orig_h - bottom_h, :, :]

    # --- merge vertically ---
    merged = np.vstack([top_part, center_part, bottom_part])

    # --- SAVE LOGIC ---
    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / "common files" / "Final Versions"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / "final_merged_panorama.png"
    cv2.imwrite(str(save_path), merged)

    print(f"Merged image saved to: {save_path}")
    return save_path


# --- Example usage ---
if __name__ == "__main__":
    unwrap_and_merge(
        top_disk_path=r"F:/TextTo360/pipeline/common files/Edited Images/top_circle.png",
        main_image_path=r"F:/TextTo360/pipeline/common files/Generated Images/generated_random.png",
        bottom_disk_path=r"F:/TextTo360/pipeline/common files/Edited Images/bottom_circle.png",
        strip_height=256,
        cut_height=200,
        top_infilled=True,       # Merge only top cut_height
        bottom_infilled=True    # Merge full bottom disk
    )