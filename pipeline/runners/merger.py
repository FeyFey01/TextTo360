import cv2
import numpy as np
from pathlib import Path
import cv2
from pathlib import Path

def disk_to_rect(disk, rect_height):
    """
    Converts a circular disk image to a rectangular strip.
    
    disk: H x W x C (BGR or BGRA) image of circle/disk
    rect_height: output rect height (pixels)
    returns: rect H x W x C image
    """
    h, w = disk.shape[:2]
    R = w // 2
    out_h = rect_height
    out_w = w

    ys, xs = np.indices((out_h, out_w), dtype=np.float32)
    theta = xs / out_w * 2 * np.pi           # 0..2pi
    r_norm = ys / (out_h - 1)               # 0..1
    r_pixels = r_norm * (R - 1)

    map_x = R + r_pixels * np.cos(theta)
    map_y = R + r_pixels * np.sin(theta)

    rect = cv2.remap(disk, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rect


def unwrap_and_merge(top_disk_path, main_image_path, bottom_disk_path, strip_height=256, cut_height=200):
    """
    Unwraps top and bottom disk images, merges them with the main image.
    
    top_disk_path: path to top circular disk image
    main_image_path: path to main center image
    bottom_disk_path: path to bottom circular disk image
    strip_height: how tall the unwrapped strips should be
    cut_height: how many pixels to take from top/bottom strips for merging
    returns: merged image as np.ndarray
    """
    # --- load images ---
    top_disk = cv2.imread(top_disk_path, cv2.IMREAD_UNCHANGED)
    bottom_disk = cv2.imread(bottom_disk_path, cv2.IMREAD_UNCHANGED)
    main_img = cv2.imread(main_image_path, cv2.IMREAD_UNCHANGED)

    if top_disk is None or bottom_disk is None or main_img is None:
        raise FileNotFoundError("One of the input images could not be loaded.")

    # --- unwrap top ---
    top_unwrap = disk_to_rect(top_disk, rect_height=strip_height)
    top_part = top_unwrap[:cut_height, :, :3] if top_unwrap.shape[2] == 4 else top_unwrap[:cut_height, :, :]

    # --- unwrap bottom ---
    bottom_unwrap = disk_to_rect(bottom_disk, rect_height=strip_height)
    bottom_unwrap_flipped = cv2.flip(bottom_unwrap, 0)
    bottom_part = bottom_unwrap_flipped[-cut_height:, :, :3] if bottom_unwrap_flipped.shape[2] == 4 else bottom_unwrap_flipped[-cut_height:, :, :]

    # --- center part from main image ---
    orig_h, orig_w = main_img.shape[:2]
    center_part = main_img[cut_height:orig_h - cut_height, :, :]

    # --- merge vertically ---
    merged = np.vstack([top_part, center_part, bottom_part])

    # --- SAVE LOGIC ---
    # pipeline\runners\merger.py -> .. -> .. -> pipeline\
    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / "common files" / "Final Versions"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Kayıt yolu (Varsayılan olarak merged_output.png)
    save_path = save_dir / "final_merged_panorama.png"
    
    # OpenCV ile kaydet (merged dizisi BGR veya BGRA olmalıdır)
    cv2.imwrite(str(save_path), merged)

    print(f"Merged image saved to: {save_path}")

    # Path nesnesini döndür
    return save_path


# # --- Example usage ---
# if __name__ == "__main__":
#     merged_img = unwrap_and_merge(
#         top_disk_path=r"F:/TextTo360/pipeline/common files/Generated Images/top_circle_inpainted.png",
#         main_image_path=r"F:/TextTo360/pipeline/common files/Generated Images/infilled_result.png",
#         bottom_disk_path=r"F:/TextTo360/pipeline/common files/Generated Images/bottom_circle_inpainted.png",
#         strip_height=256,
#         cut_height=200
#     )


