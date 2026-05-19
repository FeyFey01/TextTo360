import cv2
import numpy as np
from pathlib import Path


# ===============================
# ROOT PATHS
# ===============================
PIPELINE_DIR = Path(__file__).resolve().parents[1]
EDITED_DIR = PIPELINE_DIR / "common files" / "Edited Images"
EDITED_DIR.mkdir(parents=True, exist_ok=True)


def extend_image_horizontal_wrap(
    input_image,
    output_path: Path | None = None
):
    """
    Extends an image horizontally for 360 processing.

    Layout:
        [ right half | original image | left half ]
    """

    # -------- LOAD IMAGE --------
    if isinstance(input_image, (str, Path)):
        input_path = Path(input_image)

        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found: {input_path}")

        stem = input_path.stem
        suffix = input_path.suffix
    else:
        img = input_image
        stem = "extended_image"
        suffix = ".png"

    h, w = img.shape[:2]
    mid = w // 2

    # -------- SPLIT --------
    left_half = img[:, :mid]
    right_half = img[:, mid:]

    # -------- EXTEND --------
    extended = np.hstack((right_half, img, left_half))

    # -------- OUTPUT PATH --------
    if output_path is None:
        output_path = EDITED_DIR / f"{stem}_extended{suffix}"
    else:
        output_path = Path(output_path)

    # -------- SAVE --------
    cv2.imwrite(str(output_path), extended)

    print(f"✅ Saved extended image for depth generation: {output_path}")
    print(f"Original size: {w}x{h}")
    print(f"Extended size: {extended.shape[1]}x{h}")

    return output_path


# ===============================
# RUNNER
# ===============================
# if __name__ == "__main__":
#     input_image_path = (
#         PIPELINE_DIR
#         / "common files"
#         / "Final Versions"
#         / "upscaled_final.png"
#     )

#     extend_image_horizontal_wrap("F:/TextTo360/00049.jpg")
