import cv2
import numpy as np
from pathlib import Path


# Root paths
PIPELINE_DIR = Path(__file__).resolve().parents[1]
EDITED_DIR = PIPELINE_DIR / "common files" / "Edited Images"
EDITED_DIR.mkdir(parents=True, exist_ok=True)


def swap_image_halves(
    input_image,
    output_path: Path | None = None
):
    """
    Swaps left/right halves of an image without changing resolution.

    input_image: str | Path | np.ndarray
    output_path: optional Path
    returns: Path to saved image
    """

    # -------- LOAD IMAGE --------
    if isinstance(input_image, (str, Path)):
        input_path = Path(input_image)
        print(f"Input is this: {input_path}")

        img = cv2.imread(str(input_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {input_path}")
        stem = input_path.stem
        suffix = input_path.suffix
    else:
        img = input_image
        stem = "swapped_image"
        suffix = ".png"

    h, w, c = img.shape
    mid = w // 2

    # -------- SPLIT & SWAP --------
    left = img[:, :mid]
    right = img[:, mid:]
    swapped = np.hstack((right, left))

    print("Image is swapped yey 🌀")

    # -------- OUTPUT PATH --------
    if output_path is None:
        output_path = EDITED_DIR / f"{stem}_swapped{suffix}"
    else:
        output_path = Path(output_path)

    # -------- SAVE --------
    cv2.imwrite(str(output_path), swapped)

    print(f"Image saved to: {output_path}")
    print(f"Resolution preserved: {w}x{h}")

    return output_path
