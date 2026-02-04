import cv2
import numpy as np
import argparse
from pathlib import Path
import json

# -------- ARGUMENTS --------
parser = argparse.ArgumentParser(description="Swap left/right halves of an image without changing resolution")
parser.add_argument(
    "-i", "--input",
    required=True,
    help="Path to input image"
)
parser.add_argument(
    "-o", "--output",
    default=None,
    help="Path to output image (optional)"
)
args = parser.parse_args()

input_path = Path(args.input)

if args.output:
    output_path = Path(args.output)
else:
    output_path = input_path.with_name(input_path.stem + "_swapped" + input_path.suffix)

# -------- LOAD IMAGE --------
img = cv2.imread(str(input_path))
if img is None:
    raise FileNotFoundError(f"Image not found: {input_path}")

h, w, c = img.shape
mid = w // 2

# -------- SPLIT & SWAP --------
left = img[:, :mid]
right = img[:, mid:]

# If width is odd, keep the extra column in the right half
wrapped = np.hstack((right, left))

# -------- SAVE --------
cv2.imwrite(str(output_path), wrapped)

# Print human-readable info
print(f"Panorama swapped without resizing")
print(f"Resolution preserved: {w}x{h}")
print(f"Saved to: {output_path}")

# Print JSON as last line for run_cmd_live_json()
print(json.dumps(str(output_path)))
