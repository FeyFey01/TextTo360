import subprocess
import sys
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

from runners.gpt import generate_scene_keywords
from runners.run_cmd import run_cmd_live_json
from runners.visual_selector import select_image_gui
from runners.center_mask import create_center_alpha_mask
from runners.horizontal_swap_func import swap_image_halves
from runners.top_circle import top_strip_to_circle_and_mask
from runners.bottom_circle import bottom_strip_to_circle_and_mask
from runners.merger import unwrap_and_merge
from runners.viewer_360 import upload_and_view

from webui.image_upscale_v1 import upscale_image_extras
from webui.image_inpaint import img2img_inpaint
from webui.image_infill_seam import img2img_inpaint_midseam

# =========================================================
# COMMON FILES DIRECTORY SETUP
# =========================================================


PIPELINE_DIR = Path(__file__).resolve().parent

IMAGE_GEN_SCRIPT = PIPELINE_DIR / "image_generation_v1.py"

COMMON_FILES_DIR = PIPELINE_DIR / "common files"

GENERATED_IMAGES_DIR = COMMON_FILES_DIR / "Generated Images"
EDITED_IMAGES_DIR = COMMON_FILES_DIR / "Edited Images"
FINAL_VERSIONS_DIR = COMMON_FILES_DIR / "Final Versions"

# Create directories if they do not exist
for directory in [
    COMMON_FILES_DIR,
    GENERATED_IMAGES_DIR,
    EDITED_IMAGES_DIR,
    FINAL_VERSIONS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)



# ---- ASK USER INPUT ----
user_prompt = input("Enter scene description: ").strip()

# -------- ENHANCE KEYWORDS --------
keywords = generate_scene_keywords(user_prompt)
print("\nEnhanced keywords for image generation:", keywords["scene"])



# ============= COMMENT TO DEBUG WITH PRE-GENERATED IMAGE ==============

# -------- RUN IMAGE GENERATION (VISIBLE OUTPUT) --------
generate_cmd = [
    sys.executable,
    str(IMAGE_GEN_SCRIPT),
    "-t",
    keywords["scene"]
]

generated_images = run_cmd_live_json(generate_cmd)


# -------- SELECT IMAGE FROM GENERATED ONES --------
print("\nOpening image selector...")

selected_image = select_image_gui(generated_images)

if selected_image:
    print("\nSelected image:", Path(selected_image).name)
else:
    print("\nNo image selected")

# ================= END COMMENT ===================

# UNCOMMENT HERE IF DEBUGGING WITH PRE-GENERATED IMAGE
# selected_image = r"F:\TextTo360\pipeline\common files\Generated Images\abyssorangemix_seed_517337055.png"


# -------- RUN PANORAMA SWAP (VISIBLE OUTPUT) --------
# swap_cmd = [
#     sys.executable,
#     str(PIPELINE_DIR / "runners" / "horizontal_swap.py"),
#     "-i", selected_image,
#     "-o", selected_image.replace(".png", "_swapped.png")
# ]

# swapped_image = run_cmd_live_json(swap_cmd)

out = swap_image_halves(
    selected_image
)

mask_img = create_center_alpha_mask(
    image_path=out,
    strip_width=60
)

# -------- RUN INFILL --------

infilled_image = img2img_inpaint_midseam(
    image_path=out,
    mask_path=Path(mask_img)
)

# -------- CREATE TOP AND BOTTOM CIRCLES AND MASKS --------

top_circle, top_masked_circle = top_strip_to_circle_and_mask(
    infilled_image
)

bottom_circle, bottom_masked_circle = bottom_strip_to_circle_and_mask(
    infilled_image
)

# -------- (ASK IF) RUN INFILL --------
is_fill_top = input("Infill top circle? (y/n): ").strip().lower() == "y"
is_fill_bottom = input("Infill bottom circle? (y/n): ").strip().lower() == "y"

if is_fill_top:
    print("Top circle will be infilled.")

    top_infilled = img2img_inpaint(
        image_path=top_circle,
        mask_path=top_masked_circle,
        extra_prompt=keywords["top"]
    )

    selected_top = select_image_gui([
        top_infilled,
        top_circle
    ])

else:
    selected_top = top_circle


if is_fill_bottom:
    print("Bottom circle will be infilled.")

    bottom_infilled = img2img_inpaint(
        image_path=bottom_circle,
        mask_path=bottom_masked_circle,
        extra_prompt=keywords["bottom"]
    )

    selected_bottom = select_image_gui([
        bottom_infilled,
        bottom_circle
    ])

else:
    selected_bottom = bottom_circle

# -------- MERGE 3 PARTS --------

merged_img = unwrap_and_merge(
    top_disk_path=selected_top,
    main_image_path=bottom_infilled,
    bottom_disk_path=selected_bottom
)

upload_and_view(merged_img)

#-------- UPSCALE RESULT (VISIBLE OUTPUT) --------

upscaled_image = upscale_image_extras(
    merged_img,
    scale=8
)

upload_and_view(upscaled_image)


