import base64
import requests
from pathlib import Path
from typing import Union
from dotenv import load_dotenv
import os   

# =========================================================
# API
# =========================================================
load_dotenv()
API_URL = os.getenv("IMG2IMG")

# =========================================================
# CONSTANT SETTINGS (MATCH WEBUI)
# =========================================================

MODEL_NAME = "abyssorangemix2SFW_abyssorangemix2Sfw"

PROMPT = (
    "masterpiece, high quality, render, digital illustration, sharp edges, "
    "smooth, video game, soft textures, fix the inconsistency "
    "<lora:J_game_background:0.8>"
)

NEGATIVE_PROMPT = (
    "realistic, hyperrealism, photo, rough textures, "
    "particles, small space, human, animals"
)

WIDTH = 1024
HEIGHT = 512

STEPS = 35
CFG_SCALE = 4
DENOISING_STRENGTH = 0.82

SAMPLER_NAME = "DPM++ 2M"
SCHEDULER = "Karras"

# ---- Inpainting ----
INPAINT_FILL = 0                 # fill
MASK_BLUR = 12
MASK_PADDING = 104

# ---- Soft Inpainting (EXACT MATCH) ----
SOFT_INPAINTING = True
SOFT_SCHEDULE_BIAS = 1
SOFT_PRESERVATION = 2.15
SOFT_TRANSITION_CONTRAST = 11
SOFT_MASK_INFLUENCE = 0.75
SOFT_DIFF_THRESHOLD = 3.25
SOFT_DIFF_CONTRAST = 1.75


# =========================================================
# FUNCTION
# =========================================================

def img2img_inpaint_midseam(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
) -> bytes:
    """
    Img2img inpainting call matching AUTOMATIC1111 WebUI settings exactly.
    Returns generated image as bytes.
    """

    init_image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    mask_b64 = base64.b64encode(Path(mask_path).read_bytes()).decode()

    payload = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,

        "init_images": [init_image_b64],
        "mask": mask_b64,

        "width": WIDTH,
        "height": HEIGHT,

        "steps": STEPS,
        "cfg_scale": CFG_SCALE,
        "denoising_strength": DENOISING_STRENGTH,

        "sampler_name": SAMPLER_NAME,
        "scheduler": SCHEDULER,

        "resize_mode": 0,

        "inpaint_full_res": True,
        "inpaint_full_res_padding": MASK_PADDING,
        "inpainting_fill": INPAINT_FILL,
        "inpainting_mask_invert": 0,
        "mask_blur": MASK_BLUR,

        "restore_faces": False,
        "tiling": False,

        "override_settings": {
            "sd_model_checkpoint": MODEL_NAME
        },

        "alwayson_scripts": {
            "Soft Inpainting": {
                "args": [
                    SOFT_INPAINTING,
                    SOFT_SCHEDULE_BIAS,
                    SOFT_PRESERVATION,
                    SOFT_TRANSITION_CONTRAST,
                    SOFT_MASK_INFLUENCE,
                    SOFT_DIFF_THRESHOLD,
                    SOFT_DIFF_CONTRAST,
                ]
            }
        },

        "send_images": True,
        "save_images": False,
    }

    response = requests.post(API_URL, json=payload)
    response.raise_for_status()

    result = response.json()
    image_data = base64.b64decode(result["images"][0])

    # --- SAVE LOGIC ---
    # Get the pipeline root directory relative to this script
    # pipeline\webui\image_infill_seam.py -> .. -> .. -> pipeline\
    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / "common files" / "Generated Images"
    
    # Ensure the directory exists
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Define save path (using a default name, or you can customize)
    save_path = save_dir / "infilled_result.png"
    
    with open(save_path, "wb") as f:
        f.write(image_data)

    print(f"Image saved to: {save_path}")

    # Return the Path object so your pipeline doesn't throw a TypeError
    return save_path


# =========================================================
# USAGE
# =========================================================

# output_bytes = img2img_inpaint_midseam(
#     image_path="F:/TextTo3601/pipeline/common files/Generated Images/abyssorangemix_seed_517337055_swapped.png",
#     mask_path="F:/TextTo3601/pipeline/common files/Generated Images/abyssorangemix_seed_517337055_swapped_center_mask.png"
# )

# with open("F:/TextTo360/output_infilled.png", "wb") as f:
#     f.write(output_bytes)
