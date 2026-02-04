import base64
import requests
from pathlib import Path
from typing import Union
import json
from dotenv import load_dotenv
import os

# =========================================================
# API
# =========================================================

load_dotenv()
API_URL = os.getenv("IMG2IMG")

# =========================================================
# CONSTANT SETTINGS (EDIT HERE)
# =========================================================

MODEL_NAME = "abyssorangemix2SFW_abyssorangemix2Sfw"

BASE_PROMPT = (
    "masterpiece, high quality, render, digital illustration, sharp edges, "
    "smooth, video game, cartoon, soft textures, pov, looking up, top only, "
    "simple shapes, fisheye,"
)

NEGATIVE_PROMPT = (
    "realistic, hyperrealism, photo, rough textures, particles, "
    "small space, human, animals"
)


# ⚠️ Make sure this matches your init image if you keep it
WIDTH = 1024
HEIGHT = 1024

STEPS = 40
CFG_SCALE = 20
DENOISING_STRENGTH = 0.73

SAMPLER_NAME = "DPM++ 2M"

# =========================================================
# INPAINTING SETTINGS
# =========================================================

# WebUI mapping:
# False → Whole picture
# True  → Only masked
INPAINT_ONLY_MASKED = False

INPAINT_FILL = 1          # 0 = fill, 1 = original, 2 = latent noise, 3 = latent nothing
MASK_BLUR = 22
MASK_PADDING = 0


# =========================================================
# SOFT INPAINTING
# =========================================================

SOFT_INPAINTING = True
SOFT_SCHEDULE_BIAS = 1
SOFT_PRESERVATION = 3.25
SOFT_TRANSITION_CONTRAST = 5.5
SOFT_MASK_INFLUENCE = 0.9
SOFT_DIFF_THRESHOLD = 0.5
SOFT_DIFF_CONTRAST = 0.5

# =========================================================
# FUNCTION
# =========================================================


def img2img_inpaint(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    extra_prompt: str = "",
) -> Path:
    """
    Sends an img2img inpainting request to AUTOMATIC1111
    and saves the generated image.

    Returns:
        Path to saved image
    """

    image_path = Path(image_path)
    mask_path = Path(mask_path)

    init_image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    mask_b64 = base64.b64encode(mask_path.read_bytes()).decode()

    prompt = f"{BASE_PROMPT}, {extra_prompt}".strip(", ")

    payload = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,

        "init_images": [init_image_b64],
        "mask": mask_b64,

        "width": WIDTH,
        "height": HEIGHT,

        "steps": STEPS,
        "cfg_scale": CFG_SCALE,
        "denoising_strength": DENOISING_STRENGTH,

        "sampler_name": SAMPLER_NAME,

        # IMPORTANT for inpainting
        "resize_mode": 0,
        "inpainting_fill": INPAINT_FILL,
        "inpaint_full_res": INPAINT_ONLY_MASKED,
        "inpaint_full_res_padding": MASK_PADDING,
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

    # =====================================================
    # DEBUG OUTPUT
    # =====================================================

    print("\n===== IMG2IMG RESPONSE PARAMETERS =====")
    print(json.dumps(result.get("parameters", {}), indent=2))
    print("===== END PARAMETERS =====\n")

    print("===== IMG2IMG RESPONSE INFO =====")
    info = result.get("info", "")
    try:
        print(json.dumps(json.loads(info), indent=2))
    except Exception:
        print(info)
    print("===== END INFO =====\n")

    # =====================================================
    # SAVE IMAGE
    # =====================================================

    image_bytes = base64.b64decode(result["images"][0])

    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / "common files" / "Generated Images"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{image_path.stem}_inpainted.png"

    save_path.write_bytes(image_bytes)

    print(f"Inpainted image saved to: {save_path}")

    return save_path



# --- Example usage ---
if __name__ == "__main__":
    output_bytes = img2img_inpaint(
        image_path="F:/TextTo360/pipeline/common files/Edited Images/bottom_circle.png",
        mask_path="F:/TextTo360/pipeline/common files/Edited Images/bottom_circle_mask.png",
        extra_prompt="flat snow, snow texture, snowy ground, flat, simple, winter, J_game_background <lora:J_game_background:0.8>"  # custom text added to prompt
    )

