import base64
import requests
from pathlib import Path
from typing import Optional
import json
import os
from dotenv import load_dotenv

# =========================================================
# API
# =========================================================

load_dotenv()
API_URL = os.getenv("TXT2IMG")  # example: http://127.0.0.1:7860/sdapi/v1/txt2img


# =========================================================
# CONSTANT SETTINGS (EDIT THESE)
# =========================================================

MODEL_NAME = "dynavisionXLAllInOneStylized_releaseV0610Bakedvae"

WIDTH = 1024
HEIGHT = 512

STEPS = 20
CFG_SCALE = 7

SAMPLER_NAME = "DPM++ 2M"
SCHEDULER = "Karras"

TILE_X = True
TILE_Y = False

BATCH_SIZE = 1
N_ITER = 1

SEED = -1  # -1 = random

# Prompt additions
PROMPT_SUFFIX = "painterly, animatic, 360, 360view <lora:View360:0.6>"

NEGATIVE_PROMPT = (
    "text, watermark, deformed, glitch, noise, noisy, off-center, deformed"
)

# Output directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "common files" / "generated images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# GENERATION FUNCTION
# =========================================================

def generate_image(
    prompt: str,
    seed: Optional[int] = None,
) -> Path:
    """
    Generates an image using AUTOMATIC1111 API.

    Returns:
        Path to saved image
    """

    if seed is None:
        seed = SEED

    full_prompt = f"{prompt}, {PROMPT_SUFFIX}"

    payload = {

        "prompt": full_prompt,
        "negative_prompt": NEGATIVE_PROMPT,

        "steps": STEPS,
        "cfg_scale": CFG_SCALE,

        "sampler_name": SAMPLER_NAME,
        "scheduler": SCHEDULER,

        "width": WIDTH,
        "height": HEIGHT,

        "seed": seed,

        "batch_size": BATCH_SIZE,
        "n_iter": N_ITER,

        # disable default symmetric tiling
        "tiling": False,

        "override_settings": {
            "sd_model_checkpoint": MODEL_NAME
        },

        # IMPORTANT: asymmetric tiling script
        "alwayson_scripts": {
            "asymmetric tiling": {
                "args": [
                    True,       # Active
                    TILE_X,     # Tile X
                    TILE_Y,     # Tile Y
                    0,          # Start step
                    -1          # Stop step
                ]
            }
        },

        "send_images": True,
        "save_images": False,
    }

    response = requests.post(API_URL, json=payload)
    response.raise_for_status()

    result = response.json()

    # ============================================
    # DEBUG OUTPUT
    # ============================================

    print("\n===== TXT2IMG PARAMETERS =====")
    print(json.dumps(result.get("parameters", {}), indent=2))
    print("===== END PARAMETERS =====\n")

    print("===== TXT2IMG INFO =====")
    info = result.get("info", "")
    try:
        print(json.dumps(json.loads(info), indent=2))
    except Exception:
        print(info)
    print("===== END INFO =====\n")

    # ============================================
    # SAVE IMAGE
    # ============================================

    image_bytes = base64.b64decode(result["images"][0])

    seed_str = str(seed) if seed != -1 else "random"
    save_path = OUTPUT_DIR / f"generated_{seed_str}.png"

    save_path.write_bytes(image_bytes)

    print(f"Image saved to: {save_path}")

    return save_path


# =========================================================
# EXAMPLE USAGE
# =========================================================

if __name__ == "__main__":

    image_path = generate_image(
        prompt="a concept art of an icy lake in rockies, magnificent, painterly, epic, majestic"
    )

    print(image_path)