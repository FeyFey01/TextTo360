import base64
import requests
from pathlib import Path
from typing import Union
from dotenv import load_dotenv
import os   

load_dotenv()
API_URL = os.getenv("EXTRA_SINGLE_IMAGE")


def upscale_image_extras(
    image: Union[str, Path, bytes],
    scale: int = 8,
    upscaler_1: str = "R-ESRGAN 4x+ Anime6B",
    upscaler_2: str = "4xSmoothRealism",
    upscaler_2_visibility: float = 0.15,
) -> bytes:
    """
    Upscale an image using Stable Diffusion WebUI Extras API.

    Parameters
    ----------
    image : str | Path | bytes
        Input image path or raw image bytes.
    scale : int
        Upscaling factor.
    upscaler_1 : str
        Primary upscaler.
    upscaler_2 : str
        Secondary upscaler.
    upscaler_2_visibility : float
        Blend strength of secondary upscaler (0–1).

    Returns
    -------
    bytes
        Upscaled image as raw bytes (PNG).
    """

    # -----------------------------
    # Load image → base64
    # -----------------------------
    if isinstance(image, (str, Path)):
        image_bytes = Path(image).read_bytes()
    elif isinstance(image, bytes):
        image_bytes = image
    else:
        raise TypeError("image must be a file path or bytes")

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # -----------------------------
    # Payload
    # -----------------------------
    payload = {
        "image": image_b64,

        "resize_mode": 0,              # scale by factor
        "upscaling_resize": scale,
        "upscaling_crop": False,

        "upscaler_1": upscaler_1,
        "upscaler_2": upscaler_2,
        "extras_upscaler_2_visibility": upscaler_2_visibility,

        "upscale_first": True,

        "gfpgan_visibility": 0,
        "codeformer_visibility": 0,
    }

    # -----------------------------
    # Request
    # -----------------------------
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()

    result = response.json()

    # -----------------------------
    # Decode output
    # -----------------------------
    image_bytes = base64.b64decode(result["image"])

    # --- SAVE LOGIC ---
    # Mevcut dosyanın konumundan (pipeline/klasor/dosya.py) pipeline ana dizinine çıkış:
    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / "common files" / "Final Versions"
    
    # Klasör yoksa oluştur
    save_dir.mkdir(parents=True, exist_ok=True)

    # Dosya adını belirle (Örn: final_result.png)
    save_path = save_dir / "upscaled_final.png"

    with open(save_path, "wb") as f:
        f.write(image_bytes)

    print(f"Final version saved to: {save_path}")

    # Path nesnesini döndür (TypeError almamak için)
    return save_path


# upscaled_bytes = upscale_image_extras(
#     "F:/TextTo360/pipeline/common files/Final Versions/final_merged_panorama.png",
#     scale=8
# )

# with open("F:/TextTo360/output_upscaled.png", "wb") as f:
#     f.write(upscaled_bytes)