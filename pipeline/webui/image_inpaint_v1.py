import argparse
import json
import torch
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from pathlib import Path
from PIL import Image

# -------- ARGUMENTS --------
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prompt", required=True, help="Inpaint prompt")
parser.add_argument("-i", "--image", required=True, help="Input image path")
parser.add_argument("-m", "--mask", required=True, help="Mask image path (white=inpaint)")
args = parser.parse_args()

prompt_text = args.prompt
input_image_path = Path(args.image)
mask_path = Path(args.mask)

# -------- PATHS --------
BASE_DIR = Path("F:/TextTo360")

MODEL_PATH = BASE_DIR / "models" / "Stable-diffusion" / "abyssorangemix2SFW_abyssorangemix2Sfw.safetensors"
LORA_DIR = BASE_DIR / "models" / "Lora"
OUTPUT_DIR = BASE_DIR / "pipeline" / "common files" / "inpainted images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- DEVICE --------
device = "cuda"

# -------- LOAD PIPELINE --------
pipe = StableDiffusionInpaintPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.enable_attention_slicing()
pipe.vae.enable_slicing()

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True
)

# -------- LOAD LORAs --------
pipe.load_lora_weights(LORA_DIR, weight_name="360Diffusion_v1.safetensors", adapter_name="360Diffusion_v1")
pipe.load_lora_weights(LORA_DIR, weight_name="J_game_background.safetensors", adapter_name="J_game_background")

pipe.set_adapters(
    ["360Diffusion_v1", "J_game_background"],
    adapter_weights=[1.0, 0.8]
)

# -------- PROMPTS --------
prompt = (
    "masterpiece, high quality, render, digital illustration, sharp edges, smooth, "
    "video game, soft textures, "
    f"{prompt_text}, "
    "j_game_background"
)

negative_prompt = (
    "realistic, hyperrealism, photo, rough textures, particles, small space"
)

# -------- LOAD IMAGES --------
init_image = Image.open(input_image_path).convert("RGB")
mask_image = Image.open(mask_path).convert("L")

# -------- GENERATION --------
generator = torch.Generator(device).manual_seed(517337054)

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    mask_image=mask_image,
    width=init_image.width,
    height=init_image.height,
    num_inference_steps=25,
    guidance_scale=6,
    strength=0.99,   # critical for inpaint
    generator=generator
).images[0]

# -------- SAVE & RETURN --------
out_path = OUTPUT_DIR / f"inpainted_{input_image_path.stem}.png"
result.save(out_path)

print(json.dumps(str(out_path)))
