import argparse
import json
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from pathlib import Path

from runners.viewer_360 import upload_and_view

# -------- ARGUMENTS --------
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--text", required=True, help="Scene description")
args = parser.parse_args()

user_text = args.text

# -------- PATHS --------
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent  # <-- THIS is the missing piece

MODEL_PATH = ROOT_DIR / "models" / "Stable-diffusion" / "abyssorangemix2SFW_abyssorangemix2Sfw.safetensors"
LORA_DIR = ROOT_DIR / "models" / "Lora"
OUTPUT_DIR = ROOT_DIR / "pipeline" / "common files" / "generated images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- DEVICE --------
device = "cuda"

# -------- LOAD PIPELINE --------
pipe = StableDiffusionPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True
)

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
    f"{user_text}, "
    "j_game_background"
)

negative_prompt = (
    "realistic, hyperrealism, photo, above the clouds, rough textures, particles, small space"
)

# -------- GENERATION --------
base_seed = 517337054
batch_count = 2
generated_paths = []

for i in range(batch_count):
    seed = base_seed + i
    generator = torch.Generator(device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=512,
        num_inference_steps=25,
        guidance_scale=6,
        generator=generator
    ).images[0]

    out_path = OUTPUT_DIR / f"abyssorangemix_seed_{seed}.png"
    image.save(out_path)
    generated_paths.append(str(out_path))

    upload_and_view(str(out_path))


# -------- RETURN OUTPUT (stdout) --------
print(json.dumps(generated_paths))
