# Text to 360° VR Environment Generation Pipeline

This repository contains an end-to-end pipeline that turns a single text prompt into a 360° VR-ready environment.  
The system combines LLM-assisted prompt refinement, Stable Diffusion image generation, seam correction & infill, depth-based layer decomposition, and Blender automation to produce immersive 360° VR-ready environments designed to be accessible on standard consumer hardware.

This project is developed as a Bachelor’s Graduation Project at **Istanbul Technical University (İTÜ)** under the supervision of **Assoc. Prof. Dr. Gökhan İnce**, and is actively conducted with the guidance and collaboration of **Research Assistant Meral Korkmaz Kuyucu** and **Research Assistant Bora Şenceylan**.

---

## Quick summary

- Input: a single text prompt  
- Output: 360° equirectangular panorama processed into depth-aware layers and exported as a Blender scene (nested spheres) suitable for VR
- Key goals: seam-safe panorama generation, top/bottom pole infill, depth map generation, depth clustering, layered RGBA masks, Blender automation

---

## Models & LoRAs (used in the pipeline)

- **World generation, seam correction, top/bottom infill model**  
  `abyssorangemix2SFW_abyssorangemix2Sfw.safetensors`  
  Download: [Hugging Face / abyssorangemix2SFW](https://huggingface.co/Karumoon/test00a1/blob/main/abyssorangemix2SFW_abyssorangemix2Sfw.safetensors)

- **360 / equirectangular LoRA**  
  `360Diffusion LoRA for SD-1.5`  
  CivitAI: [360-Diffusion LoRA for SD-1.5](https://civitai.com/models/26815/360-diffusion-lora-for-sd-15)

- **Game-assets / cartoon-style LoRA**  
  CivitAI: [Game Assets — Cartoon Style / Isometric Backgrounds](https://civitai.com/models/317059/game-assets-cartoon-style-3d-isometric-background-assets-for-small-games)

> **Note:**  
> For seam correction, inseam fixes, and top/bottom infill steps, the 360 / equirectangular LoRAs are intentionally **not applied**.  
> These steps should avoid projection-specific LoRA influence so that inpainting focuses purely on local continuity and visual coherence rather than global projection constraints.

---

## Extensions & Tools

- **Asymmetric Tiling (to avoid edge issues)** — repository: [asymmetric-tiling-sd-webui](https://github.com/tjm35/asymmetric-tiling-sd-webui)  
  (Enabled in AUTOMATIC1111 as an always-on script for img2img/upsampling steps.)

- **Upscaling** — use the WebUI *Extras* upscalers (ESRGAN / UniScale or chained upscalers) from your installed upscalers list.  
  Example combo used in the pipeline:  
  `2x-UniScale-CartoonRestore-lite` → `R-ESRGAN 4x+ Anime6B`

  - `2x-UniScale-CartoonRestore-lite` — OpenModelDB:  
    [https://openmodeldb.info/models/2x-UniScale-CartoonRestore-lite](https://openmodeldb.info/models/2x-UniScale-CartoonRestore-lite)

  - `R-ESRGAN 4x+ Anime6B` — OpenModelDB:  
    [https://openmodeldb.info/models/4x-realesrgan-x4plus-anime-6b](https://openmodeldb.info/models/4x-realesrgan-x4plus-anime-6b)

- **Reference & prototype Blender extension**: Albert Bozesan — YouTube channel: [Albert Bozesan](https://www.youtube.com/@albertbozesan)  
  *Note:* Albert’s Blender extension was used for prototyping; the final pipeline uses a custom Blender Python script for reproducibility.

---

## Pipeline (high-level)

1. **Prompt → Keywords**  
   - Expand user prompt using an LLM helper to generate SD-friendly scene keywords.

2. **Image generation (AUTOMATIC1111)**  
   - Generate multiple candidates (model: `abyssorangemix2SFW_abyssorangemix2Sfw`, LoRAs as listed above).
   - User selects the best candidate via GUI selector.

3. **Horizontal seam/inseam correction**  
   - Perform left/right seam swap & mask the seam region.
   - Use `img2img` inpainting to remove seam artifacts (note: do not apply 360 LoRAs in this step).

4. **Top / bottom pole correction**  
   - Convert top/bottom strips to circular patches, create alpha masks, inpaint them individually as needed (again: avoid 360 LoRAs for these local infill operations).

5. **Merge panorama**  
   - Reassemble top / middle / bottom into a single equirectangular panorama, ensuring seam continuity.

6. **Upscale**  
   - Postprocess upscaling via Extras API (example: chain `2x-UniScale-CartoonRestore-lite` then `R-ESRGAN 4x+ Anime6B`), or use LDSR/StableSR for latent-aware super-resolution where appropriate.

*(planned / next steps)* 

7. **Depth map generation**  
   - Produce a dense depth map from the upscaled panorama (any SOTA monocular depth estimator).

8. **Depth-based clustering → RGBA layers**  
   - Cluster depth into discrete layers (foreground → background).
   - For each cluster, export an RGBA image with alpha = mask (alpha = 0 outside the layer).

9. **Blender scene assembly**  
   - Import RGBA textures into Blender.
   - Generate nested spheres (concentric), assign each layer’s texture to a sphere with its alpha as mask.
   - Adjust sphere radii based on depth layer, disable backface rendering where appropriate.

10. **VR export / testing**  
    - Export scene for VR viewer or test inside Blender’s VR preview / export to game engine.

---

## Installation & Setup

This pipeline is designed to be accessible and runnable on standard personal computers.  
All steps below assume **Python 3.10.11**, which is the version used during development and testing.

---

### 1. Prerequisites

- **Python 3.10.11**  
  Download from: https://www.python.org/downloads/release/python-31011/  
  Make sure Python is added to your system `PATH`.

- **AUTOMATIC1111 Stable Diffusion WebUI**  
  Follow the official installation guide:  
  https://github.com/AUTOMATIC1111/stable-diffusion-webui

- **Git** (optional but recommended)  
  https://git-scm.com/

---

### 2. Clone the Repository

```bash
git clone https://github.com/FeyFey01/TextTo360.git
cd TextTo360
```

---

### 3. Create and Activate a Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

---

### 4. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 5. Configure Environment Variables (`.env`)

#### API Keys
- **IMGBB_API_KEY**  
  Get from: https://api.imgbb.com/

- **OPENAI_API_KEY**  
  Get from: https://platform.openai.com/

> Change your .env file with your API keys.

---

### 6. Install Models, LoRAs, and Extensions in WebUI

Place the downloaded files into the appropriate WebUI folders:

- **Stable Diffusion models** → `models/Stable-diffusion/`
- **LoRAs** → `models/Lora/`
- **Upscalers** → `models/ESRGAN/`
- **WebUI extensions** → `extensions/`

---

### 7. Run the Pipeline

Ensure the WebUI is running locally at:
```
http://127.0.0.1:7860
```

Once the WebUI is running and environment variables are set:

```bash
python pipeline\run_me.py
```

You will be prompted for a text description.  
The pipeline will automatically handle:

- Prompt refinement (LLM-assisted)
- World image generation
- Seam correction & infill
- Top / bottom pole correction
- Upscaling
- 360° preview upload
- (Optional) depth-based layer decomposition and Blender integration

---

### Notes

* The pipeline is **modular** — individual stages can be skipped, replaced, or re-run independently.
* Designed to work **incrementally**, allowing experimentation without regenerating everything from scratch.
* Final output is intended to be compatible with **VR viewers** and downstream **3D pipelines**.
* ⚠️ **This pipeline is still a work in progress**. Some stages are experimental, manual, or partially automated, and the overall flow is expected to evolve.

---

### TODO / Roadmap

* [ ] **Depth map generation automation**

  * Automate depth map extraction from generated panoramas
  * Experiment with different depth estimation models
  * Standardize depth map resolution and format

* [ ] **Depth layer clustering & separation**

  * Cluster depth maps into discrete depth layers
  * Separate foreground / midground / background regions
  * Experiment with thresholding vs. K-means / hierarchical clustering
  * Preserve edge continuity between layers

* [ ] **Depth-aware layer processing**

  * Refine individual layers (cleanup, smoothing, inpainting if needed)
  * Maintain consistency between neighboring depth layers
  * Store depth + color layers in a reusable intermediate format

* [ ] **Blender integration**

  * Import panorama, depth maps, and separated layers into Blender
  * Set up shaders/materials using depth information (parallax, displacement, masking)
  * Configure camera and projection for 360° / VR viewing
  * Construct a basic Blender scene/environment using the generated assets

* [ ] **Scene construction & rendering**

  * Build a full Blender scene from layered depth data
  * Test lighting, shading, and depth-based effects
  * Render test outputs for VR and non-VR pipelines

* [ ] **Cross-system testing**

  * Test the full pipeline on other computers
  * Verify path handling, dependencies, and environment setup
  * Identify system-specific issues (GPU, drivers, Python versions)

---

