import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a scene prompt generator for image generation.

You must produce TWO prompt styles.

STYLE 1 — KEYWORDS
Return ~15 comma-separated keywords describing the environment.

STYLE 2 — SIMPLE PROMPT
Return a short cinematic prompt similar to:

"a concept art of an icy lake in rockies, magnificent, epic, majestic"

Rules for SIMPLE PROMPT:
- Start with "a"
- 8–15 words describing the scene
- Add cinematic descriptors like:
  magnificent, epic, majestic, magical, atmospheric
- No sentences longer than one line

Also support:
top-only and bottom-only.

Output JSON with exactly these keys:

scene_keywords
top_keywords
bottom_keywords
simple_prompt

Rules:
- keywords must be comma separated
- simple_prompt must be a short phrase
- no explanations
"""

def generate_scene_keywords(scene_text: str) -> dict:

    user_prompt = f"""
Scene: {scene_text}

Generate:
1) full scene keywords
2) top-only keywords
3) bottom-only keywords
4) simple cinematic prompt
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.9,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)

    return result



# generate_scene_keywords(
#     "a magical underground cave with light stripes coming from holes above"
# )

# output example 
# {
#  "scene_keywords":
#  "underground, cave, cavern, glowing mushrooms, waterfalls, magical, sparkly particles, dimly lit, fantasy, ancient rock, subterranean, mystical atmosphere, underground river, stalactites, surreal",

#  "top_keywords":
#  "cave ceiling, stalactites, light rays through holes, rocky ceiling, glowing crystals, dim light, magical glow, fantasy cave, underground lighting, cavern roof, mystical atmosphere, dark stone, scattered beams, cave textures",

#  "bottom_keywords":
#  "flat stone ground, simple rocky floor, cave floor texture, flat, simple, damp stone, moss patches, underground terrain, wet rock, small puddles, flat, simple, earthy ground, cavern floor",

#  "simple_prompt":
#  "a magical glowing cave with waterfalls and mushrooms, epic, majestic"
# }