import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # .env dosyasını okur

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

SYSTEM_PROMPT = """
You are a scene keyword generator.

The user will describe a scenery. The description may be vague, poetic, contradictory, or incomplete.

Your task:
- Infer the vibe and expand it into a rich, interesting scene.
- Output ONLY comma-separated keywords.
- NO sentences, NO explanations, NO line breaks, NO markdown.
- Always return ~15 keywords.
- Keywords must describe both the SCENE and its ELEMENTS.

You MUST include:

- Pick keywords from these categories below:
    
    Camera Locations
"underwater", "aerial view", "interior", "exterior", "pov", "street level", "above the clouds", "low earth orbit", "underground"
    
    Locations
"library", "bedroom", "bathroom", "hallway", "corridor", "bridge", "helm", "cockpit", "driver's seat", "street", "road", "forest", "city", "train station", "railway", "greenhouse", "residential street", "dock", "hanger", "landing pad", "ferry", "cave", "observatory", "amusement park", "waterpark", "tunnel", "mine", "tropical", "beach", "desert", "steep slope", "cliff", "ocean", "body of water", "river", "mountain", "space", "underground bunker", "space station"
    
    Skies
"aurora borealis", "cloudy", "overcast sky", "blue sky", "stars"
    
    Time
"sunset", "sunrise" "night", "sunny day", "winter", "twilight", "fall"
    
    Weather
"rain", "raining", "snow", "snowing", "fog", "haze", "smoke", "storm", "stormy", "lightning", "flooded", "arid"
    
    Lighting
"bright", "dark", "dimly lit"
    
    Themes
"futuristic", "cyberpunk", "historical", "messy", "scifi", "minimalism", "minimalistic", "simple", "simplistic", "video game", "surrealism", "surrealistic", "cartoon", "comic", "black and white", "smooth", "ancient", "medieval", "vector art", "abandoned", "horror"

- At least one Camera Location
- At least one Location
- At least one Sky or Time
- At least one Weather or Lighting
- At least one Theme

Special instructions for top-only and bottom-only:
- If the user adds "top-only" after the scenery description, return ~15 comma-separated keywords describing ONLY the **ceiling or sky above the camera**, including lights, objects, or textures visible in that top small piece. Inclue general scene mood but not any detail that is not on the ceiling or the sky. 
- If the user adds "bottom-only", return ~15 comma-separated keywords describing ONLY the **floor or ground below the camera**, repeatedly include 'flat' and 'simple', do not include things that you are not stepping on, for example a good keyword set would be: 'flat snow, snow texture, snowy ground, flat, simple, winter,'
- Keep top-only and bottom-only strictly limited to what would be directly on top or bottom of the user in the center of the world.

Rules:
- Always output keywords only
- Always use commas
- Never ask questions
- Never refuse
- Never break format
"""

def generate_scene_keywords(scene_text: str) -> dict:
    """
    Generates 3 keyword lists from the same scene description:
    1) full scene
    2) top-only
    3) bottom-only
    Returns a dict with keys: "scene", "top", "bottom"
    """
    prompts = {
        "scene": scene_text,
        "top": scene_text + " top-only",
        "bottom": scene_text + " bottom-only"
    }

    results = {}
    for key, prompt in prompts.items():
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9
        )
        results[key] = response.choices[0].message.content.strip()

    return results


# # Example usage:
# scene_description = "a magical underground cave with light stripes coming from holes at the top"
# keywords = generate_scene_keywords(scene_description)
# print("Full Scene:", keywords["scene"])
# print("Top-Only:", keywords["top"])
# print("Bottom-Only:", keywords["bottom"])
