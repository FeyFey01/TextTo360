from PIL import Image
from pathlib import Path
from typing import Optional

def create_center_alpha_mask(
    image_path: str, 
    strip_width: int = 60, 
    output_path: Optional[str] = None
) -> Path:
    """
    Creates a mask and saves it to disk.
    Returns: The Path object where the mask was saved.
    """
    # 1. Load Image
    img = Image.open(image_path).convert("RGBA")
    image_width, image_height = img.size

    # 2. Create Mask logic
    mask = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
    left = (image_width - strip_width) // 2
    right = left + strip_width
    
    center_strip = img.crop((left, 0, right, image_height))
    mask.paste(center_strip, (left, 0))

    # 3. Handle Saving Logic
    if output_path:
        save_path = Path(output_path)
    else:
        # Default: Same folder as input, adds '_mask' suffix
        input_p = Path(image_path)
        save_path = input_p.parent / f"{input_p.stem}_mask.png"

    mask.save(save_path)
    print(f"Mask successfully saved at: {save_path}")

    # 4. Return the Path object (so Path(mask_img) in your other file works)
    return save_path

# Example usage in your run_me.py:
# mask_img_path = create_center_alpha_mask(image_path="photo.jpg")