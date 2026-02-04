import requests
import webbrowser
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
UPLOAD_URL = os.getenv("IMGBB_UPLOAD_URL")
URL_TO_LOAD = os.getenv("360_VIEWER_URL")

def upload_and_view(image_path):
    # Make sure path is valid
    image_file = Path(image_path)
    if not image_file.exists():
        print("Image path does not exist!")
        return

    # Read image as binary
    with open(image_file, "rb") as f:
        image_data = f.read()

    # Upload to ImgBB
    response = requests.post(
        UPLOAD_URL,
        params={"key": IMGBB_API_KEY},
        files={"image": image_data}
    )

    if response.status_code != 200:
        print("Upload failed:", response.text)
        return

    # Get URL of uploaded image
    resp_json = response.json()
    if not resp_json.get("success"):
        print("Upload error:", resp_json)
        return

    image_url = resp_json["data"]["url"]
    print("Image uploaded successfully:", image_url)

    # Open the 360 panorama viewer with the uploaded image
    viewer_url = f"{URL_TO_LOAD}{image_url}"
    webbrowser.open(viewer_url)
    print("Opened 360 panorama viewer.")

