import requests
from PIL import Image
import io

API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
headers = {"Authorization": "Bearer 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"}

class ImageGenerator:
    def __init__(self, api_url=API_URL, headers=headers):
        self.api_url = api_url
        self.headers = headers

    def query(self, payload):
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")
            return None

    def save_image_to_file(self, image_bytes, filename):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.save(filename)
            print(f"Image saved to {filename}")
        except Exception as e:
            print(f"Error saving image to file: {e}")
