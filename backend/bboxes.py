import os
import json
import re
import base64
import google.generativeai as genai
from typing import List
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()  # Load .env into environment variables

# Configure Gemini (secure this key in production)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")

# Data class to hold box info
class BoundingBox:
    def __init__(self, box_2d: list[int], label: str, confidence: float = 1.0):
        self.box_2d = box_2d
        self.label = label
        self.confidence = confidence

    def __repr__(self):
        return f"BoundingBox(label='{self.label}', confidence={self.confidence}, box_2d={self.box_2d})"


# Encode image as bytes
def encode_image_to_bytes(image_path: str) -> bytes:
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.convert("RGB").save(buffered, format="JPEG")
        return buffered.getvalue()


# Extract fashion product bounding boxes using Gemini
def get_fashion_product_bboxes_from_local(image_path: str, confidence_threshold=0.7) -> List[BoundingBox]:
    img_bytes = encode_image_to_bytes(image_path)

    prompt = """
You are a vision model that extracts bounding boxes for fashion products in an image. 
Respond ONLY in JSON format, like this:

{
  "bounding_boxes": [
    {"label": "t-shirt", "confidence": 0.92, "box_2d": [x_min, y_min, x_max, y_max]},
    {"label": "jeans", "confidence": 0.88, "box_2d": [x_min, y_min, x_max, y_max]}
  ]
}
Make sure coordinates are pixel values (not normalized).
"""

    try:
        response = model.generate_content(
            [
                prompt.strip(),
                {"mime_type": "image/jpeg", "data": img_bytes}
            ],
            stream=False,
        )

        raw_text = response.text.strip()

        # Remove markdown code block formatting
        if raw_text.startswith("```json"):
            raw_text = re.sub(r"```json|```", "", raw_text).strip()

        parsed = json.loads(raw_text)
        items = parsed.get("bounding_boxes", [])

    except Exception as e:
        print("❌ Error parsing Gemini response:", e)
        print("↩️ Raw Gemini output:\n", response.text.strip())
        return []

    results = []
    for item in items:
        try:
            label = item["label"]
            confidence = float(item.get("confidence", 1.0))
            box = item["box_2d"]

            if confidence >= confidence_threshold and isinstance(box, list) and len(box) == 4:
                results.append(BoundingBox(box_2d=box, label=label, confidence=confidence))
        except Exception as e:
            print("⚠️ Skipping invalid item:", item, "| Error:", e)

    if not results:
        print("⚠️ No bounding boxes meet the confidence threshold.")

    return results


# === Example usage ===
if __name__ == "__main__":
    image_path = "pexels-jimmyjimmy-1484807.jpg"
    bboxes = get_fashion_product_bboxes_from_local(image_path, confidence_threshold=0.7)

    print("\n🎯 Bounding boxes above threshold:")
    for box in bboxes:
        print(box)
