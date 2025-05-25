# main.py (Gemini-only vision recommender)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import json
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()  # Load .env into environment variables

app_0 = FastAPI()

app_0.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Gemini API

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")


class Prompt(BaseModel):
    prompt: str

@app_0.post("/gemini_recommendations")
def gemini_recommendations(image: UploadFile = File(...)):
    try:
        contents = image.file.read()
        image_pil = Image.open(BytesIO(contents)).convert("RGB")
        buffer = BytesIO()
        image_pil.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        vision_prompt = """
You are a fashion stylist. The user uploaded an image.
1. Detect and describe the fashion items visible.
2. Recommend 3-5 visually similar products (with direct image URLs only — avoid homepage/product page links).
3. Suggest complementary items (e.g., shoes for pants, earrings for dresses).
4. Format your response ONLY in this JSON format:
{
  "detected_items": ["white dress", "black heels"],
  "similar_products": [
    {"name": "Zara White Midi Dress", "price": 2499, "image_url": "https://..."},
    {"name": "H&M Off-Shoulder Dress", "price": 2999, "image_url": "https://..."}
  ],
  "complementary_products": [
    {"name": "Tan Sandals", "price": 1999, "image_url": "https://..."},
    {"name": "Gold Earrings", "price": 899, "image_url": "https://..."}
  ]
}
ONLY include direct image URLs (ending in .jpg, .jpeg, .png, or .webp). Avoid product pages or shop links.
        """

        response = model.generate_content(
            [vision_prompt.strip(), {"mime_type": "image/jpeg", "data": image_bytes}],
            stream=False
        )

        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw_text)
        return parsed

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process image with Gemini: {e}")

@app_0.post("/prompt")
def handle_prompt(prompt: Prompt):
    try:
        user_prompt = prompt.prompt
        instruction = "You are a fashion chatbot assistant. Based on the user input, update filters, add items to cart, or simulate checkout if asked. Respond only in JSON."
        full_prompt = f"""
{instruction}
User prompt: "{user_prompt}"
Respond in JSON like:
{{
  "action": "set_preferences" / "add_to_cart" / "checkout",
  "details": {{ ... }}
}}
        """
        response = model.generate_content(full_prompt.strip())
        raw = response.text.strip()
        if raw.startswith("```json"):
            raw = raw.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw)
        return parsed

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prompt failed: {e}")
