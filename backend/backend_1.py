# main.py (updated with JSON serialization fixes for FastAPI)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import base64
import io
import json
import os
import pandas as pd
import google.generativeai as genai
from similarity_search_1 import find_similar_fashion_items_with_preferences

app_1 = FastAPI()

app_1.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STYLE_JSON_PATH = "C:/Users/Taher/Downloads/hackathon/fashion-dataset/styles"
META_FILE = "meta.parquet"

# Load metadata once
meta = pd.read_parquet(META_FILE)

# Temporary in-memory preference store (to be replaced with session-based storage later)
user_preferences = {
    "preferred_colors": set(),
    "preferred_brands": set(),
    "price_range": (None, None),
}

# Setup Gemini
genai.configure(api_key="AIzaSyCYXaIedN0JEmOYjStfdHe5VZdhXOx4y0E")
model = genai.GenerativeModel("gemini-2.0-flash")

class ImageRequest(BaseModel):
    image: str  # base64 string

class PreferenceRequest(BaseModel):
    colors: list[str] = []
    brands: list[str] = []
    min_price: float | None = None
    max_price: float | None = None

class Prompt(BaseModel):
    prompt: str

def get_image_url_from_json(image_id: int) -> str:
    try:
        json_path = os.path.join(STYLE_JSON_PATH, f"{image_id}.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data["data"]["styleImages"]["default"]["imageURL"]
    except Exception as e:
        print(f"⚠️ Error reading image URL for ID {image_id}:", e)
        return "URL_NOT_FOUND"

@app_1.post("/analyze")
def analyze_fashion_image(request: ImageRequest):
    try:
        image_bytes = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_path = "temp_input.jpg"
        image.save(image_path)

        bboxes, top_k_results = find_similar_fashion_items_with_preferences(
            image_path, user_preferences
        )

        results_by_box = {}
        for r in top_k_results:
            box_id = int(r['bbox_index'])
            match_id = int(r['match_id'])
            match_score = float(r['match_score'])
            image_url = get_image_url_from_json(match_id)

            if box_id not in results_by_box:
                results_by_box[box_id] = {
                    'label': str(r['bbox_label']),
                    'confidence': float(r['bbox_confidence']),
                    'matches': []
                }
            results_by_box[box_id]['matches'].append({
                'id': match_id,
                'score': match_score,
                'url': image_url
            })

        return results_by_box

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app_1.post("/set_preferences")
def set_user_preferences(pref: PreferenceRequest):
    if pref.colors:
        user_preferences["preferred_colors"].update(map(str.lower, pref.colors))
    if pref.brands:
        user_preferences["preferred_brands"].update(map(str.lower, pref.brands))
    if pref.min_price or pref.max_price:
        user_preferences["price_range"] = (pref.min_price, pref.max_price)
    return {"message": "Preferences updated", "preferences": user_preferences}

@app_1.post("/modify")
def modify_results(prompt: Prompt):
    try:
        gemini_prompt = f"""
You're a fashion assistant. Extract structured filter info (colors, brands, price_range) from the following prompt:

Prompt: "{prompt.prompt}"

Respond ONLY in valid JSON:
{{
  "brands": ["Nike"],
  "colors": ["black"],
  "price_range": [null, 2500]
}}
        """

        response = model.generate_content(gemini_prompt.strip())
        raw = response.text.strip()
        if raw.startswith("```json"):
            raw = raw.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw)
        print("🔍 Gemini parsed preferences:", parsed)

        if "brands" in parsed:
            user_preferences["preferred_brands"].update(map(str.lower, parsed["brands"]))
        if "colors" in parsed:
            user_preferences["preferred_colors"].update(map(str.lower, parsed["colors"]))
        if "price_range" in parsed:
            min_price, max_price = parsed["price_range"]
            user_preferences["price_range"] = (min_price, max_price)

        if not os.path.exists("temp_input.jpg"):
            raise HTTPException(status_code=400, detail="No previously uploaded image found to analyze")

        bboxes, top_k_results = find_similar_fashion_items_with_preferences(
            "temp_input.jpg", user_preferences
        )

        if not top_k_results:
            return {
                "message": "Preferences updated but no matching products found.",
                "preferences": user_preferences,
                "results": {}
            }

        results_by_box = {}
        for r in top_k_results:
            box_id = int(r['bbox_index'])
            match_id = int(r['match_id'])
            match_score = float(r['match_score'])
            image_url = get_image_url_from_json(match_id)

            if box_id not in results_by_box:
                results_by_box[box_id] = {
                    'label': str(r['bbox_label']),
                    'confidence': float(r['bbox_confidence']),
                    'matches': []
                }
            results_by_box[box_id]['matches'].append({
                'id': match_id,
                'score': match_score,
                'url': image_url
            })

        return {
            "message": "Preferences updated from prompt",
            "preferences": user_preferences,
            "results": results_by_box
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to parse prompt: {e}")

@app_1.post("/reset_preferences")
def reset_preferences():
    user_preferences["preferred_colors"].clear()
    user_preferences["preferred_brands"].clear()
    user_preferences["price_range"] = (None, None)
    return {"message": "Preferences reset"}
