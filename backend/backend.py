# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import base64
import io
import json
import os
import pandas as pd
from similarity_search import find_similar_fashion_items  # Replace with your filename without .py

app = FastAPI()


app.add_middleware(
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

class ImageRequest(BaseModel):
    image: str  # base64 string

def get_image_url_from_json(image_id: int) -> str:
    try:
        json_path = os.path.join(STYLE_JSON_PATH, f"{image_id}.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data["data"]["styleImages"]["default"]["imageURL"]
    except Exception as e:
        print(f"⚠️ Error reading image URL for ID {image_id}:", e)
        return "URL_NOT_FOUND"

@app.post("/analyze")
def analyze_fashion_image(request: ImageRequest):
    try:
        image_bytes = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_path = "temp_input.jpg"
        image.save(image_path)

        bboxes, top_k_results = find_similar_fashion_items(image_path)

        results_by_box = {}
        for r in top_k_results:
            box_id = r['bbox_index']
            match_id = r['match_id']
            match_score = r['match_score']
            image_url = get_image_url_from_json(match_id)

            if box_id not in results_by_box:
                results_by_box[box_id] = {
                    'label': r['bbox_label'],
                    'confidence': r['bbox_confidence'],
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
