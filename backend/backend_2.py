# main.py (updated basic backend with Gemini image context + robust embedding search)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import base64
import io
import json
import os
import numpy as np
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app_2 = FastAPI()

app_2.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STYLE_JSON_PATH = "C:/Users/Taher/Downloads/hackathon/fashion-dataset/styles"
META_FILE = "meta.parquet"

# Load metadata and models
meta = pd.read_parquet(META_FILE)
genai.configure(api_key="AIzaSyCYXaIedN0JEmOYjStfdHe5VZdhXOx4y0E")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

class ImageRequest(BaseModel):
    image: str  # base64 string

session = { "image_embedding": None }

@app_2.post("/analyze")
def analyze_fashion_image(request: ImageRequest):
    try:
        image_bytes = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_data = buffer.getvalue()

        gemini_prompt = "Describe the fashion products shown in this image. Mention item types, style, material, brand if visible."
        response = gemini_model.generate_content([
            gemini_prompt,
            {"mime_type": "image/jpeg", "data": img_data}
        ])

        print("Gemini response:", response.text)
        description = response.text.strip()
        session["image_embedding"] = sbert_model.encode(description)

        return {"description": description}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app_2.post("/modify")
def modify_results(prompt: BaseModel):
    try:
        prompt_emb = sbert_model.encode(prompt.prompt)
        image_emb = session.get("image_embedding")
        fused_emb = prompt_emb if image_emb is None else (prompt_emb + image_emb) / 2

        product_embs = np.load("product_embeddings.npy")
        with open("product_ids.json", "r") as f:
            product_ids = json.load(f)

        sims = cosine_similarity([fused_emb], product_embs)[0]
        top_indices = sims.argsort()[::-1][:5]

        results = []
        for idx in top_indices:
            pid = product_ids[idx]
            json_path = os.path.join(STYLE_JSON_PATH, f"{pid}.json")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)["data"]
            results.append({
                "name": data.get("productDisplayName", ""),
                "brand": data.get("brandName", ""),
                "color": data.get("baseColour", ""),
                "price": data.get("discountedPrice", 0),
                "url": data.get("styleImages", {}).get("default", {}).get("imageURL", "URL_NOT_FOUND")
            })

        return {
            "message": "Top matches from image + prompt fusion",
            "query": prompt.prompt,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
