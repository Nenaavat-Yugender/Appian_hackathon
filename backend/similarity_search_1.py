# similarity_search_1.py (FAISS-free version)

import os
import json
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from pathlib import Path
from similarity_search import get_fashion_product_bboxes_from_local, crop_box, embed_query

STYLE_JSON_PATH = "C:/Users/Taher/Downloads/hackathon/fashion-dataset/styles"
EMB_FILE = "embeddings.npy"
META_FILE = "meta.parquet"

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

try:
    embs = np.load(EMB_FILE)
    meta = pd.read_parquet(META_FILE)
except FileNotFoundError:
    print("❌ Error loading embedding or metadata files.")
    exit()

def find_similar_fashion_items_with_preferences(image_path: str, user_preferences: dict, confidence_threshold=0.7, top_k=10):
    img = Image.open(image_path).convert("RGB")
    bboxes = get_fashion_product_bboxes_from_local(image_path, confidence_threshold)
    if not bboxes:
        return [], []

    all_results = []
    for i, bbox in enumerate(bboxes):
        cropped_img = crop_box(img, bbox.box_2d)
        emb = embed_query(cropped_img)
        sims = embs @ emb
        idx = sims.argsort()[::-1]

        filtered_matches = []
        for j in idx:
            row = meta.iloc[j]
            match_id = row["id"]
            score = sims[j]

            try:
                json_path = os.path.join(STYLE_JSON_PATH, f"{match_id}.json")
                with open(json_path, 'r', encoding='utf-8') as f:
                    style_data = json.load(f)
                meta_info = style_data["data"]

                brand = meta_info.get("brandName", "").lower()
                color = meta_info.get("baseColour", "").lower()
                price = meta_info.get("discountedPrice", 0)

                min_price, max_price = user_preferences["price_range"]
                if user_preferences["preferred_brands"] and brand not in user_preferences["preferred_brands"]:
                    continue
                if user_preferences["preferred_colors"] and color not in user_preferences["preferred_colors"]:
                    continue
                if min_price is not None and price < min_price:
                    continue
                if max_price is not None and price > max_price:
                    continue

                filtered_matches.append({
                    "bbox_index": i,
                    "bbox_label": bbox.label,
                    "bbox_confidence": bbox.confidence,
                    "match_id": match_id,
                    "match_score": score
                })

                if len(filtered_matches) >= top_k:
                    break

            except Exception as e:
                print(f"⚠️ Error with {match_id}: {e}")
                continue

        all_results.extend(filtered_matches)

    all_results_sorted = sorted(all_results, key=lambda x: x["match_score"], reverse=True)[:top_k]
    return bboxes, all_results_sorted
