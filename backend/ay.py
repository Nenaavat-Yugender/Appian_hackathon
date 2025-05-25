import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path
import torch
import clip
import matplotlib.pyplot as plt

# === Paths ===
EMB_FILE = "embeddings.npy"
META_FILE = "meta.parquet"
IMG_DIR   = Path(r"C:\Users\Taher\Downloads\hackathon\fashion-dataset\images")

# === Load CLIP ===
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# === Load Data ===
embs = np.load(EMB_FILE)                # shape: (N, 512)
meta = pd.read_parquet(META_FILE)       # includes "id" column

# === Core Functions ===

def crop_box(img: Image.Image, box_px):
    """Crop using [x1, y1, x2, y2] format"""
    x1, y1, x2, y2 = map(int, box_px)
    return img.crop((x1, y1, x2, y2))


def embed_query(img: Image.Image) -> np.ndarray:
    img_tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        z = clip_model.encode_image(img_tensor)
        z = z / z.norm(dim=-1, keepdim=True)
    return z.cpu().numpy()[0]

def top_k_similar(query_vec: np.ndarray, k=5):
    sims = embs @ query_vec
    idx = sims.argsort()[::-1][:k]
    return meta.iloc[idx].assign(score=sims[idx])

def visualize_boxes(image_path: str, boxes: list):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box["box_2d"])
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, max(0, y1 - 10)), box["label"], fill="white")
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Detected Items")
    plt.show()


def process_image(image_path: str, box_json: str):
    img = Image.open(image_path).convert("RGB")
    boxes = json.loads(box_json)
    
    # Show boxes for confirmation
    visualize_boxes(image_path, boxes)

    # Process each bbox
    all_results = []
    for i, item in enumerate(boxes):
        cropped = crop_box(img, item["box_2d"])
        query_vec = embed_query(cropped)
        top5 = top_k_similar(query_vec, k=5)

        result = {
            "item_index": i,
            "label": item["label"],
            "confidence": item["confidence"],
            "box_2d": item["box_2d"],
            "top5": top5[["id", "score"]].to_dict(orient="records")
        }
        all_results.append(result)

        print(f"\nTop 5 similar for item {i} - {item['label']} (conf: {item['confidence']:.2f}):")
        for match in result["top5"]:
            print(f"  → ID: {match['id']}, Score: {match['score']:.4f}")

    return all_results
if __name__ == "__main__":
    image_path = "pexels-jimmyjimmy-1484807.jpg"
    
    with open("converted_bboxes.json", "r") as f:
        box_json = f.read()

    results = process_image(image_path, box_json)
