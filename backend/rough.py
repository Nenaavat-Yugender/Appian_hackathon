import argparse
import json
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from pathlib import Path

# === Paths for embeddings and metadata ===
EMB_FILE = "embeddings.npy"
META_FILE = "meta.parquet"
IMG_DIR   = Path("C:/Users/Taher/Downloads/hackathon/fashion-dataset/images")

# === Load CLIP and dataset ===
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

embs = np.load(EMB_FILE)
meta = pd.read_parquet(META_FILE)

# === Utility Functions ===
def embed_query(img: Image.Image) -> np.ndarray:
    img_tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        z = clip_model.encode_image(img_tensor)
        z = z / z.norm(dim=-1, keepdim=True)
    return z.cpu().numpy()[0]

def top_k_similar(query_vec: np.ndarray, k=10):
    sims = embs @ query_vec
    idx = sims.argsort()[::-1][:k]
    return meta.iloc[idx].assign(score=sims[idx])

def crop_box(img: Image.Image, box_norm, img_size):
    ymin, xmin, ymax, xmax = box_norm
    width, height = img_size
    ymin = int(ymin / 1000 * height)
    ymax = int(ymax / 1000 * height)
    xmin = int(xmin / 1000 * width)
    xmax = int(xmax / 1000 * width)
    return img.crop((xmin, ymin, xmax, ymax))

# === Main function ===
def process_fashion_image(image_path: Path, bbox_json_path: Path, k=10):
    img = Image.open(image_path).convert("RGB")
    img_size = img.size

    with open(bbox_json_path, 'r') as f:
        boxes = json.load(f)

    all_results = []

    for i, item in enumerate(boxes):
        cropped = crop_box(img, item["box_2d"], img_size)
        query_vec = embed_query(cropped)
        top10 = top_k_similar(query_vec, k=k)

        result = {
            "item_index": i,
            "label": item["label"],
            "confidence": item["confidence"],
            "box_2d": item["box_2d"],
            "top10": top10[["id", "score"]].to_dict(orient="records")
        }
        all_results.append(result)

        print(f"\nTop {k} matches for item {i} ('{item['label']}', conf {item['confidence']:.2f}):")
        for match in result["top10"]:
            print(f"  - ID: {match['id']} | Score: {match['score']:.4f}")

    return all_results

# === CLI Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fashion image similarity search")
    parser.add_argument("--img", required=True, help="Image filename (relative to IMG_DIR)")
    parser.add_argument("--bbox", required=True, help="Path to JSON file with bounding boxes")
    parser.add_argument("--k", type=int, default=10, help="Top K similar items to return")
    args = parser.parse_args()

    image_path = "pexels-jimmyjimmy-1484807.jpg"
    bbox_path = Path(args.bbox)

    process_fashion_image(image_path, bbox_path, k=args.k)
