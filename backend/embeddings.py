from pathlib import Path
import pandas as pd

ROOT     = Path(r"C:\Users\Taher\Downloads\hackathon\fashion-dataset")
CSV_PATH = ROOT / "styles.csv"
IMG_DIR  = ROOT / "images"           # ← now a Path, so “/” works

df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

df["filepath"] = df["id"].apply(lambda x: IMG_DIR / f"{int(x)}.jpg")

print(df.head()[["id", "filepath"]])
"""
embed_clip.py  –  create 512‑D CLIP image embeddings
====================================================

Usage
-----
$ python embed_clip.py                     # default paths
$ python embed_clip.py --root "D:/fashion-dataset" --batch 128
"""

import argparse, time, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
import torch
import clip
from tqdm import tqdm

# ----------------------------------------------------------------------
# Parse CLI arguments
# ----------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--root",  type=str, default=r"C:\Users\Taher\Downloads\hackathon\fashion-dataset",
                help="folder that contains styles.csv and the images/ directory")
ap.add_argument("--csv",   type=str, default="styles.csv", help="csv file name inside root")
ap.add_argument("--imgdir",type=str, default="images",    help="image directory inside root")
ap.add_argument("--batch", type=int, default=64,          help="batch size")
args = ap.parse_args()

ROOT     = Path(args.root)
CSV_PATH = ROOT / args.csv
IMG_DIR  = ROOT / args.imgdir
BATCH    = args.batch
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading metadata from {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

# Build absolute filepath based on `id` → "<id>.jpg"
df["filepath"] = df["id"].apply(lambda x: IMG_DIR / f"{int(x)}.jpg")
print(f"Total rows: {len(df):,}")

# ----------------------------------------------------------------------
# Load CLIP model
# ----------------------------------------------------------------------
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

def embed_batch(paths):
    """Return L2‑normalized 512‑D vectors for a list of image paths."""
    imgs = []
    for p in paths:
        try:
            imgs.append(preprocess(Image.open(p).convert("RGB")))
        except Exception as e:
            # handle unreadable images (rare)
            imgs.append(torch.zeros(3,224,224))
    imgs = torch.stack(imgs).to(DEVICE)
    with torch.no_grad():
        z = clip_model.encode_image(imgs)
        z = z / z.norm(dim=-1, keepdim=True)
    return z.cpu().numpy()

# ----------------------------------------------------------------------
# Iterate in batches and collect embeddings
# ----------------------------------------------------------------------
vecs = np.zeros((len(df), 512), dtype="float32")
start = time.time()
for i in tqdm(range(0, len(df), BATCH)):
    batch_paths = df["filepath"].iloc[i:i+BATCH].tolist()
    vecs[i:i+BATCH] = embed_batch(batch_paths)

print(f"Finished in {time.time()-start:.1f}s")

# ----------------------------------------------------------------------
# Save outputs
# ----------------------------------------------------------------------
np.save("embeddings.npy", vecs)
df.drop(columns=["filepath"]).to_parquet("meta.parquet")   # store the metadata sans path

print("Saved  ➜ embeddings.npy  (", vecs.shape, ")")
print("        ➜ meta.parquet   (pandas dataframe)")
