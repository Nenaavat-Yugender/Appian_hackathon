"""
detect_and_search.py
--------------------
Detect clothing items in a photo ➜ CLIP‑embed each crop ➜ FAISS k‑NN search
"""

import argparse, numpy as np, pandas as pd, faiss, torch, clip, cv2
from pathlib import Path
from PIL import Image

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

ROOT     = Path(r"C:\Users\Taher\Downloads\hackathon\fashion-dataset")

ap = argparse.ArgumentParser()
ap.add_argument("--img", required=True, help="input image")
ap.add_argument("--k",   type=int, default=6, help="similar results per object")
ap.add_argument("--emb", type=str, default="embeddings.npy")
ap.add_argument("--meta",type=str, default="meta.parquet")
ap.add_argument("--model",type=str, default="yolov8m.pt",
                help="Ultralytics YOLOv8 model (.pt or .onnx)")
args = ap.parse_args()

QUERY_IMG = Path(args.img)
TOP_K     = args.k

# ------------------------------------------------------------
# 1.  Load YOLOv8
# ------------------------------------------------------------
from ultralytics import YOLO

yolo = YOLO(args.model)   # m‑size is usually good (~38 MB)


import openai, os, functools, json, time

openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("Set OPENAI_API_KEY environment variable to use OpenAI API.")

@functools.lru_cache(maxsize=256)
def is_fashion_item(label: str) -> bool:
    """
    Return True if GPT‑4 thinks this label is a fashion item.
    Simple yes/no answer to minimise cost and latency.
    """
    prompt = (
        "You are a helpful assistant. Reply only 'yes' or 'no'.\n"
        "Question: Is the object class name below an item of clothing, "
        "footwear or fashion accessory?\n\n"
        f"Label: {label.lower()}\nAnswer:"
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",          # cheapest GPT‑4 tier; adjust if needed
            messages=[{"role":"user","content": prompt}],
            temperature=0.0,
            max_tokens=1,
        )
        answer = resp.choices[0].message.content.strip().lower()
        return answer == "yes"
    except Exception as e:
        # fallback: assume not fashion on API failure
        print("OpenAI error:", e)
        return False


# ------------------------------------------------------------
# 2.  Run detection
# ------------------------------------------------------------
orig = cv2.imread(str(QUERY_IMG))[:, :, ::-1]  # BGR→RGB
det  = yolo.predict(orig, verbose=False, imgsz=640)[0]  # first image

kept = []
kept = []
for box, conf, cid in zip(det.boxes.xyxy.cpu(), det.boxes.conf.cpu(), det.boxes.cls.cpu()):
    cls_name = yolo.model.names[int(cid)]
    if conf < 0.25:
        continue                                 # skip low‑confidence boxes
    if not is_fashion_item(cls_name):
        continue                                 # GPT said “no”
    x1,y1,x2,y2 = map(int, box)
    kept.append((cls_name, conf.item(), orig[y1:y2, x1:x2]))



print(f"Detected {len(kept)} relevant objects.")

# ------------------------------------------------------------
# 3.  Load FAISS / embeddings / metadata
# ------------------------------------------------------------
embs  = np.load(args.emb).astype("float32")
meta  = pd.read_parquet(args.meta)
d     = embs.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embs)

device =  "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

@torch.inference_mode()
def clip_embed(arr):
    img = Image.fromarray(arr)
    ten = preprocess(img).unsqueeze(0).to(device)
    z   = clip_model.encode_image(ten)
    z   = z / z.norm(dim=-1, keepdim=True)
    return z.cpu().numpy().astype("float32")[0]

# ------------------------------------------------------------
# 4.  For each crop → CLIP → FAISS search
# ------------------------------------------------------------
results_all = []
for idx, (cls_name, conf, crop) in enumerate(kept, 1):
    qvec = clip_embed(crop)[None, :]
    D,I  = index.search(qvec, TOP_K)
    rdf  = meta.iloc[I[0]].copy()
    rdf["score"] = D[0]
    rdf["det_class"] = cls_name
    rdf["det_conf"]  = conf
    results_all.append(rdf)

# ------------------------------------------------------------
# 5.  Display / print
# ------------------------------------------------------------
for n, rdf in enumerate(results_all, 1):
    print(f"\n=== Object {n}  ({rdf.det_class.iloc[0]}  conf {rdf.det_conf.iloc[0]:.2f}) ===")
    print(rdf[["id","articleType","score"]])

# optional thumbnail show
try:
    from IPython.display import display, HTML
    IMG_DIR = Path(r"C:\Users\Taher\Downloads\hackathon\fashion-dataset\images")
    for n,rdf in enumerate(results_all,1):
        html = [f"<h3>Object {n}: {rdf.det_class.iloc[0]}</h3><table>"]
        for _, row in rdf.iterrows():
            p = IMG_DIR / f"{int(row.id)}.jpg"
            html.append(
                f"<tr><td><img src='{p.as_posix()}' width='120'></td>"
                f"<td>{row.articleType} | score {row.score:.3f}</td></tr>")
        html.append("</table>")
        display(HTML("".join(html)))
except Exception:
    pass
