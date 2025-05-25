# embed_products.py
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

STYLE_JSON_PATH = "C:/Users/Taher/Downloads/hackathon/fashion-dataset/styles"
META_FILE = "meta.parquet"

model = SentenceTransformer('all-MiniLM-L6-v2')

meta = pd.read_parquet(META_FILE)
embeddings = []
product_ids = []

for _, row in meta.iterrows():
    product_id = int(row['id'])
    json_path = os.path.join(STYLE_JSON_PATH, f"{product_id}.json")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)["data"]

        text = f"{data.get('productDisplayName', '')}, brand: {data.get('brandName', '')}, color: {data.get('baseColour', '')}, price: {data.get('discountedPrice', 0)}"
        emb = model.encode(text)
        embeddings.append(emb)
        product_ids.append(product_id)

    except Exception as e:
        print(f"Skipping {product_id}: {e}")

# Save embeddings and product ids
np.save("product_embeddings.npy", np.vstack(embeddings))
with open("product_ids.json", "w") as f:
    json.dump(product_ids, f)
