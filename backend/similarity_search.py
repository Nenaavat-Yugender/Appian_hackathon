import os
import json
import re
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from pathlib import Path
import google.generativeai as genai
import cv2 # Ensure cv2 is imported for the drawing function
from dotenv import load_dotenv

load_dotenv()  # Load .env into environment variables

# === Setup Gemini ===
# It's highly recommended to use environment variables for API keys in production
# genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")

# === Load CLIP model and fashion dataset embeddings/meta ===
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

EMB_FILE = "embeddings.npy"
META_FILE = "meta.parquet"

# Ensure these files exist or create dummy ones for testing if not
# For demonstration purposes, you might need to create these or comment out if not available.
# Example dummy creation for testing:
# if not Path(EMB_FILE).exists():
#     np.save(EMB_FILE, np.random.rand(100, 512)) # 100 dummy embeddings of dim 512
# if not Path(META_FILE).exists():
#     pd.DataFrame({'id': range(100), 'path': [f'dummy_path_{i}.jpg' for i in range(100)]}).to_parquet(META_FILE)

try:
    embs = np.load(EMB_FILE)  # shape: (num_images, emb_dim)
    meta = pd.read_parquet(META_FILE)
except FileNotFoundError:
    print(f"Error: {EMB_FILE} or {META_FILE} not found. Please ensure they are in the same directory.")
    print("For testing, you might need to generate dummy files or adjust the paths.")
    # Exit or handle gracefully if critical files are missing
    exit()

# === Helper Classes ===
class BoundingBox:
    def __init__(self, box_2d: list[int], label: str, confidence: float):
        # box_2d is expected to be [x_min, y_min, x_max, y_max]
        self.box_2d = box_2d
        self.label = label
        self.confidence = confidence

    def __repr__(self):
        return f"BoundingBox(label='{self.label}', confidence={self.confidence}, box_2d={self.box_2d})"

# === Gemini: Get bounding boxes for fashion products ===
def encode_image_to_bytes(image_path: str) -> bytes:
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.convert("RGB").save(buffered, format="JPEG")
        return buffered.getvalue()

def get_fashion_product_bboxes_from_local(image_path: str, confidence_threshold=0.7):
    img_bytes = encode_image_to_bytes(image_path)

    # Get original image dimensions for scaling
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    # Adjusted prompt to explicitly state coordinate order as per Gemini's usual output
    prompt = """
You are a vision model that extracts bounding boxes for fashion products in an image.
Respond ONLY in JSON format, like this:

{
  "bounding_boxes": [
    {"label": "t-shirt", "confidence": 0.92, "box_2d": [y_min, x_min, y_max, x_max]},
    {"label": "jeans", "confidence": 0.88, "box_2d": [y_min, x_min, y_max, x_max]}
  ]
}
Make sure coordinates are pixel values normalized 0-1000, and the order is [y_min, x_min, y_max, x_max].
"""

    try:
        response = model.generate_content(
            [
                prompt.strip(),
                {"mime_type": "image/jpeg", "data": img_bytes}
            ],
            stream=False,
        )

        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = re.sub(r"```json|```", "", raw_text).strip()

        parsed = json.loads(raw_text)
        items = parsed.get("bounding_boxes", [])

    except Exception as e:
        print("❌ Error parsing Gemini response:", e)
        print("↩️ Raw Gemini output:\n", response.text.strip())
        return []

    results = []
    for item in items:
        try:
            label = item["label"]
            confidence = float(item.get("confidence", 1.0))
            normalized_box = item["box_2d"] # This is expected to be [y_min, x_min, y_max, x_max]

            if confidence >= confidence_threshold and isinstance(normalized_box, list) and len(normalized_box) == 4:
                # Scale normalized coordinates (0-1000) to actual pixel values
                # Correctly mapping [y_min, x_min, y_max, x_max] to [x_min, y_min, x_max, y_max]
                y_min_norm, x_min_norm, y_max_norm, x_max_norm = normalized_box

                actual_box = [
                    max(0, min(int(x_min_norm / 1000 * img_width), img_width - 1)),  # x_min
                    max(0, min(int(y_min_norm / 1000 * img_height), img_height - 1)), # y_min
                    max(0, min(int(x_max_norm / 1000 * img_width), img_width - 1)),  # x_max
                    max(0, min(int(y_max_norm / 1000 * img_height), img_height - 1)), # y_max
                ]

                results.append(BoundingBox(box_2d=actual_box, label=label, confidence=confidence))
        except Exception as e:
            print("⚠️ Skipping invalid item:", item, "| Error:", e)

    if not results:
        print("⚠️ No bounding boxes meet the confidence threshold.")

    return results

# === Crop image using pixel coords from bounding box ===
def crop_box(img: Image.Image, box_px):
    # Consistent box_px format: [x_min, y_min, x_max, y_max]
    x_min, y_min, x_max, y_max = box_px
    
    # Ensure coordinates are integers for PIL.crop
    return img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

# === Embed cropped image using CLIP ===
def embed_query(img: Image.Image) -> np.ndarray:
    img_tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        z = clip_model.encode_image(img_tensor)
        z = z / z.norm(dim=-1, keepdim=True)
    return z.cpu().numpy()[0]

# === Get top k similar images from dataset given query embedding ===
def top_k_similar(query_vec: np.ndarray, k=10):
    sims = embs @ query_vec  # dot product similarity
    idx = sims.argsort()[::-1][:k]
    return meta.iloc[idx].assign(score=sims[idx])

# === Visualization: Draw bounding boxes and labels ===
# This function seems robust and already expects [x_min, y_min, x_max, y_max]
def draw_bounding_boxes_cv2(image_path, bboxes, top_k_results, save_path=None):
    # Load image with OpenCV (BGR format)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    # Get image dimensions
    img_height, img_width = img.shape[:2]

    for bbox in bboxes:
        # Bounding box format is [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = bbox.box_2d
        label = bbox.label
        confidence = getattr(bbox, 'confidence', None)

        # Draw rectangle (BGR color, thickness 2)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Prepare label text
        if confidence is not None:
            text = f"{label} ({confidence:.2f})"
        else:
            text = label

        # Get text size for background box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw filled rectangle for text background
        # Ensure the background rectangle doesn't go out of bounds upwards
        rect_y_start = max(0, y_min - text_height - baseline - 2) # Added a small padding
        cv2.rectangle(img, (x_min, rect_y_start), (x_min + text_width + 2, y_min), (0, 255, 0), thickness=cv2.FILLED)

        # Put text above bounding box
        text_y_pos = max(text_height + baseline, y_min - baseline) # Ensure text is visible
        cv2.putText(img, text, (x_min, text_y_pos), font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)

    # Save or display
    if save_path:
        cv2.imwrite(save_path, img)
        print(f"Image saved with bounding boxes to: {save_path}")
    else:
        cv2.imshow("Image with Bounding Boxes", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# === Main integrated function ===
def find_similar_fashion_items(image_path: str, confidence_threshold=0.7, top_k=10):
    # Load full image once
    img = Image.open(image_path).convert("RGB")

    # Get bounding boxes with Gemini
    bboxes = get_fashion_product_bboxes_from_local(image_path, confidence_threshold)
    if not bboxes:
        print("No bounding boxes found above threshold.")
        return [], []

    all_results = []
    for i, bbox in enumerate(bboxes):
        cropped_img = crop_box(img, bbox.box_2d)
        emb = embed_query(cropped_img)
        top_matches = top_k_similar(emb, k=top_k)

        # Add bbox info to each match for traceability
        for _, row in top_matches.iterrows():
            all_results.append({
                "bbox_index": i,
                "bbox_label": bbox.label,
                "bbox_confidence": bbox.confidence,
                "match_id": row["id"],
                "match_score": row["score"]
            })

    # Sort ALL matches from all boxes by similarity score, pick top_k overall
    all_results_sorted = sorted(all_results, key=lambda x: x["match_score"], reverse=True)[:top_k]

    return bboxes, all_results_sorted

# === Example usage ===
if __name__ == "__main__":
    image_path = "shirt_1.jpg" # Make sure this image exists
    confidence_threshold = 0.7
    top_k = 10

    # Create a dummy image for testing if the actual image is not present
    if not Path(image_path).exists():
        print(f"Creating a dummy image for testing: {image_path}")
        dummy_img = Image.new('RGB', (800, 600), color = 'red')
        draw = ImageDraw.Draw(dummy_img)
        font = ImageFont.load_default() # Use default font
        draw.text((10, 10), "Dummy Image for Testing", fill=(255,255,255), font=font)
        dummy_img.save(image_path)
        # You might need to adjust expected labels in the prompt for dummy images
        # or mock the Gemini response for a full test.

    bboxes, top_k_results = find_similar_fashion_items(image_path, confidence_threshold, top_k)

    if bboxes: # Only print and draw if bounding boxes were found
        print("\nTop similar fashion items across detected objects:\n")
        for r in top_k_results:
            print(
                f"Box #{r['bbox_index']} ({r['bbox_label']}, conf={r['bbox_confidence']:.2f}) -> "
                f"Match ID: {r['match_id']}, Similarity: {r['match_score']:.4f}"
            )

        # Visualize results with bounding boxes + top matches
        draw_bounding_boxes_cv2(image_path, bboxes, top_k_results, save_path="annotated_output.jpg")
    else:
        print("No bounding boxes were detected for visualization.")