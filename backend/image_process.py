import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # Load .env into environment variables


client = genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(model="gemini-2.0-flash")
# Upload the first image
image1_path = "/content/pexels-jimmyjimmy-1484807.jpg"
uploaded_file = client.files.upload(file=image1_path)

# Prepare the second image as inline data
image2_path = "/content/pexels-jimmyjimmy-1484807.jpg"
with open(image2_path, 'rb') as f:
    img2_bytes = f.read()

# Create the prompt with text and multiple images
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        "Detect the all of the prominent fashion items in the image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000 , give the confidence score as well, can u give the np array .",
        uploaded_file

    ]
)
import json
import re
import cv2

# Load image to get dimensions
image_path = "/content/pexels-jimmyjimmy-1484807.jpg"
img = cv2.imread(image_path)
H, W = img.shape[:2]

# Get Gemini output
response_text = response.text.strip()

# Extract the first valid JSON array using regex
match = re.search(r"\[\s*\{.*?\}\s*\]", response_text, re.DOTALL)
if not match:
    raise ValueError("❌ No valid JSON array found in Gemini response!")

json_str = match.group(0)
detected_items = json.loads(json_str)

# Convert normalized box coordinates [ymin, xmin, ymax, xmax] from 0–1000 to image pixels
processed_items = []
for item in detected_items:
    ymin, xmin, ymax, xmax = item["box_2d"]
    x1 = int(xmin / 1000 * W)
    y1 = int(ymin / 1000 * H)
    x2 = int(xmax / 1000 * W)
    y2 = int(ymax / 1000 * H)
    processed_items.append({
        "box_2d": [x1, y1, x2, y2],
        "label": item.get("label", "unknown"),
        "confidence": int(item.get("confidence", 0) * 1000)
    })

# Save to bbox JSON
bbox_json_path = "converted_bboxes.json"
with open(bbox_json_path, "w") as f:
    json.dump(processed_items, f, indent=2)

print(f"[✔] Saved bounding boxes to: {bbox_json_path}")
