# 👗 Personalized Fashion Chatbot

This project is an intelligent fashion recommendation system that enables users to upload an image and get visually similar fashion items from a dataset. It leverages OpenAI's CLIP model and FAISS for similarity search and sets the foundation for incorporating user preferences via natural language prompts.

---

## 🎯 Project Objective

To build a web-based personalized fashion assistant that:
- Accepts user-uploaded images.
- Returns visually similar fashion products from a dataset.
- (Planned) Accepts natural language prompts to personalize the recommendations.

---

## ✅ Implemented Features

### 🖼 Image Upload + CLIP Embedding
- Users can upload a fashion image through a web interface.
- The backend uses CLIP (ViT-B/32) to extract visual embeddings from the image.

### 🔍 Similarity Search with FAISS
- All dataset images are pre-embedded and indexed using FAISS.
- Cosine similarity is used to retrieve top-K visually similar items.

### 🧩 Object Detection (Bounding Boxes)
- Gemini Vision API is used to extract bounding boxes of fashion items in the uploaded image.
- This helps focus on relevant parts of the image for embedding.

### 🖥 Frontend UI
- HTML/CSS/JS-based frontend for image upload and result display.
- Grid layout to show similar images along with metadata (title, price, brand).

---

## 🚧 Work in Progress

### 💬 Prompt-Based Personalization (Planned)
A strategy was designed but not implemented due to time constraints:
- Embed user prompt using Sentence Transformers.
- Combine prompt + image embeddings for enhanced search.
- Use Gemini to interpret prompts and extract metadata filters (e.g., "under ₹500", "Korean fashion").

---

## 🧠 Tech Stack

| Layer       | Technology         |
|-------------|--------------------|
| Embeddings  | OpenAI CLIP (ViT-B/32) |
| Search      | FAISS (cosine similarity) |
| Prompt NLP  | Sentence Transformers (planned) |
| Frontend    | HTML, CSS, JavaScript |
| Backend     | Python (FastAPI) |
| Metadata    | JSON files per dataset image |

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Nenaavat-Yugender/Appian_hackathon


---

## 📦 Python Dependencies

### Make sure you have Python 3.9+ installed. Install the dependencies using:
```bash
pip install -r requirements.txt

---

```txt
# Core Python server
fastapi==0.110.0
uvicorn==0.27.1

# CLIP and Vision Models
git+https://github.com/openai/CLIP.git
torch==2.2.0
torchvision==0.17.0

# Image processing
Pillow==10.3.0

# FAISS for similarity search
faiss-cpu==1.7.4

# Sentence Embeddings (for prompt handling)
sentence-transformers==2.6.1

# CORS (for frontend-backend communication)
python-multipart==0.0.9
aiofiles==23.2.1
jinja2==3.1.3
requests==2.31.0
