import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
def show_dataset_image(image_id):
    ROOT     = Path(r"C:\Users\Taher\Downloads\hackathon\fashion-dataset")
    IMG_DIR  = ROOT / "images"
    """Given an image ID (int), show the image from the dataset."""
    image_path = IMG_DIR / f"{int(image_id)}.jpg"
    if not image_path.exists():
        print(f"Image {image_id} not found.")
        return
    
    img = Image.open(image_path)
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Dataset Image ID: {image_id}")
    plt.show()
