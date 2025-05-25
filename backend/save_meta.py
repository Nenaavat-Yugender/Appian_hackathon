from pathlib import Path
import pandas as pd

ROOT     = Path(r"C:\Users\Taher\Downloads\hackathon\fashion-dataset")
CSV_PATH = ROOT / "styles.csv"
IMG_DIR  = ROOT / "images"

# rebuild the dataframe (quick—just reads CSV, no embeddings)
df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
df["filepath"] = df["id"].apply(lambda x: IMG_DIR / f"{int(x)}.jpg")

# drop filepath column and save as parquet
df.drop(columns=["filepath"]).to_parquet("meta.parquet")
print("✔  metadata saved to meta.parquet")
