import os
import csv
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# --- Device setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --- Load BLIP model ---
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# --- Caption generation function ---
def generate_caption(image_path):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(img, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_length=50,
        num_beams=3,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# --- Folder and output CSV ---
image_folder = "/home/aayush_shah/projects/imageCaptioning/downloads/images/"  # Replace with your folder path
output_csv = "captionsPDfFsTemp.csv"

results = []

# --- Process each image in folder ---
for i,filename in enumerate(os.listdir(image_folder)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        full_path = os.path.join(image_folder, filename)
        try:
            caption = generate_caption(full_path)
            # print(f"{filename}: {caption}")
            results.append([filename, caption])
        except Exception as e:
            print(f"Failed {filename}: {e}")
    # if i==10:
    #     break

# --- Save results to CSV ---
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "caption"])
    writer.writerows(results)

print(f"Captions saved to {output_csv}")

