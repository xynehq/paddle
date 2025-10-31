import os
import csv
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# --- Device setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --- Load GIT model ---microsoft/git-large-textcaps
print("Loading GIT model...")
processor = AutoProcessor.from_pretrained("microsoft/git-large-textcaps")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-textcaps").to(device)
print("Model loaded successfully!")

# --- Caption generation function ---
def generate_caption(image_path):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        pixel_values=inputs.pixel_values,
        max_length=50,
        num_beams=3,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption

# --- Folder and output CSV ---
image_folder = "/home/aayush_shah/ppStruct/paddlex_hps_PP-StructureV3_sdk/downloads/images/"  # Replace with your folder path
output_csv = "captionsGITTextCaps.csv"

results = []

# --- Process each image in folder ---
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        full_path = os.path.join(image_folder, filename)
        try:
            caption = generate_caption(full_path)
            print(f"{filename}: {caption}")
            results.append([filename, caption])
        except Exception as e:
            print(f"Failed {filename}: {e}")

# --- Save results to CSV ---
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "caption"])
    writer.writerows(results)

print(f"Captions saved to {output_csv}")
