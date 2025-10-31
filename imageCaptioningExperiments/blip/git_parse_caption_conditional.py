import os
import csv
import json
import pathlib
import re
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# --- Device setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --- Text cleaning function ---
def clean_extracted_text(text):
    """
    Clean extracted text by removing HTML tags and image references.
    Returns empty string if only HTML/image content remains.
    """
    if not text or not text.strip():
        return ""
    
    # Remove HTML div tags with image references
    text = re.sub(r'<div[^>]*>\s*<img[^>]*>\s*</div>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove any remaining img tags
    text = re.sub(r'<img[^>]*>', '', text, flags=re.IGNORECASE)
    
    # Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- Load GIT model ---
print("Loading GIT model...")
processor = AutoProcessor.from_pretrained("microsoft/git-large-textcaps")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-textcaps").to(device)
print("Model loaded successfully!")

# --- Caption generation function with optional text prompt ---
def generate_caption(image_path, text_prompt=None):
    img = Image.open(image_path).convert("RGB")
    
    if text_prompt:
        # Conditional captioning with text prompt
        inputs = processor(images=img, text=text_prompt, return_tensors="pt").to(device)
    else:
        # Standard captioning without prompt
        inputs = processor(images=img, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        pixel_values=inputs.pixel_values,
        input_ids=inputs.input_ids if hasattr(inputs, 'input_ids') else None,
        max_new_tokens=200,  # Allow longer captions (200 tokens)
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
        temperature=0.9
    )
    
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption

# --- Folder and output CSV ---
image_folder = "/home/aayush_shah/ppStruct/paddlex_hps_PP-StructureV3_sdk/downloads/images/"
output_csv = "captionsGIT_conditional.csv"

# --- Check for image-text mapping file ---
mapping_file = pathlib.Path(image_folder) / "image_text_mapping.json"
image_text_mapping = {}

if mapping_file.exists():
    print(f"Loading text prompts from: {mapping_file}")
    with open(mapping_file, "r", encoding="utf-8") as f:
        image_text_mapping = json.load(f)
    print(f"Loaded prompts for {len(image_text_mapping)} images")
else:
    print("No image-text mapping found. Using standard captioning without prompts.")

results = []

# --- Process each image in folder ---
for filename in sorted(os.listdir(image_folder)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        full_path = os.path.join(image_folder, filename)
        
        # Get text prompt if available
        text_prompt = None
        extracted_text = ""
        page_num = "N/A"
        
        if filename in image_text_mapping:
            mapping_data = image_text_mapping[filename]
            raw_prompt = mapping_data.get("prompt", None)
            raw_text = mapping_data.get("text", "")
            
            # Clean the text and prompt
            extracted_text = clean_extracted_text(raw_text)
            if raw_prompt:
                text_prompt = raw_prompt  # Prompt already cleaned when created
            else:
                text_prompt = None
            page_num = mapping_data.get("page", "N/A")
        
        try:
            caption = generate_caption(full_path, text_prompt)
            
            if text_prompt:
                print(f"{filename} (Page {page_num}):")
                print(f"  Prompt: {text_prompt[:80]}...")
                print(f"  Caption: {caption}")
            else:
                print(f"{filename}: {caption}")
            
            results.append([
                filename,
                page_num,
                extracted_text[:200] if extracted_text else "",
                text_prompt if text_prompt else "N/A",
                caption
            ])
        except Exception as e:
            print(f"Failed {filename}: {e}")
            results.append([filename, page_num, extracted_text[:200], text_prompt if text_prompt else "N/A", f"ERROR: {e}"])

# --- Save results to CSV ---
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "page", "extracted_text", "prompt", "caption"])
    writer.writerows(results)

print(f"\nCaptions saved to {output_csv}")
print(f"Total images processed: {len(results)}")
