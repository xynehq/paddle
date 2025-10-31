"""
End-to-End GIT Conditional Image Captioning with Text Prompts
This script processes PDFs, extracts images and text, then generates captions using GIT with text prompts.
"""
import base64
import json
import requests
import pathlib
import os
import csv
import re
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from bbox_text_matcher import (
    parse_bbox_from_image_filename,
    normalize_bbox,
    find_relevant_text_for_image,
    clean_extracted_text
)

# ============== CONFIGURATION ==============
API_URL = "http://localhost:8000/v2/models/layout-parsing/infer"
PDF_FILE_PATH = "/home/aayush_shah/Downloads/samplePPT.pdf"
IMAGES_DIR = pathlib.Path("downloads/images")
OUTPUT_CSV = "git_conditional_captions.csv"

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============== STEP 1: PROCESS PDF ==============
print("\n=== STEP 1: Processing PDF ===")

# Encode the PDF file
with open(PDF_FILE_PATH, "rb") as file:
    file_bytes = file.read()
    file_b64 = base64.b64encode(file_bytes).decode("ascii")

input_payload = {
    "file": file_b64,
    "fileType": 0,
    "visualize": False,
}

request_payload = {
    "inputs": [
        {
            "name": "input",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [json.dumps(input_payload)],
        }
    ],
    "outputs": [
        {
            "name": "output"
        }
    ],
}

# Call the API
response = requests.post(
    API_URL,
    headers={"Content-Type": "application/json"},
    json=request_payload
)

print(f"API Response Status: {response.status_code}")
if response.status_code != 200:
    print(f"Error response: {response.text}")
    exit(1)

response_json = response.json()

try:
    output_data = response_json["outputs"][0]["data"][0]
    result = json.loads(output_data)["result"]
except (KeyError, IndexError, json.JSONDecodeError) as exc:
    print(f"Failed to parse response payload: {exc}")
    exit(1)

# ============== STEP 2: EXTRACT IMAGES & TEXT ==============
print("\n=== STEP 2: Extracting Images and Text ===")

if "layoutParsingResults" not in result:
    print("No layoutParsingResults found in response")
    exit(1)

# Create images directory
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

image_counter = 0
image_text_mapping = {}

for page_idx, res in enumerate(result["layoutParsingResults"]):
    # Extract page text from markdown
    page_text = ""
    if "markdown" in res and "text" in res["markdown"]:
        page_text = res["markdown"]["text"]
    
    # Save images from markdown
    if "markdown" in res and "images" in res["markdown"]:
        for img_path, img_b64 in res["markdown"]["images"].items():
            original_name = pathlib.Path(img_path).name
            new_img_name = f"page_{page_idx + 1}_{image_counter}_{original_name}"
            new_img_path = IMAGES_DIR / new_img_name
            
            # Save the image
            with open(new_img_path, "wb") as f:
                f.write(base64.b64decode(img_b64))
            
            # Get image bbox from filename
            img_bbox_key = parse_bbox_from_image_filename(original_name)
            image_bbox = []
            
            # Get the parsing_res_list for this page
            parsing_res_list = res.get("prunedResult", {}).get("parsing_res_list", [])
            
            # Try to get the actual bbox array from parsing_res_list by matching bbox key
            if img_bbox_key and parsing_res_list:
                for block in parsing_res_list:
                    if block.get("block_label") == "image":
                        block_bbox = block.get("block_bbox", [])
                        if block_bbox and len(block_bbox) == 4:
                            block_bbox_key = normalize_bbox(block_bbox)
                            if block_bbox_key == img_bbox_key:
                                image_bbox = block_bbox
                                break
            
            # Find relevant text using bbox-based spatial matching
            # Pass the original_name (which has bbox), the actual bbox array, and parsing_res_list
            relevant_text = find_relevant_text_for_image(
                original_name,
                image_bbox,
                parsing_res_list,
                page_text
            )
            
            # Clean the extracted text
            cleaned_text = clean_extracted_text(relevant_text)
            
            # Create prompt from extracted text
            if cleaned_text and len(cleaned_text) > 10:  # Only use if meaningful text exists
                prompt = f"Context: {cleaned_text[:150]}. Describe this image:"
            else:
                prompt = "Describe this image in detail:"
            
            image_text_mapping[new_img_name] = {
                "page": page_idx + 1,
                "text": cleaned_text[:500] if cleaned_text else "",
                "prompt": prompt,
                "bbox": img_bbox_key if img_bbox_key else "unknown"
            }
            
            image_counter += 1
    
    # Handle outputImages
    if "outputImages" in res:
        for img_name, img_b64 in res["outputImages"].items():
            output_img_name = f"output_page_{page_idx + 1}_{img_name}.jpg"
            output_img_path = IMAGES_DIR / output_img_name
            
            with open(output_img_path, "wb") as f:
                f.write(base64.b64decode(img_b64))
            
            # Clean page text
            cleaned_page_text = clean_extracted_text(page_text)
            
            # Create prompt
            if cleaned_page_text and len(cleaned_page_text) > 10:
                prompt = f"Context: {cleaned_page_text[:150]}. Describe this image:"
            else:
                prompt = "Describe this image in detail:"
            
            image_text_mapping[output_img_name] = {
                "page": page_idx + 1,
                "text": cleaned_page_text[:500] if cleaned_page_text else "",
                "prompt": prompt
            }

# Save mapping
mapping_file = IMAGES_DIR / "image_text_mapping.json"
with open(mapping_file, "w", encoding="utf-8") as f:
    json.dump(image_text_mapping, f, indent=2, ensure_ascii=False)

print(f"Extracted {image_counter} images with text prompts")
print(f"Image-text mapping saved to: {mapping_file}")

# ============== STEP 3: LOAD GIT MODEL ==============
print("\n=== STEP 3: Loading GIT Model ===")

processor = AutoProcessor.from_pretrained("microsoft/git-large-textcaps")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-textcaps").to(device)
print("GIT model loaded successfully!")

# ============== STEP 4: GENERATE CONDITIONAL CAPTIONS ==============
print("\n=== STEP 4: Generating Conditional Captions ===")

def generate_conditional_caption(image_path, text_prompt):
    """Generate caption with text prompt for GIT model"""
    img = Image.open(image_path).convert("RGB")
    
    # GIT accepts text prompts - concatenate prompt with special tokens
    # The model will use the prompt as context for generation
    inputs = processor(images=img, text=text_prompt, return_tensors="pt").to(device)
    
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

results = []

# Process each image with its associated text prompt
for filename in sorted(os.listdir(IMAGES_DIR)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        full_path = IMAGES_DIR / filename
        
        # Get the prompt for this image
        text_prompt = image_text_mapping.get(filename, {}).get("prompt", "Describe this image in detail:")
        extracted_text = image_text_mapping.get(filename, {}).get("text", "")
        page_num = image_text_mapping.get(filename, {}).get("page", "N/A")
        
        try:
            caption = generate_conditional_caption(full_path, text_prompt)
            print(f"\n{filename} (Page {page_num}):")
            print(f"  Prompt: {text_prompt[:100]}...")
            print(f"  Caption: {caption}")
            
            results.append([
                filename,
                page_num,
                extracted_text[:200],  # First 200 chars of extracted text
                text_prompt,
                caption
            ])
        except Exception as e:
            print(f"Failed to caption {filename}: {e}")
            results.append([filename, page_num, extracted_text[:200], text_prompt, f"ERROR: {e}"])

# ============== STEP 5: SAVE RESULTS ==============
print("\n=== STEP 5: Saving Results ===")

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "page", "extracted_text", "prompt", "caption"])
    writer.writerows(results)

print(f"\n=== COMPLETED ===")
print(f"Conditional captions saved to: {OUTPUT_CSV}")
print(f"Total images processed: {len(results)}")
