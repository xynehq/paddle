import base64
import json
import requests
import pathlib
import re
from bbox_text_matcher import (
    parse_bbox_from_image_filename,
    normalize_bbox,
    find_relevant_text_for_image,
    clean_extracted_text
)

outDirLocal="inovice0_1_5xyneHqImagesOutput"
API_URL = "http://localhost:8000/v2/models/layout-parsing/infer"  # Triton HTTP endpoint

file_path = "/home/aayush_shah/Downloads/samplePPT.pdf"

# Encode the local file with Base64
with open(file_path, "rb") as file:
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

print(response.status_code)
if response.status_code != 200:
    print(f"Error response: {response.text}")
    exit(1)

response_json = response.json()

try:
    output_data = response_json["outputs"][0]["data"][0]
    result = json.loads(output_data)["result"]
except (KeyError, IndexError, json.JSONDecodeError) as exc:
    print(f"Failed to parse response payload: {exc}\n{response_json}")
    exit(1)

print("\nDetected layout elements:")
print(len(result["layoutParsingResults"]))

# Extract and save images WITH associated text data
if "layoutParsingResults" in result:
    # Create downloads/images directory
    images_dir = pathlib.Path("downloads/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    image_counter = 0
    image_text_mapping = {}  # Map image filename to associated text
    
    for page_idx, res in enumerate(result["layoutParsingResults"]):
        # Extract page text from markdown
        page_text = ""
        if "markdown" in res and "text" in res["markdown"]:
            page_text = res["markdown"]["text"]
        
        # Save images from markdown if they exist
        if "markdown" in res and "images" in res["markdown"]:
            for img_path, img_b64 in res["markdown"]["images"].items():
                # Create new image filename
                original_name = pathlib.Path(img_path).name
                new_img_name = f"page_{page_idx + 1}_{image_counter}_{original_name}"
                new_img_path = images_dir / new_img_name
                
                # Save the image
                with open(new_img_path, "wb") as f:
                    f.write(base64.b64decode(img_b64))
                
                # Get image bbox from filename
                img_bbox_key = parse_bbox_from_image_filename(original_name)
                image_bbox = []
                
                # Try to get the actual bbox array from parsing_res_list
                if img_bbox_key and "prunedResult" in res and "parsing_res_list" in res["prunedResult"]:
                    parsing_res_list = res["prunedResult"]["parsing_res_list"]
                    for block in parsing_res_list:
                        if block.get("block_label") == "image":
                            block_bbox = block.get("block_bbox", [])
                            block_bbox_key = normalize_bbox(block_bbox)
                            if block_bbox_key == img_bbox_key:
                                image_bbox = block_bbox
                                break
                
                # Find relevant text using bbox-based spatial matching
                parsing_res_list = res.get("prunedResult", {}).get("parsing_res_list", [])
                relevant_text = find_relevant_text_for_image(
                    original_name,
                    image_bbox,
                    parsing_res_list,
                    page_text
                )
                
                # Clean the extracted text
                cleaned_text = clean_extracted_text(relevant_text)
                
                # Create prompt from cleaned text
                if cleaned_text and len(cleaned_text) > 10:
                    prompt = f"Based on this context: '{cleaned_text[:200]}', describe this image in detail"
                else:
                    prompt = "Describe this image in detail, focusing on key elements and their relationships"
                
                image_text_mapping[new_img_name] = {
                    "page": page_idx + 1,
                    "text": cleaned_text[:500] if cleaned_text else "",
                    "prompt": prompt,
                    "bbox": img_bbox_key if img_bbox_key else "unknown"
                }
                
                image_counter += 1
        
        # Handle outputImages if they exist
        if "outputImages" in res:
            for img_name, img_b64 in res["outputImages"].items():
                output_img_name = f"output_page_{page_idx + 1}_{img_name}.jpg"
                output_img_path = images_dir / output_img_name
                
                with open(output_img_path, "wb") as f:
                    f.write(base64.b64decode(img_b64))
                
                # Clean page text and create prompt
                cleaned_page_text = clean_extracted_text(page_text)
                
                if cleaned_page_text and len(cleaned_page_text) > 10:
                    prompt = f"Based on this context: '{cleaned_page_text[:200]}', describe this image in detail"
                else:
                    prompt = "Describe this image in detail, focusing on key elements and their relationships"
                
                # Store text for output images too
                image_text_mapping[output_img_name] = {
                    "page": page_idx + 1,
                    "text": cleaned_page_text[:500] if cleaned_page_text else "",
                    "prompt": prompt
                }
    
    # Save the image-text mapping to a JSON file
    mapping_file = images_dir / "image_text_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(image_text_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== SUMMARY ===")
    print(f"All images saved to: {images_dir}")
    print(f"Image-text mapping saved to: {mapping_file}")
    print(f"Total pages processed: {len(result['layoutParsingResults'])}")
    print(f"Total images saved: {image_counter}")
    
else:
    print("No layoutParsingResults found in response")
