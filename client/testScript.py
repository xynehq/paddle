import base64
import json
import requests
import pathlib

outDirLocal="juspayWithImageCaptioningOutput"
API_URL = "http://localhost:8000/v2/models/layout-parsing/infer"  # Triton HTTP endpoint
#API_URL = "https://unmasked-geri-blaneless.ngrok-free.app/v2/models/layout-parsing/infer"

#file_path = "/home/aayush_shah/Downloads/LIC.pdf"
file_path = "/home/aayush_shah/Downloads/juspay.pdf"
# file_path = "/home/aayush_shah/Downloads/passport3.pdf"
# file_path = "/home/aayush_shah/Downloads/invoiceMMT.pdf"
# file_path = "/home/aayush_shah/Downloads/aadharTemp.pdf"
# file_path = "/home/aayush_shah/Downloads/aadhar.png"
#file_path = "/home/aayush_shah/Downloads/rbiLib.pdf"
# file_path = '/home/aayush_shah/projects/pdfIngestionPaddle/LIC.pdf'
# file_path = './execCharts2.pdf'  # Example file path
# Encode the local file with Base64
with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_b64 = base64.b64encode(file_bytes).decode("ascii")

# "useChartRecognition": True,
#     "useDocUnwarping": True,
#     "useDocOrientationClassify": True,
#     "useSealRecognition": True,
#     "useTableRecognition": True,
#     "useFormulaRecognition": True,
#     "useRegionDetection": True,
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

# Process the response data
# assert response.status_code == 200
print(response.status_code)
if response.status_code != 200:
    print(f"Error response: {response.text}")
    exit(1)

response_json = response.json()

# Write response_json to a new JSON file
with open("response_outputNew2.json", "w") as f:
    json.dump(response_json, f, indent=2)
print(f"Response JSON saved to: response_outputNew2.json")

# print(len(response_json["outputs"][0]["data"]))
try:
    output_data = response_json["outputs"][0]["data"][0]
    result = json.loads(output_data)["result"]
except (KeyError, IndexError, json.JSONDecodeError) as exc:
    print(f"Failed to parse response payload: {exc}\n{response_json}")
    exit(1)




print("\nDetected layout elements:")
print(len(result["layoutParsingResults"]))

# # Print parsing_res_list and markdown for ALL pages
# if "layoutParsingResults" in result and len(result["layoutParsingResults"]) > 0:
#     for page_idx, layout_result in enumerate(result["layoutParsingResults"]):
#         print(f"\n=== PAGE {page_idx + 1} - PARSING RESULTS LIST ===")
        
#         # Print parsing_res_list for this page
#         if "prunedResult" in layout_result and "parsing_res_list" in layout_result["prunedResult"]:
#             parsing_res_list = layout_result["prunedResult"]["parsing_res_list"]
#             # for i, parsing_result in enumerate(parsing_res_list):
#             #     print(f"Block {i+1}:")
#             #     print(f"  Label: {parsing_result.get('block_label', 'N/A')}")
#             #     print(f"  Content: {parsing_result.get('block_content', 'N/A')}")
#             #     print(f"  BBox: {parsing_result.get('block_bbox', 'N/A')}")
#             #     print()
#             print(layout_result["prunedResult"]["parsing_res_list"])
#         else:
#             print("No prunedResult or parsing_res_list found for this page")
        
#         print(f"\n=== PAGE {page_idx + 1} - MARKDOWN TEXT ===")
        
#         # Print markdown text for this page
#         if "markdown" in layout_result and "text" in layout_result["markdown"]:
#             print(layout_result["markdown"]["text"])
#         else:
#             print("No markdown text found for this page")
        
#         print("\n" + "="*80 + "\n")
# else:
#     print("No layoutParsingResults found in response")

#======
# Extract block_content from all pages and concatenate into one file
if "layoutParsingResults" in result:
    # Create output directory
    output_dir = pathlib.Path(outDirLocal)
    output_dir.mkdir(exist_ok=True)
    images_dir = output_dir / "images2"
    images_dir.mkdir(exist_ok=True)
    
    combined_markdown = []
    combined_block_content = []
    image_counter = 0
    total_blocks = 0
    
    for page_idx, res in enumerate(result["layoutParsingResults"]):
        # print(f"Processing layoutParsingResult {page_idx}")
        
        # Extract ALL block information from parsing_res_list
        if "prunedResult" in res and "parsing_res_list" in res["prunedResult"]:
            parsing_res_list = res["prunedResult"]["parsing_res_list"]
            
            # Add page separator for block content
            if page_idx > 0:
                combined_block_content.append(f"\n\n=== PAGE {page_idx + 1} ===\n\n")
            else:
                combined_block_content.append(f"=== PAGE {page_idx + 1} ===\n\n")
            
            # Extract just block_content from each parsing_result
            for block_idx, parsing_result in enumerate(parsing_res_list):
                block_content = parsing_result.get('block_content', '')
                # Write all block content, including empty ones
                combined_block_content.append(f"{block_content}\n")
                total_blocks += 1
        
        # Collect markdown text (keeping existing functionality)
        if "markdown" in res and "text" in res["markdown"]:
            page_markdown = res["markdown"]["text"]
            
            # Add page separator if not the first page
            if page_idx > 0:
                combined_markdown.append(f"\n\n---\n# Page {page_idx + 1}\n\n")
            else:
                combined_markdown.append(f"# Page {page_idx + 1}\n\n")
            
            # Save images from markdown and update image references
            if "images" in res["markdown"]:
                for img_path, img_b64 in res["markdown"]["images"].items():
                    # Create new image filename
                    original_name = pathlib.Path(img_path).name
                    new_img_name = f"page_{page_idx + 1}_{image_counter}_{original_name}"
                    new_img_path = images_dir / new_img_name
                    
                    # Save the image
                    with open(new_img_path, "wb") as f:
                        f.write(base64.b64decode(img_b64))
                    # print(f"Saved image: {new_img_path}")
                    
                    # Update image reference in markdown
                    page_markdown = page_markdown.replace(img_path, f"images/{new_img_name}")
                    image_counter += 1
            
            combined_markdown.append(page_markdown)
        
        # Handle outputImages if they exist
        if "outputImages" in res:
            for img_name, img_b64 in res["outputImages"].items():
                output_img_name = f"output_page_{page_idx + 1}_{img_name}.jpg"
                output_img_path = images_dir / output_img_name
                
                with open(output_img_path, "wb") as f:
                    f.write(base64.b64decode(img_b64))
                # print(f"Saved output image: {output_img_path}")
        # else:
            # print(f"No outputImages found for page {page_idx}")
    
    # Save combined block content file
    combined_block_path = output_dir / "combined_block_content.txt"
    with open(combined_block_path, "w", encoding="utf-8") as f:
        f.write("".join(combined_block_content))
    
    # Save combined markdown file (keeping existing functionality)
    combined_md_path = output_dir / "complete_document2.md"
    with open(combined_md_path, "w", encoding="utf-8") as f:
        f.write("".join(combined_markdown))
    
    print(f"\n=== SUMMARY ===")
    print(f"Combined block content saved to: {combined_block_path}")
    print(f"Combined markdown saved to: {combined_md_path}")
    print(f"All images saved to: {images_dir}")
    print(f"Total pages processed: {len(result['layoutParsingResults'])}")
    print(f"Total blocks extracted: {total_blocks}")
    
else:
    print("No layoutParsingResults found in response")

# Print the full res    ult structure for debugging


#=====
# The detailed output above already shows all pages, so removing this redundant section

# Uncomment below to see the full JSON structure for debugging
# print(json.dumps(result, indent=2))
