"""
Utility to properly map text to images using bounding box spatial relationships
Based on the logic from chunkByOCR.ts
"""
import json
import re
import pathlib
from typing import Dict, List, Optional, Tuple

def parse_bbox_from_image_filename(filename: str) -> Optional[str]:
    """
    Extract bbox key from image filename.
    Example: img_in_image_box_1169_347_2199_1236.jpg -> "1169_347_2199_1236"
    """
    if not filename:
        return None
    
    # Extract all numbers from filename
    numbers = re.findall(r'\d+', filename)
    
    # We need at least 4 numbers for bbox (x1, y1, x2, y2)
    if len(numbers) < 4:
        return None
    
    # Take the last 4 numbers as bbox coordinates
    return "_".join(numbers[-4:])

def normalize_bbox(bbox: List[float]) -> Optional[str]:
    """
    Normalize bbox array to string key by rounding coordinates.
    """
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    
    try:
        return "_".join(str(round(float(x))) for x in bbox)
    except (ValueError, TypeError):
        return None

def boxes_overlap(bbox1: List[float], bbox2: List[float], threshold: float = 0.1) -> bool:
    """
    Check if two bounding boxes overlap or are nearby.
    threshold: percentage of box size to consider as "nearby"
    """
    if len(bbox1) != 4 or len(bbox2) != 4:
        return False
    
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate box dimensions
    box1_width = x1_max - x1_min
    box1_height = y1_max - y1_min
    box2_width = x2_max - x2_min
    box2_height = y2_max - y2_min
    
    # Expand boxes slightly to catch nearby text
    margin_x = max(box1_width, box2_width) * threshold
    margin_y = max(box1_height, box2_height) * threshold
    
    x1_min_exp = x1_min - margin_x
    x1_max_exp = x1_max + margin_x
    y1_min_exp = y1_min - margin_y
    y1_max_exp = y1_max + margin_y
    
    # Check if boxes overlap or are nearby
    return not (x1_max_exp < x2_min or x1_min_exp > x2_max or
                y1_max_exp < y2_min or y1_min_exp > y2_max)

def is_text_above_image(text_bbox: List[float], img_bbox: List[float], max_distance: float = 100) -> bool:
    """Check if text is directly above the image"""
    if len(text_bbox) != 4 or len(img_bbox) != 4:
        return False
    
    text_x1, text_y1, text_x2, text_y2 = text_bbox
    img_x1, img_y1, img_x2, img_y2 = img_bbox
    
    # Text should be above (y2 of text < y1 of image)
    # And horizontally aligned (x ranges overlap)
    is_above = text_y2 <= img_y1 and (img_y1 - text_y2) <= max_distance
    x_overlap = not (text_x2 < img_x1 or text_x1 > img_x2)
    
    return is_above and x_overlap

def find_relevant_text_for_image(
    image_filename: str,
    image_bbox: List[float],
    parsing_res_list: List[Dict],
    page_text: str = ""
) -> str:
    """
    Find the text content for an image by matching its bbox.
    
    The block_content of image blocks IS the text describing that image,
    extracted by the layout parsing service.
    
    Strategy:
    1. Find the image block with matching bbox
    2. Return its block_content (the text for that image)
    3. Return empty string if not found
    """
    
    # Get image bbox key from filename
    img_bbox_key = parse_bbox_from_image_filename(image_filename)
    
    # Look for the image block with matching bbox
    for block in parsing_res_list:
        block_label = block.get("block_label", "")
        
        # We ONLY want image blocks
        if block_label != "image":
            continue
        
        block_bbox = block.get("block_bbox", [])
        block_content = block.get("block_content", "")
        
        # Check if this image block matches our bbox
        if len(block_bbox) == 4:
            block_bbox_key = normalize_bbox(block_bbox)
            if block_bbox_key and img_bbox_key and block_bbox_key == img_bbox_key:
                # Found the matching image block - return its content
                return block_content.strip() if block_content else ""
    
    # No matching image block found
    return ""

def clean_extracted_text(text: str) -> str:
    """
    Clean extracted text by removing HTML tags and image references.
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

# Example usage and testing
if __name__ == "__main__":
    # Test bbox parsing
    test_filename = "page_1_0_img_in_image_box_1169_347_2199_1236.jpg"
    bbox_key = parse_bbox_from_image_filename(test_filename)
    print(f"Filename: {test_filename}")
    print(f"BBox Key: {bbox_key}")
    
    # Test bbox normalization
    test_bbox = [1169.5, 347.2, 2199.8, 1236.1]
    normalized = normalize_bbox(test_bbox)
    print(f"\nBBox: {test_bbox}")
    print(f"Normalized: {normalized}")
