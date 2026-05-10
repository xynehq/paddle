
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class VlmConfig:
    preset:       str
    endpoint_url: str
    model:        str
    timeout:      float
    max_tokens:   int
    image_prompt: str
    token:        str = ""  # Bearer token for the Authorization header


@dataclass
class TocEntry:
    section_number: str
    section_title:  str
    page_number:    int
    level:          int
    bbox:           Optional[Dict[str, float]] = None
    parent_index:   Optional[int] = None


@dataclass
class DocumentChunk:
    text:         str
    headings:     List[str]
    page_numbers: List[int]
    bbox:         Optional[Dict[str, float]] = None


@dataclass
class ImageChunk:
    text:          str
    image_base64:  str
    page_number:   int
    bbox:          Optional[Dict[str, float]] = None
    width:         Optional[int] = None
    height:        Optional[int] = None


@dataclass
class VlmDetectedImage:
    marker:     str
    bbox:       Dict[str, float]
    pixel_bbox: Tuple[int, int, int, int]
    width:      int
    height:     int
    start:      int
    end:        int


@dataclass
class VlmPageResult:
    text:                    str
    image_chunks:            List[ImageChunk] = field(default_factory=list)
    image_regions_detected:  int = 0
    crop_ocr_attempted:      int = 0
    crop_ocr_success:        int = 0
    crop_ocr_failed:         int = 0
    crop_ocr_skipped:        int = 0
    crop_ocr_skipped_small:  int = 0
