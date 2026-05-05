
from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
