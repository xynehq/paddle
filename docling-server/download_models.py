"""Run at docker build time to download and cache all docling models."""
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption

opts = PdfPipelineOptions()
opts.do_ocr = False
opts.generate_page_images = True
opts.generate_picture_images = True
opts.generate_table_images = False

DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
)
print("Models downloaded and cached.")
