# Docling OCR Service

Drop-in replacement for PaddleOCR using Docling. Processes PDFs with layout analysis and outputs structured data for Xyne.

## Quick Start

```bash
# Build and run
docker-compose up -d

# Test
curl http://localhost:8000/health

# Process PDF
curl -X POST http://localhost:8000/process \
  -F "file=@your-document.pdf"
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /instance_status` | Triton-compatible status |
| `POST /process` | Process PDF (multipart) |
| `POST /v2/models/layout-parsing/infer` | Triton-compatible |

## Integration with Xyne

Update Xyne config to use port 8000:

```typescript
// config.ts
const OCR_SERVICE_URL = process.env.OCR_URL || "http://localhost:8000"
```

## Output Format

Matches PaddleOCR format for backward compatibility:

```json
{
  "chunks": ["text chunk 1", "text chunk 2"],
  "chunks_pos": [0, 1],
  "chunks_map": [{
    "chunk_index": 0,
    "page_numbers": [1],
    "block_labels": ["paragraph"],
    "bbox": {"l": 0.1, "t": 0.2, "r": 0.9, "b": 0.3}
  }],
  "metadata": {
    "num_pages": 10,
    "processing_method": "docling"
  }
}
```

## Differences from Paddle

- **CPU-based**: No GPU required
- **Better layout detection**: Uses RT-DETR model
- **Table structure**: Uses TableFormer for accurate tables
- **Slower on first run**: Downloads models on startup

## Troubleshooting

**First start is slow**: Models download on first run (~2GB). Subsequent starts are fast.

**Out of memory**: Increase container memory limit to 4GB+.

**Slow processing**: Normal - layout analysis is CPU-intensive.
