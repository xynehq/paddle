
import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from init_models import initialize_models
from processor import process_document


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    converter, chunker, vlm_config = initialize_models()
    app.state.doc_converter = converter
    app.state.chunker       = chunker
    app.state.vlm_config    = vlm_config
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Docling Document Processing Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    if not getattr(app.state, "doc_converter", None) or not getattr(app.state, "chunker", None):
        raise HTTPException(status_code=503, detail="Models not initialized")
    return {"status": "ok", "models_loaded": True}


@app.post("/process")
async def process_document_endpoint(
    file:   UploadFile = File(...),
    doc_id: str        = Form(...),
):
    """Process a PDF and return structured TOC, text chunks, and images.

    Response fields:
    - **metadata**     - document info and processing stats
    - **toc**          - table of contents with hierarchy
    - **chunks**       - contextualized text chunks with page/section metadata
    - **image_chunks** - image metadata + descriptions (for search indexing)
    - **images**       - base64-encoded images keyed as ``img_0``, ``img_1``, …
    """
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    suffix      = Path(file.filename).suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)

    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())

        result = await asyncio.get_running_loop().run_in_executor(
            None,
            process_document,
            tmp_path,
            doc_id,
            app.state.doc_converter,
            app.state.chunker,
            app.state.vlm_config,
        )

        # Debug: save result to file
        debug_path = f"/tmp/debug_result_{doc_id}.json"
        with open(debug_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[DEBUG] Result saved to: {debug_path}")

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
