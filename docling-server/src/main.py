
import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from init_models import initialize_models
from job_tracker import tracker
from processor import process_document


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    converter, hybrid_chunker, vlm_config, sem_chunker = initialize_models()
    app.state.doc_converter = converter
    app.state.hybrid_chunker = hybrid_chunker
    app.state.sem_chunker = sem_chunker
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
    if not getattr(app.state, "doc_converter", None) or not getattr(app.state, "hybrid_chunker", None):
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

    print(f"Processing file: {file.filename} (doc_id={doc_id})", flush=True)
    suffix      = Path(file.filename).suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)

    tracker.start(doc_id, filename=file.filename)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())

        result = await asyncio.get_running_loop().run_in_executor(
            None,
            process_document,
            tmp_path,
            doc_id,
            app.state.doc_converter,
            app.state.hybrid_chunker,
            app.state.sem_chunker,
            app.state.vlm_config,
            tracker.stage_setter(doc_id),
        )

        # Debug: save result to file with timestamp to avoid overwriting
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = debug_dir / f"{doc_id}_{timestamp}.json"
        with open(debug_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[DEBUG] Result saved to: {debug_path}")

        tracker.done(doc_id)
        return JSONResponse(content=result)

    except HTTPException as exc:
        tracker.fail(doc_id, exc.detail if isinstance(exc.detail, str) else str(exc.detail))
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        tracker.fail(doc_id, str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.get("/status")
async def get_all_status():
    """Snapshot of every tracked job (running + recently completed).

    Use this for a dashboard-style view. Entries are pruned
    STATUS_TTL_SECONDS after they finish (default 1h).
    """
    return tracker.all()


@app.get("/status/{identifier}")
async def get_status(identifier: str):
    """Return the current processing state for a job.

    ``identifier`` can be either the ``doc_id`` or the original ``filename``.
    If multiple jobs share the same filename, the most recent one is returned.

    States: running | done | failed. 404 if no job matches (or its entry has
    been pruned after STATUS_TTL_SECONDS).
    """
    entry = tracker.find(identifier)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No job found for '{identifier}'")
    return entry


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
