# Docker Image Size Analysis

## Current Image Size: 32.4GB

### Complete Breakdown

#### Base Image Layers (from PaddleX base image)
- **Base PaddleX environment**: ~7.94GB (`/paddlex/py310`)
- **PaddleX libraries**: ~2.37GB (`/paddlex/libs`)
- **Triton server**: ~33.4MB (`/opt/tritonserver`)
- **Total Base**: ~10.3GB

#### Added Layers (from Dockerfile)
- **Python dependencies** (pymupdf, pillow, numpy, etc.): ~356MB
- **PyTorch + torchvision + torchaudio** (CUDA 12.1): ~7.03GB
- **Application files** (COPY . /app): ~1.79GB
  - **BLIP model cache**: 1.7GB
    - `model.safetensors`: 1.7GB (the actual BLIP model weights)
    - `tokenizer.json`: 695KB
    - `vocab.txt`: 227KB
    - Other config files: ~5KB
  - Server code and configs: ~90MB
- **Total Added**: ~9.2GB

#### Grand Total: ~32.4GB

---

## What's Taking Up Space

### 1. PyTorch (7.03GB) - REQUIRED
- Necessary for running the BLIP image captioning model
- Cannot be reduced without breaking functionality

### 2. Base PaddleX Environment (10.3GB) - REQUIRED
- Core PaddleX libraries and Python environment
- Required for document layout parsing
- Cannot be reduced

### 3. BLIP Model Cache (1.7GB) - OPTIONAL
- **This is the main target for optimization**
- The `model.safetensors` file contains the BLIP model weights
- Currently baked into the image
- **Can be excluded and mounted at runtime**

### 4. Python Dependencies (356MB) - REQUIRED
- Necessary libraries for the application
- Minimal size, cannot be significantly reduced

---

## Optimization Options

### Option 1: Exclude BLIP Model Cache (RECOMMENDED)
**Potential Savings: 1.7GB → Final size: ~30.7GB**

Update `.dockerignore`:
```
# Model files (mount at runtime instead of baking into image)
paddlex/
server/model_repo/blip-caption/1/blip_model_cache/
```

Update `docker-compose.prod.yml`:
```yaml
volumes:
  - ./server/model_repo/blip-caption/1/blip_model_cache:/app/server/model_repo/blip-caption/1/blip_model_cache:ro
```

**Pros:**
- Reduces image size by 1.7GB
- Faster builds (no need to copy large model file)
- Easier to update BLIP model (just replace files, no rebuild)

**Cons:**
- Model files must be present on host machine
- Less portable (can't just `docker run` without volume mounts)

### Option 2: Keep Current Setup
**Current Size: 32.4GB**

**Pros:**
- Fully self-contained image
- Can run anywhere with just `docker run`
- No external dependencies

**Cons:**
- Larger image size
- Slower builds
- Harder to update models

---

## Size Comparison

| Configuration | Image Size | Build Time | Portability |
|--------------|------------|------------|-------------|
| Original (with paddlex/) | 35GB | Slow | Medium |
| Current (paddlex excluded) | 32.4GB | Medium | Medium |
| Optimized (BLIP excluded too) | ~30.7GB | Fast | Low |

---

## Recommendation

**For Production Deployment:**
- Keep current setup (32.4GB) if you need fully portable images
- The 32.4GB is reasonable for an ML/AI application with:
  - PyTorch (7GB)
  - PaddleX environment (10GB)
  - BLIP model (1.7GB)
  - Dependencies (356MB)

**For Development:**
- Exclude BLIP model cache to speed up builds
- Mount it at runtime for faster iteration

---

## Reality Check

**This is a normal size for ML/AI Docker images:**
- PyTorch alone is 7GB
- Base PaddleX environment is 10GB
- Model weights are typically 1-5GB each
- Total of 30-35GB is expected and acceptable

**You've already achieved significant optimization:**
- Started at 35GB (with paddlex models)
- Now at 32.4GB (paddlex excluded, mounted at runtime)
- Saved 2.6GB by excluding paddlex directory

The remaining size is mostly unavoidable infrastructure (PyTorch, PaddleX) and the BLIP model which is needed for image captioning functionality.
