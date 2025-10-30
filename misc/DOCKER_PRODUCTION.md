# Docker Production Setup Guide

## Overview

This project now includes optimized Docker configurations for both development and production environments.

## Files Created

1. **`.dockerignore`** - Excludes unnecessary files from Docker build context
2. **`docker-compose.prod.yml`** - Production-optimized compose configuration
3. **`docker-compose.yml`** - Development configuration (existing, with volume mounts)

## What's Excluded from Production Image

The `.dockerignore` file prevents these items from being copied into the Docker image:

- Development files (`.env`, `docker-compose.yml`)
- Output directories (`juspayWithImageCaptioningOutput/`)
- Log files (`*.log`, `*.txt`)
- Client code and test scripts
- Python cache files (`__pycache__/`, `*.pyc`)
- IDE configuration files
- Git metadata
- Documentation files
- **Model files (`paddlex/` directory)** - These are mounted at runtime instead

**Important:** The `paddlex/` directory containing all model files is excluded from the image and mounted at runtime. This significantly reduces the image size from ~35GB to ~10-15GB.

## Usage

### Development Mode

Use the standard docker-compose file with live volume mounts for code changes:

```bash
docker-compose up --build
```

This mounts your local directory into the container, allowing real-time code updates without rebuilding.

### Production Mode

Use the production compose file that relies on files baked into the image:

```bash
docker-compose -f docker-compose.prod.yml up --build
```

Or build and run separately:

```bash
# Build the production image
docker-compose -f docker-compose.prod.yml build

# Run in production
docker-compose -f docker-compose.prod.yml up -d
```

## Key Differences

| Aspect | Development | Production |
|--------|-------------|------------|
| Volume Mounts | `.:/app` (full directory) | Only specific config files |
| Image Size | Larger (includes all files) | Smaller (excludes dev files) |
| Code Updates | Live reload | Requires rebuild |
| Use Case | Local development | Deployment |

## Benefits of Production Setup

1. **Smaller Image Size** - Excludes unnecessary files, reducing image size
2. **Faster Builds** - Less data to copy and process
3. **Security** - No sensitive dev files (`.env`) in production image
4. **Cleaner Deployments** - Only production-necessary files included
5. **Better Performance** - No volume mount overhead

## Environment Variables

Both configurations support the same environment variables:

- `IMAGE_CAPTIONING_ENABLED` - Enable/disable image captioning (default: true)
- `PADDLEX_HPS_DEVICE_TYPE` - Device type (gpu/cpu)

### Two Approaches for Production:

#### Option 1: Keep .env file (Current Setup - RECOMMENDED)
The `.env` file is currently excluded from the Docker image for security. You have two ways to use it:

**A. Mount .env at runtime:**
```yaml
# In docker-compose.prod.yml, add to volumes:
volumes:
  - ./.env:/app/.env:ro  # Read-only mount
```

**B. Pass variables directly:**
```bash
# Set environment variables when running
IMAGE_CAPTIONING_ENABLED=false docker-compose -f docker-compose.prod.yml up
```

**C. Use a production .env file:**
```bash
# Create a production-specific .env file
cp .env .env.production

# Update docker-compose.prod.yml to use it
docker-compose -f docker-compose.prod.yml --env-file .env.production up
```

#### Option 2: Include .env in image (Less Secure)
If you want to bake the .env into the image:

1. Remove `.env` from `.dockerignore`
2. The .env will be copied into the image during build
3. **Warning**: This means secrets are in the image - not recommended for production

### Recommended Production Approach

**For turning off captioning in production:**

```bash
# Method 1: Pass at runtime
IMAGE_CAPTIONING_ENABLED=false docker-compose -f docker-compose.prod.yml up -d

# Method 2: Create production env file
echo "IMAGE_CAPTIONING_ENABLED=false" > .env.production
docker-compose -f docker-compose.prod.yml --env-file .env.production up -d

# Method 3: Mount .env file (add to docker-compose.prod.yml volumes)
volumes:
  - ./.env:/app/.env:ro
```

## Verifying Image Size

Check the difference in image sizes:

```bash
# Build with old method (no .dockerignore)
docker build -t paddlex/app:old .

# Build with new method (with .dockerignore)
docker build -t paddlex/app:new .

# Compare sizes
docker images | grep paddlex/app
```

## Notes

- The production configuration still mounts `config_gpu.pbtxt` to allow runtime configuration changes
- Remove this mount if you want a fully immutable image
- The Dockerfile itself doesn't need changes - `.dockerignore` handles the filtering
