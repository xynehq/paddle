# Image Captioning Feature Configuration

This document explains how to control the image captioning feature in the PaddleX server.

## Overview

The image captioning feature uses a BLIP model to automatically generate captions for images found in documents during layout parsing. This feature is **enabled by default**.

## How It Works

1. **Global Flag**: The `IMAGE_CAPTIONING_ENABLED` environment variable controls the feature
2. **Default Value**: `true` (captioning is enabled)
3. **When Enabled**:
   - The `blip-server` container starts the BLIP Triton server
   - Images in documents get automatic captions via the BLIP model
   - Captions are added to the document's structured output
   
4. **When Disabled**:
   - The `blip-server` container sleeps and doesn't start the server
   - No BLIP server calls are made from model.py
   - Image captioning is completely skipped
   - Saves resources (GPU memory, processing time)

## Configuration Methods

### Method 1: Using .env File (Recommended)

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set the flag:
   ```bash
   # Enable captioning (default)
   IMAGE_CAPTIONING_ENABLED=true
   
   # OR disable captioning
   IMAGE_CAPTIONING_ENABLED=false
   ```

3. Start services:
   ```bash
   docker-compose up
   ```

### Method 2: Environment Variable

Set the environment variable before running docker-compose:

```bash
# Enable captioning
export IMAGE_CAPTIONING_ENABLED=true
docker-compose up

# Disable captioning
export IMAGE_CAPTIONING_ENABLED=false
docker-compose up
```

### Method 3: Inline with Docker Compose

```bash
# Enable captioning
IMAGE_CAPTIONING_ENABLED=true docker-compose up

# Disable captioning
IMAGE_CAPTIONING_ENABLED=false docker-compose up
```

## Quick Start Commands

### Start with Image Captioning (Default)

```bash
# Just run docker-compose (captioning enabled by default)
docker-compose up

# Or explicitly set the flag
IMAGE_CAPTIONING_ENABLED=true docker-compose up
```

### Start WITHOUT Image Captioning

```bash
# Set the flag to false
IMAGE_CAPTIONING_ENABLED=false docker-compose up

# Or use .env file
echo "IMAGE_CAPTIONING_ENABLED=false" > .env
docker-compose up
```

## How the Configuration Works

The `blip-server` container always starts, but its behavior depends on the environment variable:

- **When `IMAGE_CAPTIONING_ENABLED=true`** (default): The BLIP Triton server starts normally and processes caption requests
- **When `IMAGE_CAPTIONING_ENABLED=false`**: The container sleeps indefinitely and doesn't start the server (saves GPU resources)

Both `paddlex-server` and `blip-server` read the same `IMAGE_CAPTIONING_ENABLED` environment variable to coordinate their behavior.

## Verification

### Check if Captioning is Enabled

1. **Check environment variable in paddlex-server**:
   ```bash
   docker exec paddlex-server env | grep IMAGE_CAPTIONING_ENABLED
   ```

2. **Check blip-server logs**:
   ```bash
   docker logs blip-server
   ```
   - If enabled: You'll see Triton server startup logs
   - If disabled: You'll see "Image captioning is disabled. Blip-server will not start."

3. **Check paddlex-server logs**:
   ```bash
   docker logs paddlex-server | grep -i caption
   ```
   Look for messages like:
   - `Captioning cfg: enabled=True` (when enabled)
   - `Captioning cfg: enabled=False` (when disabled)

## Configuration File

The captioning behavior is also controlled by `server/model_repo/layout-parsing/1/caption_config.yaml`:

```yaml
captioning:
  enabled: true
  provider: triton
  triton_url: grpc://blip-server:8004
  triton_model: blip-caption
  timeout_ms: 5000
  max_length: 50
  num_beams: 3
  no_repeat_ngram_size: 2
```

**Note**: The `IMAGE_CAPTIONING_ENABLED` environment variable takes precedence over the YAML configuration.

## Troubleshooting

### Captioning Not Working

1. Verify the environment variable is set correctly:
   ```bash
   docker exec paddlex-server env | grep IMAGE_CAPTIONING_ENABLED
   docker exec blip-server env | grep IMAGE_CAPTIONING_ENABLED
   ```
2. Check blip-server logs to ensure Triton server started:
   ```bash
   docker logs blip-server
   ```
3. Check for connection errors in paddlex-server logs

### Want to Disable Captioning

1. Set `IMAGE_CAPTIONING_ENABLED=false` in `.env`
2. Restart services: `docker-compose up`
3. Verify in blip-server logs that it shows "Image captioning is disabled"

## Resource Savings

Disabling image captioning when not needed can save:
- **GPU Memory**: ~2-4GB (BLIP model size)
- **Processing Time**: ~0.5-2 seconds per image
- **Container Resources**: BLIP server doesn't consume GPU/CPU

## Examples

### Production with Captioning
```bash
# Create .env file
echo "IMAGE_CAPTIONING_ENABLED=true" > .env

# Start services
docker-compose up -d
```

### Development without Captioning (Faster)
```bash
# Create .env file
echo "IMAGE_CAPTIONING_ENABLED=false" > .env

# Start services
docker-compose up -d
```

### Testing Both Modes
```bash
# Test with captioning
IMAGE_CAPTIONING_ENABLED=true docker-compose up -d
# ... test ...
docker-compose down

# Test without captioning
IMAGE_CAPTIONING_ENABLED=false docker-compose up -d
# ... test ...
docker-compose down
```

## Summary

- **No additional flags needed**: Just set `IMAGE_CAPTIONING_ENABLED` in `.env` or as an environment variable
- **Simple command**: Always use `docker-compose up` (no profiles or extra flags)
- **Default behavior**: Captioning is enabled by default
- **Easy to toggle**: Change one environment variable to enable/disable the feature
