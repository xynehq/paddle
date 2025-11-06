# Docker Compose Profiles Usage Guide

## Overview

Both `docker-compose.yml` and `docker-compose.prod.yml` now use Docker Compose **profiles** to conditionally start the `blip-server` container for image captioning functionality.

## How It Works

### Container Behavior

1. **paddlex-server**: Always starts (no profile required)
   - Reads `IMAGE_CAPTIONING_ENABLED` environment variable
   - Controls captioning behavior internally via Python code
   - Default: `IMAGE_CAPTIONING_ENABLED=true`

2. **blip-server**: Only starts when `image-captioning` profile is active
   - Provides the BLIP Triton server for image captioning
   - No longer checks `IMAGE_CAPTIONING_ENABLED` in its startup script
   - Controlled entirely by Docker Compose profiles

## Usage

### Development (docker-compose.yml)

#### With Image Captioning (Both Containers)
```bash
# Start both paddlex-server and blip-server
IMAGE_CAPTIONING_ENABLED=true docker-compose --profile image-captioning up

# Or with detached mode
IMAGE_CAPTIONING_ENABLED=true docker-compose --profile image-captioning up -d
```

#### Without Image Captioning (Only paddlex-server)
```bash
# Start only paddlex-server (blip-server won't start)
IMAGE_CAPTIONING_ENABLED=false docker-compose up

# Or with detached mode
IMAGE_CAPTIONING_ENABLED=false docker-compose up -d
```

### Production (docker-compose.prod.yml)

#### Build the Image

**Option 1: Build with docker-compose (uses image name from compose file)**
```bash
# Build the Docker image (will use image: paddlex/app:latest from compose file)
docker-compose -f docker-compose.prod.yml build
```

**Option 2: Build and tag with docker command**
```bash
# Build with custom tag using docker directly
docker build -t paddlex-app:prod .

# Or build with multiple tags
docker build -t paddlex-app:prod -t paddlex-app:latest .
```

**Option 3: Build with docker-compose then tag**
```bash
# First build with docker-compose
docker-compose -f docker-compose.prod.yml build

# Then tag the built image
docker tag paddlex/app:latest paddlex-app:prod
```

#### With Image Captioning (Both Containers)

**Option 1: Using .env file (Recommended)**
```bash
# Set in .env file: IMAGE_CAPTIONING_ENABLED=true
# Then run:
docker-compose -f docker-compose.prod.yml --profile image-captioning up -d
```

**Option 2: Override .env with command line**
```bash
# This overrides whatever is in .env
IMAGE_CAPTIONING_ENABLED=true docker-compose -f docker-compose.prod.yml --profile image-captioning up -d
```

#### Without Image Captioning (Only paddlex-server)

**Option 1: Using .env file (Recommended)**
```bash
# Set in .env file: IMAGE_CAPTIONING_ENABLED=false
# Then run:
docker-compose -f docker-compose.prod.yml up -d
```

**Option 2: Override .env with command line**
```bash
# This overrides whatever is in .env
IMAGE_CAPTIONING_ENABLED=false docker-compose -f docker-compose.prod.yml up -d
```

#### Build and Start in One Command

**With .env file:**
```bash
# With image captioning (IMAGE_CAPTIONING_ENABLED=true in .env)
docker-compose -f docker-compose.prod.yml --profile image-captioning up -d --build

# Without image captioning (IMAGE_CAPTIONING_ENABLED=false in .env)
docker-compose -f docker-compose.prod.yml up -d --build
```

**Override .env with command line:**
```bash
# With image captioning
IMAGE_CAPTIONING_ENABLED=true docker-compose -f docker-compose.prod.yml --profile image-captioning up -d --build

# Without image captioning
IMAGE_CAPTIONING_ENABLED=false docker-compose -f docker-compose.prod.yml up -d --build
```

## Using .env File (Recommended)

**No, you don't need to pass `IMAGE_CAPTIONING_ENABLED` in the command if you have it in `.env` file.**

Set the environment variable in a `.env` file:

```bash
# .env file for image captioning ENABLED
IMAGE_CAPTIONING_ENABLED=true
PADDLEX_HPS_DEVICE_TYPE=gpu
```

```bash
# .env file for image captioning DISABLED
IMAGE_CAPTIONING_ENABLED=false
PADDLEX_HPS_DEVICE_TYPE=gpu
```

Then run (docker-compose automatically reads `.env`):
```bash
# With captioning (if .env has IMAGE_CAPTIONING_ENABLED=true)
docker-compose -f docker-compose.prod.yml --profile image-captioning up -d

# Without captioning (if .env has IMAGE_CAPTIONING_ENABLED=false)
docker-compose -f docker-compose.prod.yml up -d
```

**Note:** The `.env` file is automatically loaded by docker-compose, so you only need to pass the environment variable in the command if you want to override what's in `.env`.

## Benefits of This Approach

1. **Resource Efficiency**: When captioning is disabled, the `blip-server` container doesn't start at all, saving:
   - GPU memory allocation
   - 1GB shared memory
   - Container overhead

2. **Clear Separation**: 
   - Profile controls whether `blip-server` starts
   - Environment variable controls whether `paddlex-server` uses captioning

3. **No Idle Containers**: Eliminates the previous `sleep infinity` pattern where containers would run but do nothing

## Important Notes

- **paddlex-server** always needs `IMAGE_CAPTIONING_ENABLED` in its environment (used by Python code)
- **blip-server** no longer needs `IMAGE_CAPTIONING_ENABLED` (controlled by profile)
- When `IMAGE_CAPTIONING_ENABLED=false`, the paddlex-server will internally disable captioning even if blip-server is running
- When the `image-captioning` profile is not active, blip-server simply won't start

## Checking Running Containers

```bash
# See which containers are running
docker-compose ps

# With captioning enabled, you should see both:
# - paddlex-server
# - blip-server

# Without captioning, you should only see:
# - paddlex-server
```

## Troubleshooting

### Both containers start even when IMAGE_CAPTIONING_ENABLED=false
- Make sure you're NOT using the `--profile image-captioning` flag when you want captioning disabled

### blip-server doesn't start when IMAGE_CAPTIONING_ENABLED=true
- Make sure you include the `--profile image-captioning` flag in your docker-compose command

### paddlex-server still tries to use captioning when blip-server isn't running
- Ensure `IMAGE_CAPTIONING_ENABLED=false` is set in the environment for paddlex-server
- The Python code in paddlex-server checks this variable to disable captioning internally
