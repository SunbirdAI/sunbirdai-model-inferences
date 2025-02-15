# Setting Up Network Volume Storage for RunPod Serverless Endpoints

This guide walks you through the process of creating and attaching a network volume to your RunPod serverless endpoint for efficient model caching.

## Prerequisites

- A RunPod account
- A serverless endpoint (existing or to be created)

## 1. Creating a Network Volume

1. Log in to your RunPod account
2. Navigate to the Storage Console
3. Click on "Create Network Volume"
4. Fill in the required details:
   - Name: Give your volume a descriptive name (e.g., "model-cache-volume")
   - Size: Choose the storage size (consider your model sizes)
   - Data Center: Select a data center (ensure it has the GPU types you need)
   - Type: Choose the storage type (SSD recommended for better performance)
5. Click "Create" to create your network volume

> **Note**: The data center you choose will constrain your serverless endpoint to that location.

## 2. Attaching the Network Volume to Your Serverless Endpoint

1. Go to [Serverless Endpoints](https://www.runpod.io/console/serverless)
2. Select your existing endpoint or create a new one
3. Click "Edit Endpoint" (or continue with endpoint creation)
4. Expand the "Advanced" section
5. Look for "Select Network Volume"
6. Choose the network volume you created
7. Expand the "Environment Variables" section
8. Add the following environment variables:
   - Click "Add Environment Variable"
   - Add first variable:
     - Key: `BASE_PATH`
     - Value: `/runpod-volume`
   - Click "Add Environment Variable" again
   - Add second variable:
     - Key: `HF_HOME`
     - Value: `/runpod-volume/huggingface`
9. Click "Update Endpoint" (or complete endpoint creation)

## 3. Configuring Your Container (Optional)

Update your Dockerfile to properly utilize the network volume:

```dockerfile
FROM runpod/base:0.4.0-cuda11.8.0

# Set the base path for model caching to the network volume
ENV BASE_PATH=/runpod-volume
ENV HF_HOME=/runpod-volume/huggingface

# Create network volume directory
RUN mkdir -p /runpod-volume

# Your other container configurations...
```

Create an entrypoint script (`entrypoint.sh`):

```bash
#!/bin/bash

# Check if we're running in a serverless environment
if [ -n "$RUNPOD_ENDPOINT_ID" ]; then
    export RUNPOD_NETWORK_VOLUME_PATH="/runpod-volume"
else
    export RUNPOD_NETWORK_VOLUME_PATH="/workspace"
    mkdir -p $RUNPOD_NETWORK_VOLUME_PATH
fi

# Set up environment variables for caching
export HF_HOME="$RUNPOD_NETWORK_VOLUME_PATH/huggingface"

# Execute the CMD
exec "$@"
```

## 4. Using the Network Volume in Your Code (Optional)

When loading models in your Python code, specify the cache directory:

```python
import os

# Get cache directory from environment
cache_dir = os.environ.get('HF_HOME')

# Use cache directory when loading models
model = AutoModel.from_pretrained("model-name", cache_dir=cache_dir)
```

## 5. Verifying the Setup

1. Deploy your endpoint with the updated configuration
2. Send a test request to your endpoint
3. Monitor the logs for the first request:
   - You should see model download messages
4. Send a second request:
   - You should NOT see model download messages
   - The initialization time should be significantly faster

## Troubleshooting

If you encounter issues:

1. Check if the volume is properly mounted:
   - Add debug logs to print the contents of `/runpod-volume`
   - Verify environment variables are set correctly

2. Common issues:
   - "No volume" in telemetry: Double-check volume attachment in endpoint settings
   - Slow performance: Ensure you selected SSD storage type
   - Models not caching: Verify cache directory paths in your code

## Best Practices

1. Choose a data center with your required GPU types before creating the network volume
2. Size your volume appropriately for your models
3. Use SSD storage for better performance
4. Implement proper error handling for cases where the volume might be unavailable

## Limitations

- Network volumes are data center specific
- Your serverless endpoint will be constrained to the same data center as your network volume
- Storage costs are based on the volume size, regardless of usage

For more information, refer to the [RunPod documentation](https://docs.runpod.io/docs). 