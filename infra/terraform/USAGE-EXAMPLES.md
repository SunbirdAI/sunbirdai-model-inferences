# GPU Instance Usage Examples

This guide provides practical examples for using your GPU instance with ML workloads.

## Prerequisites

- Instance deployed and GPU setup complete
- Instance rebooted after initial setup
- GPU verified with `nvidia-smi`

## Example 1: Running the Sunbird AI Inference Service

### With Environment Variables

```bash
# Export your environment variables
export HF_TOKEN="your-huggingface-token"
export RUNPOD_ENDPOINT_ID="your-runpod-endpoint"
export AUDIO_CONTENT_BUCKET_NAME="your-bucket-name"
export GCP_CREDENTIALS="your-gcp-credentials"

# Run the container
docker run --rm --gpus all -p 8088:8088 \
    -e HF_TOKEN \
    -e RUNPOD_ENDPOINT_ID \
    -e AUDIO_CONTENT_BUCKET_NAME \
    -e GCP_CREDENTIALS \
    sunbirddocker/sunbirdai-model-inferences:v2.3.11
```

### With Environment File

Create `.env` file:
```bash
cat > inference.env <<EOF
HF_TOKEN=your-huggingface-token
RUNPOD_ENDPOINT_ID=your-runpod-endpoint
AUDIO_CONTENT_BUCKET_NAME=your-bucket-name
GCP_CREDENTIALS=your-gcp-credentials
EOF
```

Run with env file:
```bash
docker run --rm --gpus all -p 8088:8088 \
    --env-file inference.env \
    sunbirddocker/sunbirdai-model-inferences:v2.3.11
```

### Running as Daemon (Background)

```bash
docker run -d --name sunbird-inference \
    --gpus all \
    -p 8088:8088 \
    --restart unless-stopped \
    -e HF_TOKEN \
    -e RUNPOD_ENDPOINT_ID \
    -e AUDIO_CONTENT_BUCKET_NAME \
    -e GCP_CREDENTIALS \
    sunbirddocker/sunbirdai-model-inferences:v2.3.11

# Check logs
docker logs -f sunbird-inference

# Stop container
docker stop sunbird-inference

# Start container
docker start sunbird-inference
```

## Example 2: PyTorch with GPU

### Interactive Session

```bash
docker run --rm --gpus all -it \
    -v $(pwd):/workspace \
    -w /workspace \
    pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
    bash
```

Inside container:
```python
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Create tensor on GPU
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = torch.matmul(x, y)
print(f"Result shape: {z.shape}")
```

### Running Training Script

```bash
# Create a simple training script
cat > train.py <<'EOF'
import torch
import torch.nn as nn

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simple model
model = nn.Linear(100, 10).to(device)
optimizer = torch.optim.Adam(model.parameters())

# Dummy training loop
for epoch in range(10):
    x = torch.randn(32, 100).to(device)
    y = torch.randn(32, 10).to(device)
    
    output = model(x)
    loss = nn.functional.mse_loss(output, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
EOF

# Run training
docker run --rm --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
    python train.py
```

## Example 3: TensorFlow with GPU

```bash
docker run --rm --gpus all -it \
    tensorflow/tensorflow:2.14.0-gpu \
    python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPU available:', tf.config.list_physical_devices('GPU'))
print('GPU name:', tf.test.gpu_device_name())
"
```

## Example 4: Jupyter Notebook with GPU

```bash
# Run Jupyter with GPU support
docker run --rm --gpus all \
    -p 8888:8888 \
    -v $(pwd):/workspace \
    -e JUPYTER_ENABLE_LAB=yes \
    jupyter/tensorflow-notebook

# Access at: http://<your-instance-ip>:8888
# Token will be shown in the output
```

## Example 5: vLLM for LLM Inference

```bash
# Run vLLM server
docker run --rm --gpus all \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-2-7b-hf \
    --dtype float16

# Test the endpoint
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-hf",
        "prompt": "San Francisco is a",
        "max_tokens": 50
    }'
```

## Example 6: Docker Compose Setup

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  ml-inference:
    image: sunbirddocker/sunbirdai-model-inferences:v2.3.11
    ports:
      - "8088:8088"
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - RUNPOD_ENDPOINT_ID=${RUNPOD_ENDPOINT_ID}
      - AUDIO_CONTENT_BUCKET_NAME=${AUDIO_CONTENT_BUCKET_NAME}
      - GCP_CREDENTIALS=${GCP_CREDENTIALS}
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  jupyter:
    image: jupyter/tensorflow-notebook
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
    environment:
      - JUPYTER_ENABLE_LAB=yes
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  monitoring:
    image: nvcr.io/nvidia/k8s/dcgm-exporter:3.1.3-3.1.4-ubuntu20.04
    ports:
      - "9400:9400"
    restart: unless-stopped
    cap_add:
      - SYS_ADMIN
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Run with:
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

## Example 7: Custom ML Container

Create `Dockerfile`:
```dockerfile
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install ML libraries
RUN pip3 install --no-cache-dir \
    torch \
    transformers \
    accelerate \
    numpy \
    pandas

# Copy your application
COPY app.py /app/
WORKDIR /app

# Run application
CMD ["python3", "app.py"]
```

Build and run:
```bash
# Build image
docker build -t my-ml-app:latest .

# Run with GPU
docker run --rm --gpus all my-ml-app:latest
```

## Example 8: Multi-GPU Setup (if using larger instance)

```python
# For instances with multiple GPUs
import torch

# Use all GPUs
model = nn.DataParallel(model)

# Or specific GPUs
model = nn.DataParallel(model, device_ids=[0, 1])
```

Docker command for multi-GPU:
```bash
# Use specific GPUs
docker run --rm --gpus '"device=0,1"' your-image

# Use all GPUs
docker run --rm --gpus all your-image
```

## Example 9: Monitoring GPU Usage

### In Container
```bash
docker run --rm --gpus all \
    nvidia/cuda:12.4.0-base-ubuntu22.04 \
    watch -n 1 nvidia-smi
```

### On Host
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# GPU utilization
nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1

# Memory usage
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1

# Power usage
nvidia-smi --query-gpu=power.draw --format=csv -l 1
```

### Python Script for Monitoring
```python
import subprocess
import time

def get_gpu_stats():
    result = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
        '--format=csv,noheader,nounits'
    ]).decode()
    
    util, mem_used, mem_total, temp = result.strip().split(',')
    return {
        'utilization': float(util),
        'memory_used': int(mem_used),
        'memory_total': int(mem_total),
        'temperature': int(temp)
    }

while True:
    stats = get_gpu_stats()
    print(f"GPU: {stats['utilization']}% | "
          f"Memory: {stats['memory_used']}/{stats['memory_total']} MB | "
          f"Temp: {stats['temperature']}°C")
    time.sleep(1)
```

## Example 10: Resource Limits

### Memory Limits
```bash
docker run --rm --gpus all \
    --memory="32g" \
    --memory-swap="32g" \
    --cpus="8" \
    your-image
```

### GPU Memory Limits (PyTorch)
```python
import torch

# Set max GPU memory
torch.cuda.set_per_process_memory_fraction(0.8, device=0)  # Use 80% of GPU memory
```

## Performance Tips

### 1. Use Mixed Precision Training
```python
# PyTorch
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. Optimize Batch Size
```bash
# Start with small batch and increase
# Monitor GPU memory with nvidia-smi
```

### 3. Use DataLoader Workers
```python
train_loader = DataLoader(
    dataset, 
    batch_size=32, 
    num_workers=4,  # Utilize multiple CPU cores
    pin_memory=True  # Faster data transfer to GPU
)
```

### 4. Clear GPU Cache
```python
import torch
torch.cuda.empty_cache()
```

## Security Best Practices

### 1. Use Secrets for Environment Variables
```bash
# Store in AWS Secrets Manager
aws secretsmanager create-secret \
    --name ml-inference-secrets \
    --secret-string '{"HF_TOKEN":"xxx","RUNPOD_ENDPOINT_ID":"yyy"}'

# Retrieve and use
```

### 2. Read-only Volumes
```bash
docker run --rm --gpus all \
    -v $(pwd)/data:/data:ro \
    your-image
```

### 3. Non-root User
```dockerfile
USER 1000:1000
```

## Troubleshooting

### Issue: Out of Memory
```bash
# Check memory usage
nvidia-smi

# Reduce batch size or model size
# Clear cache: torch.cuda.empty_cache()
```

### Issue: Container Can't Access GPU
```bash
# Verify Docker GPU config
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Check Docker daemon
sudo systemctl status docker

# Restart if needed
sudo systemctl restart docker
```

### Issue: Slow Performance
```bash
# Check GPU utilization
nvidia-smi dmon -s u

# Should be close to 100% during training
# If low, you may have CPU bottleneck
```

## Additional Resources

- [NVIDIA Container Toolkit Docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [PyTorch Docker Guide](https://github.com/pytorch/pytorch#docker-image)
- [TensorFlow Docker Guide](https://www.tensorflow.org/install/docker)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
