# GPU Setup Guide

This guide explains the automated GPU setup process for your Amazon Linux 2023 instance with NVIDIA Tesla T4 GPU.

## Overview

The Terraform configuration includes an automated user data script that configures your GPU instance with:
- **Docker** - Container runtime
- **NVIDIA Drivers** - Tesla T4 optimized open-source drivers
- **CUDA Toolkit** - GPU computing platform
- **NVIDIA Container Toolkit** - GPU support for Docker containers

## Automatic Setup (Default)

By default, the GPU setup script runs automatically when the instance is created.

### What Gets Installed

1. **Docker Engine** - Latest version from Amazon Linux repositories
2. **NVIDIA Drivers** - Open DKMS module for Tesla T4
3. **CUDA Toolkit** - Complete CUDA development environment
4. **NVIDIA Container Toolkit** - Enables GPU access in Docker containers

### Setup Process

The user data script (`scripts/user-data.sh`) automatically:

1. Updates system packages
2. Installs and configures Docker
3. Installs DKMS and kernel headers
4. Adds NVIDIA CUDA repository
5. Installs NVIDIA drivers
6. Installs CUDA toolkit
7. Sets up environment variables
8. Installs NVIDIA Container Toolkit
9. Configures Docker for GPU support
10. Creates helper scripts for verification

## Usage

### Enable/Disable Automatic Setup

In `terraform.tfvars`:
```hcl
# Enable automatic GPU setup (default)
enable_gpu_setup = true

# Disable if you want to set up manually
enable_gpu_setup = false
```

### After Instance Creation

1. **Wait for setup to complete** (5-10 minutes)
   - User data script runs in the background
   - Check progress: `sudo tail -f /var/log/user-data.log`

2. **Reboot the instance** (REQUIRED for drivers to load)
   ```bash
   sudo reboot
   ```

3. **Verify GPU setup** (after reboot)
   ```bash
   ./verify-gpu-setup.sh
   ```

## Verification Scripts

The setup creates helpful scripts in `/home/ec2-user/`:

### 1. verify-gpu-setup.sh
Comprehensive verification of GPU, CUDA, and Docker GPU access.

```bash
./verify-gpu-setup.sh
```

**Expected Output:**
```
==========================================
GPU Setup Verification
==========================================

1. NVIDIA Driver Version:
550.90.07

2. CUDA Version:
release 12.4, V12.4.131

3. Docker Status:
active

4. Docker GPU Test:
[nvidia-smi output showing GPU info]

5. NVIDIA System Management Interface:
[Full nvidia-smi output]
```

### 2. run-gpu-container.sh
Helper script to quickly test GPU-enabled containers.

```bash
# Test with default NVIDIA CUDA image
./run-gpu-container.sh

# Test with custom image
./run-gpu-container.sh your-image:tag
```

### 3. GPU-SETUP-README.txt
Quick reference guide created on the instance.

```bash
cat ~/GPU-SETUP-README.txt
```

## Manual Commands

### Check NVIDIA Driver
```bash
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07    Driver Version: 550.90.07    CUDA Version: 12.4   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   30C    P8    15W /  70W |      0MiB / 15360MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### Check CUDA Version
```bash
nvcc --version
```

### Test Docker GPU Access
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Running Your GPU Application

### Basic Docker Run
```bash
docker run --rm --gpus all -p 8088:8088 \
    your-image:tag
```

### With Environment Variables
```bash
docker run --rm --gpus all -p 8088:8088 \
    -e HF_TOKEN \
    -e RUNPOD_ENDPOINT_ID \
    -e AUDIO_CONTENT_BUCKET_NAME \
    -e GCP_CREDENTIALS \
    sunbirddocker/sunbirdai-model-inferences:v2.3.11
```

### Using Docker Compose
```yaml
version: '3.8'

services:
  ml-service:
    image: sunbirddocker/sunbirdai-model-inferences:v2.3.11
    ports:
      - "8088:8088"
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - RUNPOD_ENDPOINT_ID=${RUNPOD_ENDPOINT_ID}
      - AUDIO_CONTENT_BUCKET_NAME=${AUDIO_CONTENT_BUCKET_NAME}
      - GCP_CREDENTIALS=${GCP_CREDENTIALS}
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
docker-compose up -d
```

## Environment Variables

The setup automatically configures:

```bash
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

These are set in `/etc/profile.d/cuda.sh` and loaded on login.

## Troubleshooting

### Issue: nvidia-smi not found after setup

**Cause:** Instance needs reboot to load drivers

**Solution:**
```bash
sudo reboot
```

### Issue: Docker cannot access GPU

**Symptom:**
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```

**Solution:**
1. Check Docker daemon config:
   ```bash
   cat /etc/docker/daemon.json
   ```
   Should contain:
   ```json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```

2. Restart Docker:
   ```bash
   sudo systemctl restart docker
   ```

3. Verify:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

### Issue: CUDA not found

**Solution:**
Source the environment:
```bash
source /etc/profile.d/cuda.sh
# Or logout and login again
```

### Issue: Setup failed

**Check logs:**
```bash
sudo cat /var/log/user-data.log
```

**Common causes:**
- Network connectivity issues during package download
- Insufficient disk space
- Wrong instance type (non-GPU instance)

## Manual Installation (If Auto-Setup Disabled)

If you set `enable_gpu_setup = false`, follow these manual steps:

### 1. Update System
```bash
sudo dnf update -y
sudo reboot
```

### 2. Install Docker
```bash
sudo dnf install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user
newgrp docker
```

### 3. Install NVIDIA Drivers
```bash
sudo dnf install -y dkms kernel-devel-$(uname -r)
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo
sudo dnf module enable -y nvidia-driver:open-dkms
sudo dnf install -y nvidia-open
```

### 4. Install CUDA Toolkit
```bash
sudo dnf install -y cuda-toolkit
```

### 5. Setup Environment
```bash
echo 'export PATH=$PATH:/usr/local/cuda/bin' | sudo tee -a /etc/profile
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' | sudo tee -a /etc/profile
source /etc/profile
```

### 6. Install NVIDIA Container Toolkit
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf clean expire-cache
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 7. Reboot and Verify
```bash
sudo reboot
# After reboot
nvidia-smi
nvcc --version
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Customizing the User Data Script

To modify the setup script:

1. Edit `scripts/user-data.sh`
2. Add your custom configuration
3. Run `terraform apply` to update

Example customizations:
```bash
# Add custom Docker images
docker pull your-custom-image:latest

# Install additional tools
dnf install -y git tmux htop

# Clone your repository
cd /home/ec2-user
git clone https://github.com/your-org/your-repo.git
chown -R ec2-user:ec2-user your-repo
```

## Performance Considerations

### GPU Memory
Tesla T4 has **15.36 GB** of GPU memory. Monitor usage:
```bash
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
```

### CPU and RAM
g4dn.4xlarge specifications:
- **16 vCPUs**
- **64 GB RAM**
- **1x NVIDIA T4 GPU**
- **300 GB NVMe SSD** (local instance storage)

### Docker Container Limits
Set memory limits for containers:
```bash
docker run --rm --gpus all \
  --memory="32g" \
  --memory-swap="32g" \
  -p 8088:8088 \
  your-image:tag
```

## Security Best Practices

1. **Restrict GPU access** to specific containers:
   ```bash
   docker run --rm --gpus '"device=0"' your-image:tag
   ```

2. **Use resource limits** to prevent resource exhaustion

3. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Keep drivers updated**:
   ```bash
   sudo dnf update nvidia-open cuda-toolkit
   ```

## Cost Optimization

**Running costs for g4dn.4xlarge:**
- ~$1.20/hour when running
- Stop instance when not in use to save costs

**Stop instance:**
```bash
# From your local machine
aws ec2 stop-instances --instance-ids $(terraform output -raw instance_id)
```

**Start instance:**
```bash
aws ec2 start-instances --instance-ids $(terraform output -raw instance_id)
```

## Additional Resources

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA Container Toolkit Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
- [Docker GPU Documentation](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [AWS GPU Instances](https://aws.amazon.com/ec2/instance-types/g4/)

## Support

If you encounter issues:

1. Check logs: `sudo cat /var/log/user-data.log`
2. Verify instance type is GPU-enabled (g4dn.*)
3. Ensure reboot was performed after setup
4. Review this guide's troubleshooting section
