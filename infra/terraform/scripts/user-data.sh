#!/bin/bash
# User Data Script for Amazon Linux 2023 GPU Instance Setup
# This script installs Docker, CUDA Toolkit, NVIDIA Drivers, and NVIDIA Container Toolkit
# Logs are written to /var/log/user-data.log

set -e  # Exit on error

# Log file
LOGFILE="/var/log/user-data.log"
exec > >(tee -a ${LOGFILE}) 2>&1

echo "=========================================="
echo "Starting GPU Instance Setup"
echo "Timestamp: $(date)"
echo "=========================================="

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Step 1: Updating system packages"
dnf update -y

log "Step 2: Installing Docker"
dnf install -y docker

log "Step 3: Starting Docker service"
systemctl start docker
systemctl enable docker

log "Step 4: Adding ec2-user to docker group"
usermod -a -G docker ec2-user

log "Step 5: Installing DKMS and kernel headers for NVIDIA drivers"
dnf install -y dkms kernel-devel-$(uname -r)

log "Step 6: Adding NVIDIA CUDA repository"
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo

log "Step 7: Installing NVIDIA drivers (open-dkms)"
dnf module enable -y nvidia-driver:open-dkms
dnf install -y nvidia-open

log "Step 8: Installing CUDA toolkit"
dnf install -y cuda-toolkit

log "Step 9: Setting up CUDA environment variables"
cat >> /etc/profile.d/cuda.sh << 'EOF'
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
EOF

# Make the script executable
chmod +x /etc/profile.d/cuda.sh

# Also add to ec2-user's bashrc
cat >> /home/ec2-user/.bashrc << 'EOF'
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
EOF

log "Step 10: Installing NVIDIA Container Toolkit"
# Get distribution info
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

# Add NVIDIA Container Toolkit repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Install the toolkit
dnf clean expire-cache
dnf install -y nvidia-container-toolkit

log "Step 11: Configuring Docker to use NVIDIA runtime"
nvidia-ctk runtime configure --runtime=docker

log "Step 12: Restarting Docker service"
systemctl restart docker

log "Step 13: Creating verification script"
cat > /home/ec2-user/verify-gpu-setup.sh << 'EOF'
#!/bin/bash
echo "=========================================="
echo "GPU Setup Verification"
echo "=========================================="

echo ""
echo "1. Checking NVIDIA Driver:"
nvidia-smi || echo "NVIDIA driver check failed"

echo ""
echo "2. Checking CUDA Compiler:"
nvcc -V || echo "CUDA compiler check failed"

echo ""
echo "3. Checking Docker:"
docker --version || echo "Docker check failed"

echo ""
echo "4. Checking Docker GPU Access:"
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi || echo "Docker GPU access check failed"

echo ""
echo "=========================================="
echo "Verification Complete"
echo "=========================================="
EOF

chmod +x /home/ec2-user/verify-gpu-setup.sh
chown ec2-user:ec2-user /home/ec2-user/verify-gpu-setup.sh

log "Step 14: Creating helper scripts"
# Script to run Sunbird AI model
cat > /home/ec2-user/run-sunbird-model.sh << 'EOF'
#!/bin/bash
# Helper script to run Sunbird AI model with GPU support
# Usage: ./run-sunbird-model.sh

# Check if environment variables are set
if [ -z "$HF_TOKEN" ] || [ -z "$RUNPOD_ENDPOINT_ID" ] || [ -z "$AUDIO_CONTENT_BUCKET_NAME" ] || [ -z "$GCP_CREDENTIALS" ]; then
    echo "WARNING: Some environment variables are not set."
    echo "Required: HF_TOKEN, RUNPOD_ENDPOINT_ID, AUDIO_CONTENT_BUCKET_NAME, GCP_CREDENTIALS"
    echo "Set them in ~/.bashrc or export them before running this script."
fi

echo "Starting Sunbird AI Model with GPU support..."
docker run --rm --gpus all -p 8088:8088 \
    -e HF_TOKEN \
    -e RUNPOD_ENDPOINT_ID \
    -e AUDIO_CONTENT_BUCKET_NAME \
    -e GCP_CREDENTIALS \
    sunbirddocker/sunbirdai-model-inferences:v2.3.11
EOF

chmod +x /home/ec2-user/run-sunbird-model.sh
chown ec2-user:ec2-user /home/ec2-user/run-sunbird-model.sh

# Create environment template
cat > /home/ec2-user/.env.template << 'EOF'
# Environment Variables Template
# Copy this file to ~/.bashrc and fill in the values

export HF_TOKEN="your-huggingface-token"
export RUNPOD_ENDPOINT_ID="your-runpod-endpoint-id"
export AUDIO_CONTENT_BUCKET_NAME="your-bucket-name"
export GCP_CREDENTIALS="your-gcp-credentials"
EOF

chown ec2-user:ec2-user /home/ec2-user/.env.template

log "Step 15: Creating README for ec2-user"
cat > /home/ec2-user/README.txt << 'EOF'
========================================
GPU Instance Setup Complete!
========================================

This instance has been configured with:
- Docker
- NVIDIA Drivers (open-dkms)
- CUDA Toolkit
- NVIDIA Container Toolkit

Quick Start:
------------

1. Verify GPU Setup:
   ./verify-gpu-setup.sh

2. Check NVIDIA Driver:
   nvidia-smi

3. Check CUDA:
   nvcc -V

4. Test Docker GPU Access:
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

5. Run Sunbird AI Model:
   - First, set your environment variables in ~/.bashrc
   - See .env.template for required variables
   - Then run: ./run-sunbird-model.sh

Logs:
-----
- User data execution log: /var/log/user-data.log
- Cloud-init logs: /var/log/cloud-init-output.log

Troubleshooting:
----------------
If GPU is not detected:
1. Check logs: sudo cat /var/log/user-data.log
2. Verify driver: nvidia-smi
3. Restart Docker: sudo systemctl restart docker
4. Reboot if needed: sudo reboot

For more information, see the documentation.
EOF

chown ec2-user:ec2-user /home/ec2-user/README.txt

log "Step 16: Setting up status marker"
# Create a marker file to indicate setup completion
touch /var/log/user-data-complete
echo "$(date)" > /var/log/user-data-complete

log "=========================================="
log "GPU Instance Setup Complete!"
log "A reboot is recommended for all changes to take effect."
log "Users can verify setup by running: /home/ec2-user/verify-gpu-setup.sh"
log "=========================================="

# Optional: Uncomment the line below to automatically reboot after setup
# shutdown -r +1 "System will reboot in 1 minute to complete GPU setup"
