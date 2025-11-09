# Getting Started with GPU-Enabled Terraform Infrastructure

## 🎉 What's New: Automated GPU Setup

Your Terraform configuration now includes **automated GPU setup** that eliminates manual configuration steps!

### What Gets Installed Automatically

When you deploy the instance, the user data script automatically installs:

1. **Docker Engine** - Latest version from Amazon Linux repos
2. **NVIDIA Drivers** - Tesla T4 optimized (open-source driver)
3. **CUDA Toolkit** - Version 12.4+
4. **NVIDIA Container Toolkit** - Enables GPU access in Docker containers

### Helper Scripts Created

The setup also creates helpful scripts on your instance:

- `verify-gpu-setup.sh` - Comprehensive GPU verification
- `run-gpu-container.sh` - Quick GPU container testing
- `GPU-SETUP-README.txt` - Quick reference guide

## 🚀 Complete Deployment Flow

### Step 1: Configure Terraform

```bash
cd terraform-gpu-instance
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your values:
```hcl
vpc_id   = "vpc-xxxxx"        # Your VPC ID
key_name = "your-key-name"    # Your EC2 key pair

# GPU setup is enabled by default
enable_gpu_setup = true
```

### Step 2: Deploy Infrastructure

```bash
terraform init
terraform plan
terraform apply
```

This creates:
- Security Group (SSH, HTTP, HTTPS)
- EC2 Instance (g4dn.4xlarge with Tesla T4 GPU)
- 200GB gp3 storage
- Starts GPU setup in background

### Step 3: Monitor GPU Setup

```bash
# Get instance IP
export INSTANCE_IP=$(terraform output -raw instance_public_ip)

# SSH to instance
ssh -i your-key.pem ec2-user@$INSTANCE_IP

# Monitor setup progress (takes 5-10 minutes)
sudo tail -f /var/log/user-data.log
```

You'll see output like:
```
[1/9] Updating system packages...
[2/9] Installing Docker...
[3/9] Starting Docker service...
[4/9] Adding ec2-user to docker group...
[5/9] Installing DKMS and kernel headers...
[6/9] Adding NVIDIA CUDA repository...
[7/9] Installing NVIDIA driver...
[8/9] Installing CUDA toolkit...
[9/9] Setting up CUDA environment variables...
[10/12] Installing NVIDIA Container Toolkit...
[11/12] Configuring Docker for NVIDIA runtime...
[12/12] Restarting Docker service...
```

### Step 4: Reboot (REQUIRED)

After setup completes:
```bash
sudo reboot
```

**Why?** The NVIDIA kernel modules need a reboot to load properly.

### Step 5: Verify Everything Works

After reboot, SSH back and verify:

```bash
# Verify GPU
./verify-gpu-setup.sh

# Or manually check
nvidia-smi
nvcc --version

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

Expected `nvidia-smi` output:
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

## 🎯 Running Your First GPU Workload

### Example: Sunbird AI Inference Service

```bash
# Set environment variables
export HF_TOKEN="your-token"
export RUNPOD_ENDPOINT_ID="your-endpoint"
export AUDIO_CONTENT_BUCKET_NAME="your-bucket"
export GCP_CREDENTIALS="your-credentials"

# Run the service
docker run --rm --gpus all -p 8088:8088 \
    -e HF_TOKEN \
    -e RUNPOD_ENDPOINT_ID \
    -e AUDIO_CONTENT_BUCKET_NAME \
    -e GCP_CREDENTIALS \
    sunbirddocker/sunbirdai-model-inferences:v2.3.11
```

Service will be available at: `http://<instance-ip>:8088`

### Example: PyTorch Quick Test

```bash
docker run --rm --gpus all -it pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

Expected output:
```
CUDA available: True
GPU name: Tesla T4
```

## 📁 Project Structure

```
terraform-gpu-instance/
├── main.tf                      # Root orchestration
├── variables.tf                 # Configuration parameters
├── outputs.tf                   # Resource outputs
├── terraform.tfvars.example     # Configuration template
├── Makefile                     # Quick commands
│
├── scripts/
│   └── user-data.sh            # GPU setup automation
│
├── modules/
│   ├── security_group/         # Security group module
│   └── ec2_instance/          # EC2 instance module
│
└── Documentation/
    ├── README.md               # Complete guide
    ├── QUICKSTART.md          # 5-minute setup
    ├── GPU-SETUP-GUIDE.md     # GPU setup details
    ├── USAGE-EXAMPLES.md      # ML workload examples
    ├── ARCHITECTURE.md        # System diagrams
    └── CLOUDFORMATION_VS_TERRAFORM.md
```

## 🛠️ Common Operations

### View Instance Details
```bash
terraform output
```

### Stop Instance (Save Costs)
```bash
aws ec2 stop-instances --instance-ids $(terraform output -raw instance_id)
```

### Start Instance
```bash
aws ec2 start-instances --instance-ids $(terraform output -raw instance_id)
```

### Update Configuration
```bash
# Edit terraform.tfvars
vim terraform.tfvars

# Apply changes
terraform apply
```

### Destroy Everything
```bash
terraform destroy
```

## 📚 Documentation Guide

Read in this order for best learning experience:

1. **This File** (GETTING-STARTED.md) - You are here! ✓
2. **QUICKSTART.md** - Fast deployment checklist
3. **GPU-SETUP-GUIDE.md** - Deep dive into GPU configuration
4. **USAGE-EXAMPLES.md** - 10+ practical examples
5. **README.md** - Complete feature reference
6. **ARCHITECTURE.md** - System design and flow diagrams

## 🔧 Troubleshooting

### Issue: Setup Taking Too Long

**Check progress:**
```bash
sudo tail -f /var/log/user-data.log
```

**Common causes:**
- Large package downloads (CUDA toolkit is ~3GB)
- Network connectivity issues
- Repository mirrors being slow

### Issue: nvidia-smi Not Found

**Solution:** Reboot required
```bash
sudo reboot
```

### Issue: Docker Can't Access GPU

**Check Docker config:**
```bash
cat /etc/docker/daemon.json
```

**Should contain:**
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

**Fix:**
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Issue: Out of GPU Memory

**Monitor usage:**
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

**Solutions:**
- Reduce batch size
- Use smaller model
- Clear cache: `torch.cuda.empty_cache()`

## 💰 Cost Management

### Running Costs
- **g4dn.4xlarge**: ~$1.20/hour = ~$876/month (24/7)
- **200GB gp3**: ~$16/month
- **Total**: ~$892/month for continuous operation

### Save Money
```bash
# Stop when not in use (only pay for storage)
aws ec2 stop-instances --instance-ids $(terraform output -raw instance_id)

# Start when needed
aws ec2 start-instances --instance-ids $(terraform output -raw instance_id)

# Or use spot instances (edit instance_type in config)
```

## 🔒 Security Best Practices

### 1. Restrict SSH Access

In `terraform.tfvars`:
```hcl
ingress_rules = [
  {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["YOUR_IP/32"]  # Your IP only
    description = "SSH from my IP"
  },
  # Keep other rules...
]
```

### 2. Use Secrets Manager

```bash
# Store secrets in AWS Secrets Manager
aws secretsmanager create-secret \
    --name ml-inference-secrets \
    --secret-string file://secrets.json
```

### 3. Enable Volume Encryption

In `terraform.tfvars`:
```hcl
root_volume_encrypted = true
```

## ✨ Key Features Summary

✅ **Automated Setup** - No manual GPU configuration needed  
✅ **Modular Design** - Reusable Terraform modules  
✅ **Production Ready** - Security best practices built-in  
✅ **Well Documented** - 6 comprehensive guides  
✅ **Helper Scripts** - Verification and testing tools included  
✅ **Docker Ready** - GPU access configured for containers  
✅ **Cost Aware** - Easy stop/start for cost savings  

## 🎓 Next Steps

Choose your path:

**For Quick Deployment:**
→ Follow QUICKSTART.md for minimal steps

**For Understanding GPU Setup:**
→ Read GPU-SETUP-GUIDE.md for details

**For Running ML Workloads:**
→ See USAGE-EXAMPLES.md for 10+ examples

**For Complete Reference:**
→ Read README.md for all features

## 🆘 Getting Help

1. Check troubleshooting sections in each guide
2. Review logs: `sudo cat /var/log/user-data.log`
3. Verify GPU: `./verify-gpu-setup.sh`
4. Test Docker GPU: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`

## 🎉 Success Checklist

After setup, you should have:

- [ ] Instance running and accessible via SSH
- [ ] `nvidia-smi` showing Tesla T4 GPU
- [ ] `nvcc --version` showing CUDA 12.4+
- [ ] Docker running: `docker ps`
- [ ] GPU accessible in Docker: Test passed
- [ ] Helper scripts present: `ls ~/*.sh`
- [ ] Application running on port 8088

**All checked?** You're ready to run GPU workloads! 🚀

---

**Welcome to your automated GPU infrastructure!** 🎊
