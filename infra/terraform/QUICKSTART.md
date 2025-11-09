# Quick Start Guide

Get your GPU instance running in 5 minutes!

## Prerequisites Checklist

- [ ] Terraform installed (`terraform --version`)
- [ ] AWS CLI configured (`aws configure`)
- [ ] EC2 key pair created in AWS
- [ ] VPC ID noted down

## 3-Step Deployment

### Step 1: Configure
```bash
# Copy example configuration
cp terraform.tfvars.example terraform.tfvars

# Edit with your values (required changes marked with ⚠️)
nano terraform.tfvars
```

**Required Changes:**
```hcl
⚠️ vpc_id  = "vpc-XXXXXXXXX"      # Your VPC ID
⚠️ key_name = "your-key-name"      # Your EC2 key pair name

# Optional but recommended for security:
ingress_rules = [
  {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["YOUR_IP/32"]   # Your IP address
    description = "SSH from my IP"
  },
  # Keep other rules as-is
]
```

### Step 2: Deploy
```bash
# Initialize Terraform
terraform init

# Deploy (review plan and type 'yes')
terraform apply
```

### Step 3: Connect
```bash
# Get your instance IP
terraform output instance_public_ip

# SSH to your instance
ssh -i /path/to/your-key.pem ec2-user@<instance-ip>
```

### Step 4: Wait for GPU Setup & Reboot
```bash
# Monitor setup progress (takes 5-10 minutes)
sudo tail -f /var/log/user-data.log

# When setup completes, reboot (REQUIRED for drivers)
sudo reboot

# After reboot, SSH back and verify
./verify-gpu-setup.sh
nvidia-smi
```

## Using the Makefile (Even Faster!)

```bash
# One-time setup
make setup    # Creates terraform.tfvars from example
# Edit terraform.tfvars with your values

# Deploy
make init     # Initialize
make apply    # Deploy (will ask for confirmation)

# Get outputs
make ip       # Show public IP
make output   # Show all outputs

# Connect
make ssh      # Automatically SSH to instance

# Cleanup
make destroy  # Remove all resources
```

## What Gets Created

```
┌─────────────────────────────────┐
│  VPC: vpc-92dd37ef              │
│                                  │
│  ┌──────────────────────────┐  │
│  │ Security Group           │  │
│  │ - SSH (22)               │  │
│  │ - HTTP (80)              │  │
│  │ - HTTPS (443)            │  │
│  └──────────────────────────┘  │
│           │                      │
│           ▼                      │
│  ┌──────────────────────────┐  │
│  │ EC2 Instance             │  │
│  │ - g4dn.4xlarge           │  │
│  │ - NVIDIA T4 GPU          │  │
│  │ - 200GB gp3 Storage      │  │
│  │ - Public IP              │  │
│  │                          │  │
│  │ Auto-installed:          │  │
│  │ ✓ Docker                 │  │
│  │ ✓ NVIDIA Drivers         │  │
│  │ ✓ CUDA Toolkit           │  │
│  │ ✓ Container Toolkit      │  │
│  └──────────────────────────┘  │
└─────────────────────────────────┘
```

## Verify GPU

After connecting via SSH:
```bash
# Check for GPU
lspci | grep -i nvidia

# If NVIDIA drivers are installed
nvidia-smi
```

## Common Issues & Fixes

### Issue: "VPC not found"
```bash
# Get your VPC ID
aws ec2 describe-vpcs --query 'Vpcs[*].[VpcId,Tags[?Key==`Name`].Value|[0]]' --output table

# Update terraform.tfvars with correct VPC ID
```

### Issue: "Key pair not found"
```bash
# List your key pairs
aws ec2 describe-key-pairs --query 'KeyPairs[*].KeyName' --output table

# Update terraform.tfvars with correct key name
```

### Issue: "AMI not found in region"
```bash
# Find Amazon Linux 2023 AMI in your region
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=al2023-ami-*-kernel-*-x86_64" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].[ImageId,Name]' \
  --output table

# Update ami_id in terraform.tfvars
```

## Cost Alert 💰

Running this instance 24/7 costs approximately:
- **~$1.20/hour** = **~$876/month**

**To save money:**
```bash
# Stop instance when not in use
aws ec2 stop-instances --instance-ids $(terraform output -raw instance_id)

# Start when needed
aws ec2 start-instances --instance-ids $(terraform output -raw instance_id)

# Or destroy completely
make destroy
```

## Next Steps

1. **Install NVIDIA Drivers** (if not pre-installed):
   ```bash
   # See README.md "GPU Setup" section
   ```

2. **Install ML Frameworks**:
   ```bash
   # PyTorch
   pip3 install torch torchvision torchaudio

   # TensorFlow
   pip3 install tensorflow
   ```

3. **Setup Development Environment**:
   ```bash
   # Install common tools
   sudo yum install -y git tmux htop
   
   # Clone your project
   git clone <your-repo-url>
   ```

## Quick Commands Reference

```bash
# Deployment
make init          # Initialize Terraform
make plan          # Preview changes
make apply         # Deploy infrastructure
make destroy       # Remove all resources

# Information
make output        # Show all outputs
make ip            # Show public IP only

# Connection
make ssh           # SSH to instance

# Maintenance
make validate      # Validate configuration
make fmt           # Format Terraform files
make clean         # Clean local files
```

## Getting Help

1. Check `README.md` for detailed documentation
2. Review `ARCHITECTURE.md` for system design
3. See `CLOUDFORMATION_VS_TERRAFORM.md` for migration details

## Security Reminder 🔒

Before production use:
- [ ] Restrict SSH to your IP only
- [ ] Enable encryption (`root_volume_encrypted = true`)
- [ ] Review security group rules
- [ ] Use AWS Secrets Manager for sensitive data
- [ ] Enable CloudWatch monitoring

## Cleanup

When you're done:
```bash
# Remove all resources to stop charges
make destroy

# Or via Terraform directly
terraform destroy
```

---

**Need help?** Check the main `README.md` for comprehensive documentation.
