# AWS GPU Instance Terraform Configuration

This Terraform configuration creates an Amazon Linux 2023 GPU instance (g4dn.4xlarge) on AWS with a security group, based on a CloudFormation template.

## Architecture

The configuration is organized into reusable modules:

```
terraform-gpu-instance/
├── main.tf                          # Root module orchestration
├── variables.tf                     # Root module variables
├── outputs.tf                       # Root module outputs
├── terraform.tfvars.example         # Example variable values
└── modules/
    ├── security_group/              # Security group module
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    └── ec2_instance/                # EC2 instance module
        ├── main.tf
        ├── variables.tf
        └── outputs.tf
```

## Features

- **Modular Design**: Reusable modules for security groups and EC2 instances
- **Automatic GPU Setup**: Optional user data script that installs Docker, CUDA, NVIDIA drivers, and NVIDIA Container Toolkit
- **Security Best Practices**:
  - IMDSv2 required by default
  - Separate security group rules for better management
  - Configurable ingress rules
- **Flexible Configuration**: All parameters can be customized via variables
- **GPU Support**: Configured for g4dn.4xlarge instance with NVIDIA T4 GPU
- **Storage**: 200GB gp3 volume with 3000 IOPS and 125 MB/s throughput

## Prerequisites

- [Terraform](https://www.terraform.io/downloads.html) >= 1.0
- AWS CLI configured with appropriate credentials
- An existing VPC (default: vpc-92dd37ef)
- An EC2 key pair (default: nwtbootcampkey-pem)

## Quick Start

1. **Clone or download this repository**

2. **Create a `terraform.tfvars` file** (copy from the example):
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

3. **Edit `terraform.tfvars`** with your specific values:
   ```hcl
   aws_region = "us-east-1"
   vpc_id     = "vpc-xxxxx"  # Your VPC ID
   key_name   = "your-key-name"
   
   # Restrict SSH access to your IP for better security
   ingress_rules = [
     {
       from_port   = 22
       to_port     = 22
       protocol    = "tcp"
       cidr_blocks = ["YOUR_IP/32"]
       description = "SSH from my IP"
     },
     # ... other rules
   ]
   ```

4. **Initialize Terraform**:
   ```bash
   terraform init
   ```

5. **Review the execution plan**:
   ```bash
   terraform plan
   ```

6. **Apply the configuration**:
   ```bash
   terraform apply
   ```

7. **Connect to your instance**:
   ```bash
   # Get the public IP from outputs
   terraform output instance_public_ip
   
   # SSH to the instance
   ssh -i /path/to/your-key.pem ec2-user@<public-ip>
   ```

8. **Wait for GPU setup to complete and reboot** (if `enable_gpu_setup = true`):
   ```bash
   # Monitor setup progress
   sudo tail -f /var/log/user-data.log
   
   # After setup completes, reboot for drivers to load
   sudo reboot
   
   # After reboot, verify GPU setup
   ./verify-gpu-setup.sh
   nvidia-smi
   ```

## GPU Setup

The configuration includes **automatic GPU setup** that installs:
- Docker Engine
- NVIDIA Drivers (Tesla T4 optimized)
- CUDA Toolkit
- NVIDIA Container Toolkit

### Enable/Disable GPU Setup

In `terraform.tfvars`:
```hcl
# Enable automatic GPU setup (default)
enable_gpu_setup = true

# Disable if you want manual setup
enable_gpu_setup = false
```

### After Instance Creation

1. **Wait for setup** (5-10 minutes) - monitor with:
   ```bash
   sudo tail -f /var/log/user-data.log
   ```

2. **Reboot** (REQUIRED):
   ```bash
   sudo reboot
   ```

3. **Verify**:
   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

For detailed GPU setup documentation, see **[USER_DATA_GUIDE.md](USER_DATA_GUIDE.md)**.

### Running GPU Containers

```bash
docker run --rm --gpus all -p 8088:8088 \
    -e HF_TOKEN \
    -e RUNPOD_ENDPOINT_ID \
    -e AUDIO_CONTENT_BUCKET_NAME \
    -e GCP_CREDENTIALS \
    sunbirddocker/sunbirdai-model-inferences:v2.3.11
```

## Configuration Variables

### Essential Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `aws_region` | AWS region to deploy resources | `us-east-1` |
| `vpc_id` | VPC ID where resources will be created | `vpc-92dd37ef` |
| `key_name` | EC2 key pair name | `nwtbootcampkey-pem` |
| `instance_type` | EC2 instance type | `g4dn.4xlarge` |
| `ami_id` | AMI ID for Amazon Linux 2023 | `ami-0157af9aea2eef346` |

### Security Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `security_group_name` | Name of the security group | `launch-wizard-8` |
| `ingress_rules` | List of ingress rules | SSH, HTTP, HTTPS from 0.0.0.0/0 |
| `metadata_http_tokens` | IMDSv2 requirement | `required` |

### Storage Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `root_volume_size` | Root volume size in GB | `200` |
| `root_volume_type` | Volume type | `gp3` |
| `root_volume_iops` | IOPS for the volume | `3000` |
| `root_volume_throughput` | Throughput in MB/s | `125` |

## Outputs

After applying, Terraform will output:

- `instance_id`: EC2 instance ID
- `instance_public_ip`: Public IP address
- `instance_private_ip`: Private IP address
- `instance_public_dns`: Public DNS name
- `security_group_id`: Security group ID

View all outputs:
```bash
terraform output
```

## Module Usage

### Using the Security Group Module Independently

```hcl
module "my_security_group" {
  source = "./modules/security_group"

  sg_name        = "my-custom-sg"
  sg_description = "My custom security group"
  vpc_id         = "vpc-xxxxx"
  
  ingress_rules = [
    {
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = ["10.0.0.0/8"]
      description = "SSH from internal network"
    }
  ]
  
  tags = {
    Environment = "Production"
  }
}
```

### Using the EC2 Instance Module Independently

```hcl
module "my_instance" {
  source = "./modules/ec2_instance"

  instance_name      = "my-gpu-instance"
  ami_id             = "ami-xxxxx"
  instance_type      = "g4dn.xlarge"
  key_name           = "my-key"
  security_group_ids = ["sg-xxxxx"]
  
  root_volume_size = 100
  
  tags = {
    Environment = "Production"
  }
}
```

## Security Considerations

⚠️ **Important Security Notes:**

1. **SSH Access**: The default configuration allows SSH from anywhere (0.0.0.0/0). For production, restrict this to your IP:
   ```hcl
   cidr_blocks = ["YOUR_IP/32"]
   ```

2. **IMDSv2**: The configuration requires IMDSv2 by default for enhanced security.

3. **Encryption**: Root volume encryption is disabled by default. Enable it for sensitive workloads:
   ```hcl
   root_volume_encrypted = true
   ```

## Cost Estimation

The g4dn.4xlarge instance costs approximately:
- **On-Demand**: ~$1.20/hour (~$876/month)
- **1-Year Reserved**: ~$0.72/hour (~$525/month)
- **3-Year Reserved**: ~$0.43/hour (~$314/month)

Storage (200GB gp3): ~$16/month

> **Note**: Prices vary by region. Check [AWS Pricing](https://aws.amazon.com/ec2/pricing/) for current rates.

## Maintenance

### Updating the Instance

1. Modify variables in `terraform.tfvars`
2. Run `terraform plan` to review changes
3. Run `terraform apply` to apply changes

### Destroying Resources

To delete all resources:
```bash
terraform destroy
```

## Troubleshooting

### Common Issues

1. **Key pair not found**: Ensure your key pair exists in the AWS region
2. **AMI not available**: The AMI might not be available in your region. Find the correct AMI ID:
   ```bash
   aws ec2 describe-images --owners amazon \
     --filters "Name=name,Values=al2023-ami-*-kernel-*-x86_64" \
     --query 'Images[*].[ImageId,Name,CreationDate]' \
     --output table
   ```
3. **VPC not found**: Verify your VPC ID exists in the specified region

## GPU Setup

After instance creation, you may need to install NVIDIA drivers and CUDA:

```bash
# Check for GPU
lspci | grep -i nvidia

# Install NVIDIA drivers (if not already installed)
sudo yum install -y gcc kernel-devel-$(uname -r)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-$distribution.repo
sudo mv cuda-$distribution.repo /etc/yum.repos.d/
sudo yum clean all
sudo yum -y install cuda-drivers

# Reboot
sudo reboot

# Verify installation
nvidia-smi
```

## License

This configuration is provided as-is for educational and production use.

## Contributing

Feel free to submit issues or pull requests to improve this configuration.
