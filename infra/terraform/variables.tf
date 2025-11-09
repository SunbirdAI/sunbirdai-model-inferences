# General Configuration
variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# Security Group Variables
variable "security_group_name" {
  description = "Name of the security group"
  type        = string
  default     = "launch-wizard-8"
}

variable "security_group_description" {
  description = "Description of the security group"
  type        = string
  default     = "Security group for inference test server"
}

variable "vpc_id" {
  description = "VPC ID where resources will be created"
  type        = string
  default     = "vpc-92dd37ef"
}

variable "ingress_rules" {
  description = "List of ingress rules for the security group"
  type = list(object({
    from_port   = number
    to_port     = number
    protocol    = string
    cidr_blocks = list(string)
    description = string
  }))
  default = [
    {
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
      description = "SSH access"
    },
    {
      from_port   = 443
      to_port     = 443
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
      description = "HTTPS access"
    },
    {
      from_port   = 80
      to_port     = 80
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
      description = "HTTP access"
    }
  ]
}

# EC2 Instance Variables
variable "instance_name" {
  description = "Name tag for the EC2 instance"
  type        = string
  default     = "inference-test-server"
}

variable "ami_id" {
  description = "AMI ID for the EC2 instance (Amazon Linux 2023)"
  type        = string
  default     = "ami-0157af9aea2eef346"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g4dn.4xlarge"
}

variable "key_name" {
  description = "Name of the EC2 key pair"
  type        = string
  default     = "nwtbootcampkey-pem"
}

variable "root_volume_size" {
  description = "Size of the root volume in GB"
  type        = number
  default     = 250
}

variable "root_volume_type" {
  description = "Type of the root volume"
  type        = string
  default     = "gp3"
}

variable "root_volume_iops" {
  description = "IOPS for the root volume"
  type        = number
  default     = 3000
}

variable "root_volume_throughput" {
  description = "Throughput for the root volume in MB/s"
  type        = number
  default     = 125
}

variable "root_volume_encrypted" {
  description = "Whether to encrypt the root volume"
  type        = bool
  default     = false
}

variable "associate_public_ip" {
  description = "Whether to associate a public IP address"
  type        = bool
  default     = true
}

variable "metadata_http_tokens" {
  description = "Whether to require IMDSv2 (required or optional)"
  type        = string
  default     = "required"
}

variable "metadata_hop_limit" {
  description = "HTTP PUT response hop limit for instance metadata requests"
  type        = number
  default     = 2
}

variable "enable_dns_hostnames" {
  description = "Enable DNS hostnames for the instance"
  type        = bool
  default     = true
}

# User Data Variables
variable "enable_gpu_setup" {
  description = "Enable automatic GPU setup (Docker, CUDA, NVIDIA drivers)"
  type        = bool
  default     = true
}

variable "user_data" {
  description = "Custom user data script (only used if enable_gpu_setup is false)"
  type        = string
  default     = null
}

variable "user_data_replace_on_change" {
  description = "Whether to replace the instance when user_data changes"
  type        = bool
  default     = false
}
