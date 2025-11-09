terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.20.0"
    }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = "sunbirdai"
}

# Security Group Module
module "security_group" {
  source = "./modules/security_group"

  sg_name        = var.security_group_name
  sg_description = var.security_group_description
  vpc_id         = var.vpc_id
  ingress_rules  = var.ingress_rules
  tags           = var.common_tags
}

# EC2 Instance Module
module "ec2_instance" {
  source = "./modules/ec2_instance"

  instance_name               = var.instance_name
  ami_id                      = var.ami_id
  instance_type               = var.instance_type
  key_name                    = var.key_name
  user_data                   = var.enable_gpu_setup ? file("${path.module}/scripts/user-data.sh") : var.user_data
  user_data_replace_on_change = var.user_data_replace_on_change
  security_group_ids          = [module.security_group.security_group_id]
  root_volume_size            = var.root_volume_size
  root_volume_type            = var.root_volume_type
  root_volume_iops            = var.root_volume_iops
  root_volume_throughput      = var.root_volume_throughput
  root_volume_encrypted       = var.root_volume_encrypted
  associate_public_ip         = var.associate_public_ip
  metadata_http_tokens        = var.metadata_http_tokens
  metadata_hop_limit          = var.metadata_hop_limit
  enable_dns_hostnames        = var.enable_dns_hostnames
  tags                        = var.common_tags
}
