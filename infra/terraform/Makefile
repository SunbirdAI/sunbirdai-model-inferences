.PHONY: help init plan apply destroy validate fmt clean

# Default target
help:
	@echo "Available targets:"
	@echo "  init      - Initialize Terraform"
	@echo "  plan      - Show what Terraform will do"
	@echo "  apply     - Apply Terraform configuration"
	@echo "  destroy   - Destroy all resources"
	@echo "  validate  - Validate Terraform configuration"
	@echo "  fmt       - Format Terraform files"
	@echo "  clean     - Remove Terraform state and cache"
	@echo "  ssh       - SSH into the instance"
	@echo "  output    - Show all outputs"

# Initialize Terraform
init:
	terraform init

# Plan changes
plan:
	terraform plan

# Apply changes
apply:
	terraform apply

# Apply without confirmation (use with caution)
apply-auto:
	terraform apply -auto-approve

# Destroy resources
destroy:
	terraform destroy

# Destroy without confirmation (use with caution)
destroy-auto:
	terraform destroy -auto-approve

# Validate configuration
validate:
	terraform validate

# Format all Terraform files
fmt:
	terraform fmt -recursive

# Check formatting
fmt-check:
	terraform fmt -recursive -check

# Clean up Terraform files
clean:
	rm -rf .terraform
	rm -f .terraform.lock.hcl
	rm -f terraform.tfstate*
	rm -f crash.log

# Show all outputs
output:
	terraform output

# Get instance public IP
ip:
	@terraform output -raw instance_public_ip

# SSH into the instance
ssh:
	@echo "Connecting to instance..."
	@ssh -i ~/.ssh/nwtbootcampkey-pem.pem ec2-user@$$(terraform output -raw instance_public_ip)

# Create terraform.tfvars from example
setup:
	@if [ ! -f terraform.tfvars ]; then \
		cp terraform.tfvars.example terraform.tfvars; \
		echo "Created terraform.tfvars from example. Please edit it with your values."; \
	else \
		echo "terraform.tfvars already exists."; \
	fi
