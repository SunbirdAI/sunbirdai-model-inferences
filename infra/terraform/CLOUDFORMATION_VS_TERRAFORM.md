# CloudFormation vs Terraform Comparison

## Overview

This document compares the original CloudFormation template with the Terraform implementation.

## Key Differences

### 1. Resource Definition

**CloudFormation:**
```yaml
Resources:
  LaunchWizard8SecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupName: launch-wizard-8
      # ...
```

**Terraform:**
```hcl
resource "aws_security_group" "this" {
  name        = var.sg_name
  description = var.sg_description
  # ...
}
```

### 2. Modularity

| Feature | CloudFormation | Terraform |
|---------|----------------|-----------|
| **Modularity** | Limited (Nested Stacks) | Excellent (Built-in Modules) |
| **Reusability** | Stack Sets, Nested Stacks | Native Module System |
| **Code Organization** | Single or nested templates | Multiple files and directories |

**CloudFormation:** Single template file with all resources.

**Terraform:** Organized into reusable modules:
- `modules/security_group/` - Reusable security group module
- `modules/ec2_instance/` - Reusable EC2 instance module

### 3. Variables and Parameterization

**CloudFormation (if using parameters):**
```yaml
Parameters:
  InstanceType:
    Type: String
    Default: g4dn.4xlarge
```

**Terraform:**
```hcl
variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g4dn.4xlarge"
}
```

### 4. Outputs

**CloudFormation:**
```yaml
Outputs:
  InstanceId:
    Description: The Instance ID
    Value: !Ref InferenceTestServer
```

**Terraform:**
```hcl
output "instance_id" {
  description = "ID of the EC2 instance"
  value       = module.ec2_instance.instance_id
}

# Plus many more outputs: public_ip, private_ip, dns, etc.
```

### 5. Security Group Rules

**CloudFormation:** Inline rules within SecurityGroup resource.

**Terraform:** Separate `aws_security_group_rule` resources for better management:
```hcl
resource "aws_security_group_rule" "ingress" {
  count             = length(var.ingress_rules)
  type              = "ingress"
  from_port         = var.ingress_rules[count.index].from_port
  # ...
}
```

## Feature Comparison

| Feature | CloudFormation | Terraform |
|---------|----------------|-----------|
| **State Management** | Managed by AWS | Local or remote (S3, Terraform Cloud) |
| **Syntax** | YAML/JSON | HCL (HashiCorp Configuration Language) |
| **Provider Support** | AWS Only | Multi-cloud (AWS, Azure, GCP, etc.) |
| **Preview Changes** | Change Sets | `terraform plan` |
| **Module Registry** | AWS Serverless Application Repository | Terraform Registry |
| **Drift Detection** | CloudFormation Drift Detection | `terraform plan` |
| **Import Existing Resources** | Limited | Excellent |

## Advantages of This Terraform Implementation

### 1. Enhanced Modularity
```
terraform-gpu-instance/
├── main.tf                    # Orchestration
├── modules/
│   ├── security_group/        # Reusable SG module
│   └── ec2_instance/          # Reusable EC2 module
```

### 2. Better Variable Management
- Strongly typed variables with validation
- Separate `terraform.tfvars` for environment-specific values
- Clear variable descriptions and defaults

### 3. Comprehensive Outputs
The Terraform version provides more outputs:
- Instance ID
- Public IP
- Private IP
- Public DNS
- Private DNS
- Instance State
- Availability Zone
- Security Group ID

### 4. Improved Security
- IMDSv2 required by default
- Separate egress rules
- Easy to restrict SSH to specific IPs

### 5. Development Workflow
- `Makefile` for common operations
- `.gitignore` for sensitive files
- Comprehensive documentation

## Migration Benefits

### From CloudFormation to Terraform

✅ **Multi-cloud capability** - Same syntax for AWS, Azure, GCP  
✅ **Better module system** - Reuse across projects  
✅ **Larger community** - More modules and examples  
✅ **Better state management** - More control over state files  
✅ **Plan before apply** - See changes before applying  
✅ **Easier testing** - Better tooling for testing IaC  

### Trade-offs

❌ **AWS Integration** - CloudFormation has tighter AWS integration  
❌ **Learning Curve** - New syntax to learn  
❌ **State Management** - Need to manage state files  

## Resource Mapping

| CloudFormation | Terraform | Notes |
|----------------|-----------|-------|
| `AWS::EC2::SecurityGroup` | `aws_security_group` | + `aws_security_group_rule` |
| `AWS::EC2::Instance` | `aws_instance` | Similar properties |
| `!Ref` | `module.<name>.<output>` | Reference other resources |
| `!GetAtt` | `.attribute` | Get resource attributes |
| `Parameters` | `variable` | Input variables |
| `Outputs` | `output` | Output values |

## Execution Comparison

### CloudFormation
```bash
# Create stack
aws cloudformation create-stack \
  --stack-name gpu-instance \
  --template-body file://template.yaml

# Update stack
aws cloudformation update-stack \
  --stack-name gpu-instance \
  --template-body file://template.yaml

# Delete stack
aws cloudformation delete-stack \
  --stack-name gpu-instance
```

### Terraform
```bash
# Initialize
terraform init

# Plan changes
terraform plan

# Apply changes
terraform apply

# Destroy resources
terraform destroy
```

Or using the Makefile:
```bash
make init
make plan
make apply
make destroy
```

## Best Practices Applied

### Terraform Implementation

1. **Module Structure**: Separated concerns into reusable modules
2. **Variable Validation**: Added validation rules where appropriate
3. **Documentation**: Comprehensive README and inline comments
4. **Version Control**: Proper `.gitignore` for sensitive files
5. **Tagging Strategy**: Consistent tagging with `common_tags`
6. **Security**: IMDSv2 required, egress rules defined
7. **Flexibility**: All hard-coded values converted to variables

## Cost Considerations

Both implementations create the same resources, so costs are identical:

- **g4dn.4xlarge**: ~$1.20/hour
- **200GB gp3 (3000 IOPS, 125 MB/s)**: ~$16/month
- **Data transfer**: Variable

**Total**: ~$876-900/month for 24/7 operation

## Migration Path

To migrate from the CloudFormation template to Terraform:

1. **Import existing resources** (if already deployed):
   ```bash
   terraform import module.security_group.aws_security_group.this sg-xxxxx
   terraform import module.ec2_instance.aws_instance.this i-xxxxx
   ```

2. **Verify configuration**:
   ```bash
   terraform plan
   ```

3. **Apply if needed**:
   ```bash
   terraform apply
   ```

## Conclusion

The Terraform implementation provides:
- ✅ Better modularity and reusability
- ✅ More comprehensive outputs
- ✅ Enhanced security defaults
- ✅ Better development workflow
- ✅ Multi-cloud portability

While maintaining the same infrastructure functionality as the original CloudFormation template.
