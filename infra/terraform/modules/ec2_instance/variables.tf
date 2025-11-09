variable "instance_name" {
  description = "Name tag for the EC2 instance"
  type        = string
}

variable "ami_id" {
  description = "AMI ID for the EC2 instance"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
}

variable "key_name" {
  description = "Name of the EC2 key pair"
  type        = string
}

variable "user_data" {
  description = "User data script to run on instance launch"
  type        = string
  default     = null
}

variable "user_data_replace_on_change" {
  description = "Whether to replace the instance when user_data changes"
  type        = bool
  default     = false
}

variable "security_group_ids" {
  description = "List of security group IDs"
  type        = list(string)
}

variable "associate_public_ip" {
  description = "Whether to associate a public IP address"
  type        = bool
  default     = true
}

variable "root_volume_size" {
  description = "Size of the root volume in GB"
  type        = number
  default     = 200
}

variable "root_volume_type" {
  description = "Type of the root volume (gp2, gp3, io1, io2)"
  type        = string
  default     = "gp3"
}

variable "root_volume_iops" {
  description = "IOPS for the root volume (only for gp3, io1, io2)"
  type        = number
  default     = 3000
}

variable "root_volume_throughput" {
  description = "Throughput for the root volume in MB/s (only for gp3)"
  type        = number
  default     = 125
}

variable "root_volume_encrypted" {
  description = "Whether to encrypt the root volume"
  type        = bool
  default     = false
}

variable "metadata_http_tokens" {
  description = "Whether to require IMDSv2 (required or optional)"
  type        = string
  default     = "required"
  validation {
    condition     = contains(["required", "optional"], var.metadata_http_tokens)
    error_message = "metadata_http_tokens must be either 'required' or 'optional'"
  }
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

variable "tags" {
  description = "Tags to apply to the instance and volumes"
  type        = map(string)
  default     = {}
}
