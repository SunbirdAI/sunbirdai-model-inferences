resource "aws_instance" "this" {
  ami                         = var.ami_id
  instance_type               = var.instance_type
  key_name                    = var.key_name
  user_data                   = var.user_data
  user_data_replace_on_change = var.user_data_replace_on_change

  vpc_security_group_ids      = var.security_group_ids
  associate_public_ip_address = var.associate_public_ip

  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = var.root_volume_type
    iops                  = var.root_volume_type == "gp3" || var.root_volume_type == "io1" || var.root_volume_type == "io2" ? var.root_volume_iops : null
    throughput            = var.root_volume_type == "gp3" ? var.root_volume_throughput : null
    encrypted             = var.root_volume_encrypted
    delete_on_termination = true
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = var.metadata_http_tokens
    http_put_response_hop_limit = var.metadata_hop_limit
  }

  private_dns_name_options {
    hostname_type                        = "ip-name"
    enable_resource_name_dns_a_record    = var.enable_dns_hostnames
    enable_resource_name_dns_aaaa_record = false
  }

  tags = merge(
    {
      Name = var.instance_name
    },
    var.tags
  )

  volume_tags = merge(
    {
      Name = "${var.instance_name}-root-volume"
    },
    var.tags
  )
}
