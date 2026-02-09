# ============================================================================
# Universal Terraform Module Template
# ============================================================================
# This template provides a foundation for creating reusable infrastructure
# modules across different cloud providers (AWS, Azure, GCP).
# Customize the provider and resources based on your specific requirements.
# ============================================================================

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    # Customize provider based on your cloud platform
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    # Uncomment for Azure:
    # azurerm = {
    #   source  = "hashicorp/azurerm"
    #   version = "~> 3.0"
    # }
    # Uncomment for GCP:
    # google = {
    #   source  = "hashicorp/google"
    #   version = "~> 5.0"
    # }
  }
}

# ============================================================================
# VARIABLES - Input parameters with validation
# ============================================================================

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "Project name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "region" {
  description = "Cloud region for resource deployment"
  type        = string
  default     = "us-east-1"  # AWS example, customize for your provider
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "enable_monitoring" {
  description = "Enable monitoring and alerting"
  type        = bool
  default     = true
}

variable "retention_days" {
  description = "Log retention period in days"
  type        = number
  default     = 30
  validation {
    condition     = var.retention_days >= 1 && var.retention_days <= 365
    error_message = "Retention days must be between 1 and 365."
  }
}

# ============================================================================
# LOCALS - Computed values and common tags
# ============================================================================

locals {
  # Common naming convention
  resource_prefix = "${var.project_name}-${var.environment}"
  
  # Merge common tags with custom tags
  common_tags = merge(
    {
      Environment = var.environment
      Project     = var.project_name
      ManagedBy   = "Terraform"
      CreatedDate = timestamp()
    },
    var.tags
  )
}

# ============================================================================
# PROVIDER CONFIGURATION
# ============================================================================

# AWS Provider Configuration
provider "aws" {
  region = var.region
  
  default_tags {
    tags = local.common_tags
  }
}

# Azure Provider Configuration (Uncomment if using Azure)
# provider "azurerm" {
#   features {}
# }

# GCP Provider Configuration (Uncomment if using GCP)
# provider "google" {
#   project = var.project_name
#   region  = var.region
# }

# ============================================================================
# RESOURCE GROUP / PROJECT SETUP
# ============================================================================

# AWS: Use tags for logical grouping
# Azure: Resource Group
# resource "azurerm_resource_group" "main" {
#   name     = "${local.resource_prefix}-rg"
#   location = var.region
#   tags     = local.common_tags
# }

# GCP: Project
# resource "google_project" "main" {
#   name       = local.resource_prefix
#   project_id = local.resource_prefix
#   labels     = local.common_tags
# }

# ============================================================================
# EXAMPLE RESOURCES (Customize based on your needs)
# ============================================================================

# AWS S3 Bucket Example
resource "aws_s3_bucket" "main" {
  bucket = "${local.resource_prefix}-data"
  
  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-data"
    }
  )
}

resource "aws_s3_bucket_versioning" "main" {
  bucket = aws_s3_bucket.main.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "main" {
  bucket = aws_s3_bucket.main.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# AWS CloudWatch Log Group Example
resource "aws_cloudwatch_log_group" "main" {
  count             = var.enable_monitoring ? 1 : 0
  name              = "/aws/${local.resource_prefix}"
  retention_in_days = var.retention_days
  
  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-logs"
    }
  )
}

# ============================================================================
# OUTPUTS - Export important values
# ============================================================================

output "resource_prefix" {
  description = "Common resource prefix used for naming"
  value       = local.resource_prefix
}

output "bucket_name" {
  description = "Name of the S3 bucket"
  value       = aws_s3_bucket.main.id
}

output "bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.main.arn
}

output "log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = var.enable_monitoring ? aws_cloudwatch_log_group.main[0].name : null
}

output "region" {
  description = "Deployment region"
  value       = var.region
}

output "tags" {
  description = "Common tags applied to resources"
  value       = local.common_tags
}

# ============================================================================
# NOTES FOR CUSTOMIZATION:
# ============================================================================
# 1. Replace AWS resources with Azure/GCP equivalents as needed
# 2. Add your specific infrastructure resources (VPC, compute, databases, etc.)
# 3. Implement remote state backend (S3, Azure Storage, GCS)
# 4. Add data sources for existing resources
# 5. Create separate modules for complex components
# 6. Implement proper secret management (AWS Secrets Manager, Azure Key Vault, etc.)
# 7. Add lifecycle rules and backup configurations
# 8. Implement proper IAM roles and policies
# 9. Add monitoring and alerting resources
# 10. Configure networking and security groups as needed
# ============================================================================
