"""
Multi-Cloud IAC Security Hardening Templates

Apply to: AWS, Azure, GCP, Kubernetes
Use for: Infrastructure as Code security baselines

Features:
- Security baselines for all major cloud providers
- Compliance-ready configurations
- Best practices enforcement
- Modular and reusable
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class CloudProvider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    KUBERNETES = "kubernetes"
    MULTI = "multi-cloud"


class ComplianceStandard(Enum):
    """Compliance standards"""
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    GDPR = "gdpr"


# ============================================================================
# AWS Security Baseline
# ============================================================================

AWS_SECURITY_BASELINE = {
    'vpc': {
        'enable_flow_logs': True,
        'enable_dns_hostnames': True,
        'enable_dns_support': True,
        'default_security_group_rules': 'deny_all',
    },
    'ec2': {
        'require_imdsv2': True,  # Prevent SSRF to metadata service
        'disable_public_ip_default': True,
        'encrypted_ebs_by_default': True,
        'instance_metadata_tags': True,
        'detailed_monitoring': True,
    },
    's3': {
        'block_public_access': True,
        'enable_versioning': True,
        'encryption_at_rest': 'AES256',
        'enable_access_logging': True,
        'mfa_delete': True,
        'lifecycle_policy': 'enabled',
    },
    'rds': {
        'storage_encrypted': True,
        'backup_retention_days': 30,
        'multi_az': True,
        'deletion_protection': True,
        'enable_iam_auth': True,
        'enable_audit_logging': True,
    },
    'lambda': {
        'reserved_concurrency': True,
        'dead_letter_queue': True,
        'vpc_config': 'required',
        'environment_encryption': True,
    },
    'iam': {
        'require_mfa': True,
        'password_policy': {
            'minimum_length': 14,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_numbers': True,
            'require_symbols': True,
            'max_age_days': 90,
        },
        'access_analyzer': True,
        'credential_report_enabled': True,
    },
    'cloudtrail': {
        'enabled': True,
        'multi_region': True,
        'log_file_validation': True,
        'kms_encryption': True,
    },
    'cloudwatch': {
        'alarm_on_root_usage': True,
        'alarm_on_unauthorized_api_calls': True,
        'alarm_on_console_signin_failures': True,
        'log_retention_days': 365,
    },
    'kms': {
        'enable_key_rotation': True,
        'deletion_window_days': 30,
    },
}

# ============================================================================
# Azure Security Baseline
# ============================================================================

AZURE_SECURITY_BASELINE = {
    'network': {
        'nsg_default_rules': 'deny_all_inbound',
        'ddos_protection_standard': True,
        'service_endpoints': True,
        'private_link': True,
    },
    'storage': {
        'secure_transfer_required': True,
        'encryption': 'customer_managed_keys',
        'allow_blob_public_access': False,
        'soft_delete_enabled': True,
        'soft_delete_retention_days': 90,
        'versioning_enabled': True,
    },
    'sql': {
        'transparent_data_encryption': True,
        'auditing_enabled': True,
        'threat_detection_enabled': True,
        'azure_ad_authentication': True,
        'minimum_tls_version': '1.2',
    },
    'vm': {
        'managed_disks_only': True,
        'disk_encryption': True,
        'boot_diagnostics': True,
        'backup_enabled': True,
        'no_public_ip': True,
    },
    'key_vault': {
        'purge_protection': True,
        'soft_delete_enabled': True,
        'rbac_authorization': True,
        'network_acls': 'deny_by_default',
    },
    'monitor': {
        'activity_log_retention_days': 365,
        'diagnostic_settings': 'all_resources',
        'action_groups_configured': True,
    },
    'defender': {
        'enabled_for_servers': True,
        'enabled_for_storage': True,
        'enabled_for_sql': True,
        'enabled_for_containers': True,
    },
}

# ============================================================================
# GCP Security Baseline
# ============================================================================

GCP_SECURITY_BASELINE = {
    'compute': {
        'shielded_vm': True,
        'confidential_computing': True,
        'os_login': True,
        'block_project_ssh_keys': True,
        'enable_vtpm': True,
        'enable_integrity_monitoring': True,
    },
    'storage': {
        'uniform_bucket_level_access': True,
        'encryption': 'CMEK',  # Customer-managed encryption keys
        'versioning_enabled': True,
        'retention_policy': 90,
        'public_access_prevention': 'enforced',
    },
    'cloud_sql': {
        'require_ssl': True,
        'automated_backups': True,
        'binary_logging': True,
        'point_in_time_recovery': True,
        'deletion_protection': True,
    },
    'iam': {
        'enforce_mfa': True,
        'service_account_key_rotation': 90,
        'least_privilege': True,
        'no_primitive_roles': True,
    },
    'networking': {
        'vpc_flow_logs': True,
        'private_google_access': True,
        'cloud_armor': True,
        'ssl_policies': 'MODERN',
    },
    'logging': {
        'audit_logs_enabled': True,
        'data_access_logs': True,
        'retention_days': 365,
        'log_sinks_configured': True,
    },
    'kms': {
        'rotation_period_days': 90,
        'algorithm': 'GOOGLE_SYMMETRIC_ENCRYPTION',
    },
}

# ============================================================================
# Kubernetes Security Baseline
# ============================================================================

K8S_SECURITY_BASELINE = {
    'pod_security': {
        'run_as_non_root': True,
        'read_only_root_filesystem': True,
        'drop_all_capabilities': True,
        'no_privilege_escalation': True,
        'seccomp_profile': 'RuntimeDefault',
        'apparmor_profile': 'runtime/default',
    },
    'network_policy': {
        'default_deny_all': True,
        'egress_rules': 'explicit_allow_only',
        'ingress_rules': 'explicit_allow_only',
    },
    'rbac': {
        'enabled': True,
        'default_service_account_automount': False,
        'least_privilege': True,
        'no_cluster_admin_default': True,
    },
    'secrets': {
        'encryption_at_rest': True,
        'encryption_provider': 'kms',
        'no_secrets_in_env': True,
        'use_external_secrets': True,
    },
    'admission_control': {
        'pod_security_policy': True,
        'image_policy_webhook': True,
        'always_pull_images': True,
        'limit_ranges': True,
        'resource_quotas': True,
    },
    'audit_logging': {
        'enabled': True,
        'retention_days': 365,
        'log_level': 'Metadata',
    },
    'tls': {
        'minimum_version': '1.2',
        'certificate_rotation': True,
        'rotate_kubelet_certificates': True,
    },
}

# ============================================================================
# Security Validator
# ============================================================================

@dataclass
class SecurityViolation:
    """Represents a security configuration violation"""
    resource: str
    violation: str
    severity: str
    recommendation: str
    compliance_standards: List[ComplianceStandard]


class IACSecurityValidator:
    """
    Validates Infrastructure as Code configurations against security baselines.
    
    Apply to: Terraform, CloudFormation, ARM templates, Kubernetes manifests
    
    Features:
    - Multi-cloud support
    - Compliance checking
    - Best practices enforcement
    """
    
    def __init__(self, provider: CloudProvider):
        self.provider = provider
        self.violations: List[SecurityViolation] = []
        
        # Load appropriate baseline
        if provider == CloudProvider.AWS:
            self.baseline = AWS_SECURITY_BASELINE
        elif provider == CloudProvider.AZURE:
            self.baseline = AZURE_SECURITY_BASELINE
        elif provider == CloudProvider.GCP:
            self.baseline = GCP_SECURITY_BASELINE
        elif provider == CloudProvider.KUBERNETES:
            self.baseline = K8S_SECURITY_BASELINE
        else:
            self.baseline = {}
    
    def validate_configuration(self, config: Dict[str, Any]) -> List[SecurityViolation]:
        """
        Validate infrastructure configuration against security baseline.
        
        Args:
            config: Infrastructure configuration dictionary
            
        Returns:
            List of security violations found
        """
        self.violations = []
        
        # Validate each category
        for category, rules in self.baseline.items():
            if category in config:
                self._validate_category(category, rules, config[category])
        
        return self.violations
    
    def _validate_category(
        self, 
        category: str, 
        baseline_rules: Dict[str, Any],
        actual_config: Dict[str, Any]
    ) -> None:
        """Validate a specific category against baseline"""
        for rule, expected_value in baseline_rules.items():
            if isinstance(expected_value, dict):
                # Nested configuration
                if rule in actual_config:
                    self._validate_category(
                        f"{category}.{rule}",
                        expected_value,
                        actual_config[rule]
                    )
            else:
                # Simple value check
                actual_value = actual_config.get(rule)
                
                if actual_value != expected_value:
                    self.violations.append(SecurityViolation(
                        resource=f"{category}.{rule}",
                        violation=f"Expected {expected_value}, got {actual_value}",
                        severity="HIGH" if expected_value is True else "MEDIUM",
                        recommendation=f"Set {category}.{rule} to {expected_value}",
                        compliance_standards=[
                            ComplianceStandard.SOC2,
                            ComplianceStandard.ISO27001
                        ]
                    ))
    
    def generate_compliance_report(self) -> str:
        """Generate compliance report"""
        if not self.violations:
            return "✅ All security checks passed!"
        
        report = f"\n⚠️  Found {len(self.violations)} security violations:\n\n"
        
        for i, violation in enumerate(self.violations, 1):
            report += f"{i}. {violation.resource}\n"
            report += f"   Violation: {violation.violation}\n"
            report += f"   Severity: {violation.severity}\n"
            report += f"   Recommendation: {violation.recommendation}\n\n"
        
        return report


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=== IAC Security Hardening Demo ===\n")
    
    # Example AWS configuration
    aws_config = {
        'ec2': {
            'require_imdsv2': True,
            'disable_public_ip_default': False,  # Violation
            'encrypted_ebs_by_default': True,
        },
        's3': {
            'block_public_access': False,  # Violation
            'enable_versioning': True,
            'encryption_at_rest': 'AES256',
        },
    }
    
    # Validate
    validator = IACSecurityValidator(CloudProvider.AWS)
    violations = validator.validate_configuration(aws_config)
    
    # Generate report
    report = validator.generate_compliance_report()
    print(report)
    
    print(f"\n📊 Summary:")
    print(f"   Total Violations: {len(violations)}")
    print(f"   Critical: {sum(1 for v in violations if v.severity == 'HIGH')}")
    print(f"   Medium: {sum(1 for v in violations if v.severity == 'MEDIUM')}")
    
    print("\n=== Demo Complete ===")
