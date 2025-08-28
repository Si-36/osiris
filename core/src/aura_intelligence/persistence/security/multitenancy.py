"""
Multi-tenancy and Data Isolation
================================
Provides tenant isolation, data residency, and row-level security.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TenantConfig:
    """Configuration for multi-tenancy"""
    tenant_id: str
    tenant_name: str
    
    # Data residency
    allowed_regions: List[str] = field(default_factory=list)
    primary_region: Optional[str] = None
    
    # Security
    encryption_key_id: Optional[str] = None
    isolation_level: str = "strict"  # strict, moderate, shared
    
    # Quotas
    max_storage_gb: int = 1000
    max_requests_per_minute: int = 1000
    
    # Features
    enabled_features: Set[str] = field(default_factory=set)
    

class TenantIsolation:
    """
    Ensures data isolation between tenants.
    Implements logical isolation with tenant ID filtering.
    """
    
    def __init__(self):
        self._tenants: Dict[str, TenantConfig] = {}
        self._current_tenant: Optional[str] = None
        
    def register_tenant(self, config: TenantConfig) -> None:
        """Register a new tenant"""
        self._tenants[config.tenant_id] = config
        logger.info(f"Registered tenant: {config.tenant_id}")
        
    def set_current_tenant(self, tenant_id: str) -> None:
        """Set current tenant context"""
        if tenant_id not in self._tenants:
            raise ValueError(f"Unknown tenant: {tenant_id}")
            
        self._current_tenant = tenant_id
        
    def get_current_tenant(self) -> Optional[str]:
        """Get current tenant context"""
        return self._current_tenant
        
    def apply_tenant_filter(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Apply tenant isolation to query"""
        if not self._current_tenant:
            raise RuntimeError("No tenant context set")
            
        # Add tenant filter
        filtered_query = query.copy()
        filtered_query['tenant_id'] = self._current_tenant
        
        return filtered_query
        
    def validate_tenant_access(self, tenant_id: str, resource: str) -> bool:
        """Validate tenant has access to resource"""
        if tenant_id not in self._tenants:
            return False
            
        config = self._tenants[tenant_id]
        
        # Check isolation level
        if config.isolation_level == "strict":
            # In strict mode, tenant can only access their own data
            return resource.startswith(f"tenant:{tenant_id}:")
            
        return True
        

class DataResidencyManager:
    """
    Manages data residency requirements for compliance.
    Ensures data stays within approved geographical regions.
    """
    
    def __init__(self):
        self._region_mappings: Dict[str, List[str]] = {
            'eu': ['eu-west-1', 'eu-central-1'],
            'us': ['us-east-1', 'us-west-2'],
            'apac': ['ap-southeast-1', 'ap-northeast-1']
        }
        
    def validate_region(self, tenant_id: str, region: str, tenant_config: TenantConfig) -> bool:
        """Validate if data can be stored in region"""
        if not tenant_config.allowed_regions:
            return True  # No restrictions
            
        return region in tenant_config.allowed_regions
        
    def get_allowed_regions(self, tenant_config: TenantConfig) -> List[str]:
        """Get list of allowed regions for tenant"""
        if not tenant_config.allowed_regions:
            # Return all regions if no restrictions
            all_regions = []
            for regions in self._region_mappings.values():
                all_regions.extend(regions)
            return all_regions
            
        return tenant_config.allowed_regions
        
    def select_region(self, tenant_config: TenantConfig, preferred_region: Optional[str] = None) -> str:
        """Select appropriate region for data storage"""
        allowed = self.get_allowed_regions(tenant_config)
        
        if preferred_region and preferred_region in allowed:
            return preferred_region
            
        if tenant_config.primary_region and tenant_config.primary_region in allowed:
            return tenant_config.primary_region
            
        if allowed:
            return allowed[0]
            
        raise ValueError(f"No allowed regions for tenant {tenant_config.tenant_id}")
        

class RowLevelSecurity:
    """
    Implements row-level security policies.
    Filters data based on user permissions at the row level.
    """
    
    def __init__(self):
        self._policies: Dict[str, Dict[str, Any]] = {}
        
    def create_policy(self, 
                     policy_name: str,
                     table: str,
                     predicate: str,
                     roles: List[str]) -> None:
        """Create row-level security policy"""
        self._policies[policy_name] = {
            'table': table,
            'predicate': predicate,  # SQL-like predicate
            'roles': roles,
            'created_at': datetime.utcnow()
        }
        
        logger.info(f"Created RLS policy: {policy_name} for table {table}")
        
    def apply_policies(self, 
                      query: Dict[str, Any],
                      user_role: str,
                      table: str) -> Dict[str, Any]:
        """Apply RLS policies to query"""
        filtered_query = query.copy()
        
        # Find applicable policies
        for policy_name, policy in self._policies.items():
            if policy['table'] == table and user_role in policy['roles']:
                # Add policy predicate to query
                if 'filters' not in filtered_query:
                    filtered_query['filters'] = []
                    
                filtered_query['filters'].append({
                    'type': 'rls',
                    'predicate': policy['predicate']
                })
                
        return filtered_query
        
    def check_row_access(self,
                        row_data: Dict[str, Any],
                        user_role: str,
                        table: str) -> bool:
        """Check if user has access to specific row"""
        # Find applicable policies
        for policy_name, policy in self._policies.items():
            if policy['table'] == table and user_role in policy['roles']:
                # Evaluate predicate (simplified)
                # In production, would use proper expression evaluation
                if 'tenant_id' in policy['predicate']:
                    # Extract tenant from predicate
                    # This is simplified - real implementation would parse SQL
                    return True
                    
        return False


__all__ = [
    'TenantConfig',
    'TenantIsolation',
    'DataResidencyManager', 
    'RowLevelSecurity'
]