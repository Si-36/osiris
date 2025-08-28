"""
Iceberg Branching and Tagging
=============================
Implements Git-like branching for data with isolated experiments,
safe rollbacks, and promotion workflows.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PromotionStrategy(Enum):
    """Strategies for promoting branches"""
    FAST_FORWARD = "fast_forward"      # No conflicts, direct promotion
    MERGE = "merge"                     # Create merge commit
    REBASE = "rebase"                   # Rebase changes
    CHERRY_PICK = "cherry_pick"         # Select specific snapshots


@dataclass
class BranchConfig:
    """Configuration for branch operations"""
    # Branch settings
    allow_force_push: bool = False
    require_review: bool = True
    auto_cleanup_days: Optional[int] = 30
    
    # Protection rules
    protected_branches: Set[str] = field(default_factory=lambda: {"main", "production"})
    allowed_mergers: Set[str] = field(default_factory=set)
    
    # Validation
    require_tests: bool = True
    test_namespace: str = "tests"
    
    # Conflict resolution
    conflict_resolution: str = "manual"  # manual, theirs, ours
    
    # Retention
    max_branch_age_days: int = 90
    max_branches_per_user: int = 10


@dataclass
class Branch:
    """Branch metadata"""
    name: str
    ref: str  # Current commit/snapshot
    created_at: datetime
    created_by: str
    
    # Branch info
    parent_branch: str
    parent_ref: str
    
    # Stats
    commits_ahead: int = 0
    commits_behind: int = 0
    
    # Metadata
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_protected(self, config: BranchConfig) -> bool:
        """Check if branch is protected"""
        return self.name in config.protected_branches


@dataclass
class Tag:
    """Tag metadata"""
    name: str
    ref: str  # Tagged commit/snapshot
    created_at: datetime
    created_by: str
    
    # Tag info
    tag_type: str = "lightweight"  # lightweight or annotated
    message: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MergeResult:
    """Result of a merge operation"""
    success: bool
    merge_ref: Optional[str] = None
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Stats
    files_changed: int = 0
    rows_added: int = 0
    rows_deleted: int = 0
    rows_modified: int = 0
    
    # Validation results
    tests_passed: bool = True
    test_results: Dict[str, Any] = field(default_factory=dict)
    
    # Rollback info
    rollback_ref: Optional[str] = None


class BranchManager:
    """
    Manages branches for Iceberg tables.
    Provides Git-like workflow for data experimentation.
    """
    
    def __init__(self, catalog: 'IcebergCatalog', config: Optional[BranchConfig] = None):
        self.catalog = catalog
        self.config = config or BranchConfig()
        self._branches: Dict[str, Branch] = {}
        
    async def create_branch(self,
                          branch_name: str,
                          from_ref: str = "main",
                          created_by: str = "system",
                          description: Optional[str] = None) -> Branch:
        """Create a new branch"""
        # Validate branch name
        if branch_name in self._branches:
            raise ValueError(f"Branch {branch_name} already exists")
            
        if "/" in branch_name and not branch_name.startswith(("feature/", "experiment/", "bugfix/")):
            raise ValueError("Branch names with / must start with feature/, experiment/, or bugfix/")
            
        # Check user limits
        user_branches = sum(1 for b in self._branches.values() if b.created_by == created_by)
        if user_branches >= self.config.max_branches_per_user:
            raise ValueError(f"User {created_by} has reached branch limit")
            
        # Get parent branch
        parent_branch = self._branches.get(from_ref)
        if not parent_branch and from_ref != "main":
            raise ValueError(f"Parent branch {from_ref} not found")
            
        # Create branch in catalog
        await self.catalog.create_branch(branch_name, from_ref)
        
        # Create branch object
        branch = Branch(
            name=branch_name,
            ref=from_ref,  # Initially points to parent
            created_at=datetime.utcnow(),
            created_by=created_by,
            parent_branch=from_ref,
            parent_ref=from_ref,
            description=description
        )
        
        self._branches[branch_name] = branch
        logger.info(f"Created branch {branch_name} from {from_ref}")
        
        return branch
        
    async def switch_branch(self, branch_name: str) -> Branch:
        """Switch to a different branch"""
        if branch_name not in self._branches and branch_name != "main":
            raise ValueError(f"Branch {branch_name} not found")
            
        # In a real implementation, this would update the catalog's current branch
        logger.info(f"Switched to branch {branch_name}")
        
        return self._branches.get(branch_name)
        
    async def merge_branch(self,
                          source_branch: str,
                          target_branch: str = "main",
                          strategy: PromotionStrategy = PromotionStrategy.MERGE,
                          merged_by: str = "system",
                          message: Optional[str] = None) -> MergeResult:
        """Merge source branch into target branch"""
        # Validate branches
        source = self._branches.get(source_branch)
        if not source:
            raise ValueError(f"Source branch {source_branch} not found")
            
        # Check protection
        if target_branch in self.config.protected_branches:
            if self.config.require_review and merged_by not in self.config.allowed_mergers:
                raise PermissionError(f"User {merged_by} not allowed to merge to {target_branch}")
                
        # Run tests if required
        test_results = {}
        if self.config.require_tests:
            test_results = await self._run_branch_tests(source_branch)
            if not all(r.get('passed', False) for r in test_results.values()):
                return MergeResult(
                    success=False,
                    tests_passed=False,
                    test_results=test_results
                )
                
        # Detect conflicts
        conflicts = await self._detect_conflicts(source_branch, target_branch)
        if conflicts and self.config.conflict_resolution == "manual":
            return MergeResult(
                success=False,
                conflicts=conflicts,
                test_results=test_results
            )
            
        # Perform merge based on strategy
        if strategy == PromotionStrategy.FAST_FORWARD:
            merge_ref = await self._fast_forward_merge(source_branch, target_branch)
        elif strategy == PromotionStrategy.MERGE:
            merge_ref = await self._create_merge_commit(source_branch, target_branch, message)
        elif strategy == PromotionStrategy.REBASE:
            merge_ref = await self._rebase_branch(source_branch, target_branch)
        else:
            raise ValueError(f"Unsupported merge strategy: {strategy}")
            
        # Calculate stats
        stats = await self._calculate_merge_stats(source_branch, target_branch, merge_ref)
        
        # Update branch tracking
        source.commits_ahead = 0
        
        return MergeResult(
            success=True,
            merge_ref=merge_ref,
            files_changed=stats['files_changed'],
            rows_added=stats['rows_added'],
            rows_deleted=stats['rows_deleted'],
            rows_modified=stats['rows_modified'],
            tests_passed=True,
            test_results=test_results,
            rollback_ref=target_branch  # Can rollback to previous target state
        )
        
    async def delete_branch(self, branch_name: str, force: bool = False) -> None:
        """Delete a branch"""
        if branch_name in self.config.protected_branches and not force:
            raise ValueError(f"Cannot delete protected branch {branch_name}")
            
        branch = self._branches.get(branch_name)
        if not branch:
            raise ValueError(f"Branch {branch_name} not found")
            
        # Check if branch has unmerged changes
        if branch.commits_ahead > 0 and not force:
            raise ValueError(f"Branch {branch_name} has unmerged changes")
            
        # Delete from catalog
        # await self.catalog.delete_branch(branch_name)
        
        # Remove from tracking
        del self._branches[branch_name]
        logger.info(f"Deleted branch {branch_name}")
        
    async def list_branches(self, 
                          filter_by_user: Optional[str] = None,
                          include_stats: bool = True) -> List[Branch]:
        """List all branches"""
        branches = list(self._branches.values())
        
        # Add main branch if not tracked
        if "main" not in self._branches:
            branches.append(Branch(
                name="main",
                ref="HEAD",
                created_at=datetime.utcnow(),
                created_by="system",
                parent_branch="",
                parent_ref=""
            ))
            
        # Filter by user
        if filter_by_user:
            branches = [b for b in branches if b.created_by == filter_by_user]
            
        # Update stats if requested
        if include_stats:
            for branch in branches:
                await self._update_branch_stats(branch)
                
        return branches
        
    async def _detect_conflicts(self, source: str, target: str) -> List[Dict[str, Any]]:
        """Detect merge conflicts between branches"""
        # In a real implementation, this would compare schemas, partitions, and data
        # For now, return empty list (no conflicts)
        return []
        
    async def _fast_forward_merge(self, source: str, target: str) -> str:
        """Perform fast-forward merge"""
        # Update target branch pointer to source
        source_branch = self._branches[source]
        merge_ref = source_branch.ref
        
        logger.info(f"Fast-forward merge: {target} -> {merge_ref}")
        return merge_ref
        
    async def _create_merge_commit(self, source: str, target: str, message: Optional[str]) -> str:
        """Create a merge commit"""
        # In real implementation, this would create a new snapshot combining changes
        merge_ref = f"merge_{source}_to_{target}_{datetime.utcnow().timestamp()}"
        
        logger.info(f"Created merge commit: {merge_ref}")
        return merge_ref
        
    async def _rebase_branch(self, source: str, target: str) -> str:
        """Rebase source branch onto target"""
        # In real implementation, this would replay source changes on top of target
        rebase_ref = f"rebase_{source}_onto_{target}_{datetime.utcnow().timestamp()}"
        
        logger.info(f"Rebased {source} onto {target}: {rebase_ref}")
        return rebase_ref
        
    async def _run_branch_tests(self, branch_name: str) -> Dict[str, Any]:
        """Run validation tests on branch"""
        # In real implementation, would run actual data quality tests
        return {
            'schema_validation': {'passed': True, 'message': 'Schema valid'},
            'data_quality': {'passed': True, 'score': 0.95},
            'performance': {'passed': True, 'query_time_ms': 45}
        }
        
    async def _calculate_merge_stats(self, source: str, target: str, merge_ref: str) -> Dict[str, Any]:
        """Calculate statistics for merge"""
        # In real implementation, would calculate actual changes
        return {
            'files_changed': 5,
            'rows_added': 1000,
            'rows_deleted': 50,
            'rows_modified': 200
        }
        
    async def _update_branch_stats(self, branch: Branch) -> None:
        """Update branch statistics"""
        # In real implementation, would compare with parent branch
        # For now, set some sample values
        branch.commits_ahead = 3
        branch.commits_behind = 1
        
    async def cleanup_old_branches(self) -> List[str]:
        """Clean up old branches based on retention policy"""
        cleaned = []
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.max_branch_age_days)
        
        for branch_name, branch in list(self._branches.items()):
            if (branch.created_at < cutoff_date and 
                branch_name not in self.config.protected_branches and
                branch.commits_ahead == 0):
                
                await self.delete_branch(branch_name, force=True)
                cleaned.append(branch_name)
                
        logger.info(f"Cleaned up {len(cleaned)} old branches")
        return cleaned


class TagManager:
    """
    Manages tags for Iceberg table snapshots.
    Provides immutable references to specific data states.
    """
    
    def __init__(self, catalog: 'IcebergCatalog'):
        self.catalog = catalog
        self._tags: Dict[str, Tag] = {}
        
    async def create_tag(self,
                        tag_name: str,
                        ref: str = "HEAD",
                        created_by: str = "system",
                        message: Optional[str] = None) -> Tag:
        """Create a new tag"""
        # Validate tag name
        if tag_name in self._tags:
            raise ValueError(f"Tag {tag_name} already exists")
            
        # Semantic versioning validation
        if tag_name.startswith("v") and not self._is_valid_version(tag_name[1:]):
            raise ValueError(f"Tag {tag_name} does not follow semantic versioning")
            
        # Create tag in catalog
        await self.catalog.create_tag(tag_name, ref)
        
        # Create tag object
        tag = Tag(
            name=tag_name,
            ref=ref,
            created_at=datetime.utcnow(),
            created_by=created_by,
            tag_type="annotated" if message else "lightweight",
            message=message
        )
        
        self._tags[tag_name] = tag
        logger.info(f"Created tag {tag_name} at {ref}")
        
        return tag
        
    async def delete_tag(self, tag_name: str) -> None:
        """Delete a tag"""
        if tag_name not in self._tags:
            raise ValueError(f"Tag {tag_name} not found")
            
        # Delete from catalog
        # await self.catalog.delete_tag(tag_name)
        
        # Remove from tracking
        del self._tags[tag_name]
        logger.info(f"Deleted tag {tag_name}")
        
    async def list_tags(self, pattern: Optional[str] = None) -> List[Tag]:
        """List all tags"""
        tags = list(self._tags.values())
        
        # Filter by pattern
        if pattern:
            import fnmatch
            tags = [t for t in tags if fnmatch.fnmatch(t.name, pattern)]
            
        # Sort by creation date
        tags.sort(key=lambda t: t.created_at, reverse=True)
        
        return tags
        
    async def get_tag(self, tag_name: str) -> Optional[Tag]:
        """Get a specific tag"""
        return self._tags.get(tag_name)
        
    def _is_valid_version(self, version: str) -> bool:
        """Check if version string is valid semantic version"""
        import re
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
        return bool(re.match(pattern, version))