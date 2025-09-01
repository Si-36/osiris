#!/usr/bin/env python3
"""
Safe Migration from Pickle to PostgreSQL
=======================================
Preserves all existing data while upgrading to causal persistence
"""

import asyncio
import pickle
import json
import os
from pathlib import Path
from datetime import datetime
import structlog
from typing import Dict, Any, List
import asyncpg
import hashlib

from .causal_state_manager import (
    CausalPersistenceManager,
    StateType,
    CausalContext,
    get_causal_manager
)
from .state_manager import StateSnapshot as OldSnapshot

logger = structlog.get_logger()

class PickleMigrator:
    """Safely migrate from pickle files to PostgreSQL"""
    
    def __init__(self, 
                 pickle_dir: str = "/tmp/aura_state",
                 backup_dir: str = "./pickle_backup"):
        self.pickle_dir = Path(pickle_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        self.stats = {
            "files_found": 0,
            "successfully_migrated": 0,
            "failed": 0,
            "already_migrated": 0
        }
    
    async def migrate_all(self, dry_run: bool = False):
        """Migrate all pickle files to new persistence"""
        logger.info(f"Starting pickle migration from {self.pickle_dir}")
        logger.info(f"Dry run: {dry_run}")
        
        # Get causal manager
        manager = await get_causal_manager()
        
        # Find all pickle files
        pickle_files = list(self.pickle_dir.glob("*.pkl"))
        self.stats["files_found"] = len(pickle_files)
        
        logger.info(f"Found {len(pickle_files)} pickle files to migrate")
        
        # Check if we can connect to PostgreSQL
        if not manager.legacy_mode:
            logger.info("PostgreSQL connection established")
        else:
            logger.warning("Running in legacy mode - PostgreSQL not available")
            if not dry_run:
                logger.error("Cannot migrate without PostgreSQL connection")
                return self.stats
        
        # Migrate each file
        for pkl_file in pickle_files:
            try:
                await self._migrate_file(pkl_file, manager, dry_run)
            except Exception as e:
                logger.error(f"Failed to migrate {pkl_file}: {e}")
                self.stats["failed"] += 1
        
        # Summary
        logger.info("Migration complete!")
        logger.info(f"Files found: {self.stats['files_found']}")
        logger.info(f"Successfully migrated: {self.stats['successfully_migrated']}")
        logger.info(f"Already migrated: {self.stats['already_migrated']}")
        logger.info(f"Failed: {self.stats['failed']}")
        
        return self.stats
    
    async def _migrate_file(self, pkl_file: Path, manager: CausalPersistenceManager, dry_run: bool):
        """Migrate a single pickle file"""
        logger.debug(f"Processing {pkl_file.name}")
        
        # Load pickle file
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logger.error(f"Cannot load pickle file {pkl_file}: {e}")
            self.stats["failed"] += 1
            return
        
        # Detect format and extract info
        if isinstance(data, dict) and 'state_id' in data:
            # New format snapshot
            await self._migrate_snapshot(data, manager, dry_run, pkl_file)
        elif hasattr(data, 'state_id'):
            # Old format StateSnapshot object
            await self._migrate_old_snapshot(data, manager, dry_run, pkl_file)
        else:
            # Unknown format - try to extract what we can
            await self._migrate_unknown(data, manager, dry_run, pkl_file)
    
    async def _migrate_snapshot(self, data: Dict, manager: CausalPersistenceManager, dry_run: bool, pkl_file: Path):
        """Migrate a snapshot in dict format"""
        try:
            # Extract key information
            state_type_str = data.get('state_type', 'component_state')
            if hasattr(state_type_str, 'value'):
                state_type_str = state_type_str.value
            
            # Map to new enum
            state_type = self._map_state_type(state_type_str)
            component_id = data.get('component_id', pkl_file.stem)
            
            # Check if already migrated
            existing = await manager.load_state(state_type, component_id)
            if existing:
                logger.debug(f"Already migrated: {component_id}")
                self.stats["already_migrated"] += 1
                return
            
            if not dry_run:
                # Create causal context for migration
                context = CausalContext(
                    causes=["migrated_from_pickle"],
                    effects=[],
                    counterfactuals={
                        "original_format": "pickle",
                        "original_file": str(pkl_file.name),
                        "migration_time": datetime.now().isoformat()
                    },
                    confidence=1.0
                )
                
                # Save to new persistence
                state_data = data.get('data', data)
                await manager.save_state(
                    state_type=state_type,
                    component_id=component_id,
                    state_data=state_data,
                    causal_context=context
                )
                
                # Backup original
                backup_path = self.backup_dir / pkl_file.name
                pkl_file.rename(backup_path)
                
                logger.info(f"Migrated {component_id} ({state_type.value})")
            else:
                logger.info(f"Would migrate {component_id} ({state_type.value})")
            
            self.stats["successfully_migrated"] += 1
            
        except Exception as e:
            logger.error(f"Failed to migrate snapshot: {e}")
            self.stats["failed"] += 1
    
    async def _migrate_old_snapshot(self, snapshot: OldSnapshot, manager: CausalPersistenceManager, dry_run: bool, pkl_file: Path):
        """Migrate an old StateSnapshot object"""
        try:
            # Map to new format
            state_type = self._map_state_type(snapshot.state_type.value)
            
            # Check if already migrated
            existing = await manager.load_state(state_type, snapshot.component_id)
            if existing:
                logger.debug(f"Already migrated: {snapshot.component_id}")
                self.stats["already_migrated"] += 1
                return
            
            if not dry_run:
                # Create migration context
                context = CausalContext(
                    causes=["migrated_from_old_snapshot"],
                    effects=[],
                    counterfactuals={
                        "original_state_id": snapshot.state_id,
                        "original_timestamp": snapshot.timestamp,
                        "original_checksum": snapshot.checksum
                    },
                    confidence=1.0
                )
                
                # Save to new persistence
                await manager.save_state(
                    state_type=state_type,
                    component_id=snapshot.component_id,
                    state_data=snapshot.data,
                    causal_context=context
                )
                
                # Backup
                backup_path = self.backup_dir / pkl_file.name
                pkl_file.rename(backup_path)
                
                logger.info(f"Migrated old snapshot {snapshot.component_id}")
            else:
                logger.info(f"Would migrate old snapshot {snapshot.component_id}")
            
            self.stats["successfully_migrated"] += 1
            
        except Exception as e:
            logger.error(f"Failed to migrate old snapshot: {e}")
            self.stats["failed"] += 1
    
    async def _migrate_unknown(self, data: Any, manager: CausalPersistenceManager, dry_run: bool, pkl_file: Path):
        """Try to migrate unknown format"""
        try:
            # Generate component ID from filename
            component_id = pkl_file.stem
            
            # Guess state type from filename or content
            state_type = StateType.COMPONENT_STATE
            if 'liquid' in pkl_file.name.lower():
                state_type = StateType.LIQUID_NETWORK
            elif 'memory' in pkl_file.name.lower():
                state_type = StateType.MEMORY_TIER
            elif 'agent' in pkl_file.name.lower():
                state_type = StateType.AGENT_MEMORY
            
            if not dry_run:
                # Wrap unknown data
                wrapped_data = {
                    "original_type": type(data).__name__,
                    "data": data if isinstance(data, dict) else {"value": str(data)},
                    "migrated_from": str(pkl_file.name)
                }
                
                context = CausalContext(
                    causes=["migrated_unknown_format"],
                    counterfactuals={
                        "original_file": str(pkl_file.name),
                        "guessed_type": state_type.value
                    }
                )
                
                await manager.save_state(
                    state_type=state_type,
                    component_id=component_id,
                    state_data=wrapped_data,
                    causal_context=context
                )
                
                # Backup
                backup_path = self.backup_dir / f"unknown_{pkl_file.name}"
                pkl_file.rename(backup_path)
                
                logger.info(f"Migrated unknown format as {component_id}")
            else:
                logger.info(f"Would migrate unknown format as {component_id}")
            
            self.stats["successfully_migrated"] += 1
            
        except Exception as e:
            logger.error(f"Failed to migrate unknown format: {e}")
            self.stats["failed"] += 1
    
    def _map_state_type(self, old_type: str) -> StateType:
        """Map old state type strings to new enum"""
        mapping = {
            "component_state": StateType.COMPONENT_STATE,
            "liquid_network": StateType.LIQUID_NETWORK,
            "memory_tier": StateType.MEMORY_TIER,
            "metabolic_budget": StateType.METABOLIC_BUDGET,
            "system_config": StateType.SYSTEM_CONFIG,
            # New types
            "agent_memory": StateType.AGENT_MEMORY,
            "neural_checkpoint": StateType.NEURAL_CHECKPOINT,
            "tda_cache": StateType.TDA_CACHE,
            "swarm_state": StateType.SWARM_STATE
        }
        
        return mapping.get(old_type, StateType.COMPONENT_STATE)

async def verify_migration():
    """Verify that migration was successful"""
    logger.info("Verifying migration...")
    
    manager = await get_causal_manager()
    
    if manager.legacy_mode:
        logger.error("Still in legacy mode - PostgreSQL not available")
        return False
    
    # Check some key components
    test_components = [
        ("liquid_network", StateType.LIQUID_NETWORK),
        ("memory_tier", StateType.MEMORY_TIER),
        ("agent_memory", StateType.AGENT_MEMORY)
    ]
    
    success = True
    for component_id, state_type in test_components:
        state = await manager.load_state(state_type, component_id)
        if state:
            logger.info(f"✓ Found {component_id} in new persistence")
            
            # Verify causal context
            causal_chain = await manager.get_causal_chain(
                state.get('state_id', '')
            )
            if causal_chain:
                logger.info(f"  - Has causal chain with {len(causal_chain)} entries")
        else:
            logger.warning(f"✗ Missing {component_id}")
            success = False
    
    return success

async def main():
    """Run the migration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate pickle files to PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually migrate, just show what would happen")
    parser.add_argument("--pickle-dir", default="/tmp/aura_state", help="Directory containing pickle files")
    parser.add_argument("--backup-dir", default="./pickle_backup", help="Directory to backup pickle files")
    parser.add_argument("--verify", action="store_true", help="Verify migration after completion")
    
    args = parser.parse_args()
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Run migration
    migrator = PickleMigrator(args.pickle_dir, args.backup_dir)
    stats = await migrator.migrate_all(dry_run=args.dry_run)
    
    # Verify if requested
    if args.verify and not args.dry_run:
        success = await verify_migration()
        if success:
            logger.info("✓ Migration verified successfully!")
        else:
            logger.error("✗ Migration verification failed")
            return 1
    
    return 0 if stats["failed"] == 0 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)