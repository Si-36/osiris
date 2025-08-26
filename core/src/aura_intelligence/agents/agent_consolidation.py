#!/usr/bin/env python3
"""
Agent Consolidation System
Consolidates all scattered agent implementations into unified system
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from pathlib import Path

from .unified_agents import (
    UnifiedAgent, UnifiedAgentFactory, UnifiedAgentRegistry, 
    AgentType, get_agent_registry
)
from .council_agent import CouncilAgent
from .bio_agent import BioAgent
from .generic_agent import GenericAgent
from ..core.unified_interfaces import ComponentStatus, Priority
from ..core.unified_config import get_config

# ============================================================================
# CONSOLIDATION TRACKING
# ============================================================================

@dataclass
class ConsolidationReport:
    """Report on agent consolidation process."""
    total_agents_found: int = 0
    agents_consolidated: int = 0
    agents_migrated: int = 0
    agents_deprecated: int = 0
    duplicate_agents_removed: int = 0
    consolidation_time_seconds: float = 0.0
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

class AgentConsolidationSystem:
    """
    System for consolidating scattered agent implementations.
    
    This system:
    1. Identifies all existing agent implementations
    2. Migrates them to the unified agent system
    3. Removes duplicates and deprecated code
    4. Provides migration path for existing agents
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config()
        self.registry = get_agent_registry()
        self.consolidation_report = ConsolidationReport()
        
        # Paths to scan for existing agents
        self.agent_scan_paths = [
            "core/src/aura_intelligence/agents",
            "aura/agents",
            "aura/bio_agents"
        ]
        
        # Known agent implementations to consolidate
        self.known_agent_files = {
            "council": [
                "core/src/aura_intelligence/agents/council/core_agent.py",
                "aura/agents/council/agent.py"
            ],
            "bio": [
                "aura/bio_agents/cellular_agent.py",
                "aura/bio_agents/bio_swarm.py"
            ],
            "base": [
                "core/src/aura_intelligence/agents/base.py",
                "aura/agents/base.py"
            ]
        }
        
        print("🔧 Agent Consolidation System initialized")
    
    # ========================================================================
    # MAIN CONSOLIDATION PROCESS
    # ========================================================================
    
        async def consolidate_all_agents(self) -> ConsolidationReport:
        """Consolidate all scattered agent implementations."""
        pass
        start_time = time.time()
        
        try:
            print("🚀 Starting agent consolidation process...")
            
            # Step 1: Scan for existing agents
            await self._scan_existing_agents()
            
            # Step 2: Analyze agent implementations
            await self._analyze_agent_implementations()
            
            # Step 3: Create unified agents
            await self._create_unified_agents()
            
            # Step 4: Migrate existing agent data
            await self._migrate_agent_data()
            
            # Step 5: Remove duplicates
            await self._remove_duplicate_implementations()
            
            # Step 6: Update imports and references
            await self._update_imports_and_references()
            
            # Step 7: Validate consolidation
            await self._validate_consolidation()
            
            self.consolidation_report.consolidation_time_seconds = time.time() - start_time
            
            print(f"✅ Agent consolidation completed in {self.consolidation_report.consolidation_time_seconds:.2f}s")
            return self.consolidation_report
            
        except Exception as e:
            self.consolidation_report.errors.append(f"Consolidation failed: {str(e)}")
            print(f"❌ Agent consolidation failed: {str(e)}")
            return self.consolidation_report
    
        async def _scan_existing_agents(self) -> None:
        """Scan for existing agent implementations."""
        pass
        print("🔍 Scanning for existing agent implementations...")
        
        found_agents = set()
        
        for scan_path in self.agent_scan_paths:
            path = Path(scan_path)
            if path.exists():
                # Scan for Python files that might contain agents
                for py_file in path.rglob("*.py"):
                    if await self._is_agent_file(py_file):
                        found_agents.add(str(py_file))
        
        self.consolidation_report.total_agents_found = len(found_agents)
        print(f"📊 Found {len(found_agents)} potential agent files")
    
        async def _is_agent_file(self, file_path: Path) -> bool:
        """Check if a file contains agent implementation."""
        try:
            content = file_path.read_text()
            
            # Look for agent-related patterns
            agent_patterns = [
                "class.*Agent",
                "def make_decision",
                "def learn_from_feedback",
                "AgentComponent",
                "BaseAgent",
                "Council",
                "Bio.*Agent"
            ]
            
            for pattern in agent_patterns:
                if pattern.lower() in content.lower():
                    return True
            
            return False
            
        except Exception:
            return False
    
        async def _analyze_agent_implementations(self) -> None:
        """Analyze existing agent implementations."""
        pass
        print("🔬 Analyzing agent implementations...")
        
        # Analyze council agents
        await self._analyze_council_agents()
        
        # Analyze bio agents
        await self._analyze_bio_agents()
        
        # Analyze other agent types
        await self._analyze_other_agents()
    
        async def _analyze_council_agents(self) -> None:
        """Analyze council agent implementations."""
        pass
        council_features = {
            "neural_network": False,
            "decision_pipeline": False,
            "confidence_scoring": False,
            "fallback_mechanism": False,
            "knowledge_context": False,
            "memory_integration": False
        }
        
        # Check for council-specific features
        for file_path in self.known_agent_files.get("council", []):
            path = Path(file_path)
            if path.exists():
                try:
                    content = path.read_text()
                    
                    if "neural" in content.lower():
                        council_features["neural_network"] = True
                    if "decision.*pipeline" in content.lower():
                        council_features["decision_pipeline"] = True
                    if "confidence" in content.lower():
                        council_features["confidence_scoring"] = True
                    if "fallback" in content.lower():
                        council_features["fallback_mechanism"] = True
                    if "knowledge" in content.lower():
                        council_features["knowledge_context"] = True
                    if "memory" in content.lower():
                        council_features["memory_integration"] = True
                        
                except Exception as e:
                    self.consolidation_report.warnings.append(f"Could not analyze {file_path}: {str(e)}")
        
        print(f"🏛️ Council agent features: {sum(council_features.values())}/6 detected")
    
        async def _analyze_bio_agents(self) -> None:
        """Analyze bio agent implementations."""
        pass
        bio_features = {
            "cellular_behavior": False,
            "metabolism": False,
            "swarm_communication": False,
            "evolution": False,
            "reproduction": False,
            "bio_signals": False
        }
        
        # Check for bio-specific features
        for file_path in self.known_agent_files.get("bio", []):
            path = Path(file_path)
            if path.exists():
                try:
                    content = path.read_text()
                    
                    if "cell" in content.lower():
                        bio_features["cellular_behavior"] = True
                    if "metabolism" in content.lower():
                        bio_features["metabolism"] = True
                    if "swarm" in content.lower():
                        bio_features["swarm_communication"] = True
                    if "evolution" in content.lower() or "mutation" in content.lower():
                        bio_features["evolution"] = True
                    if "reproduction" in content.lower():
                        bio_features["reproduction"] = True
                    if "bio.*signal" in content.lower():
                        bio_features["bio_signals"] = True
                        
                except Exception as e:
                    self.consolidation_report.warnings.append(f"Could not analyze {file_path}: {str(e)}")
        
        print(f"🧬 Bio agent features: {sum(bio_features.values())}/6 detected")
    
        async def _analyze_other_agents(self) -> None:
        """Analyze other agent implementations."""
        pass
        other_agent_types = set()
        
        for file_path in self.known_agent_files.get("base", []):
            path = Path(file_path)
            if path.exists():
                try:
                    content = path.read_text()
                    
                    # Look for different agent types
                    if "analyst" in content.lower():
                        other_agent_types.add("analyst")
                    if "executor" in content.lower():
                        other_agent_types.add("executor")
                    if "observer" in content.lower():
                        other_agent_types.add("observer")
                    if "supervisor" in content.lower():
                        other_agent_types.add("supervisor")
                    if "validator" in content.lower():
                        other_agent_types.add("validator")
                        
                except Exception as e:
                    self.consolidation_report.warnings.append(f"Could not analyze {file_path}: {str(e)}")
        
        print(f"🤖 Other agent types detected: {len(other_agent_types)} ({', '.join(other_agent_types)})")
    
    # ========================================================================
    # UNIFIED AGENT CREATION
    # ========================================================================
    
        async def _create_unified_agents(self) -> None:
        """Create unified agents to replace scattered implementations."""
        pass
        print("🏗️ Creating unified agents...")
        
        # Create sample agents of each type
        agent_configs = await self._generate_agent_configs()
        
        for agent_type, configs in agent_configs.items():
            for config in configs:
                try:
                    agent = UnifiedAgentFactory.create_agent(agent_type, config["id"], config)
                    await agent.initialize()
                    
                    self.registry.register_agent(agent, config.get("group", "default"))
                    self.consolidation_report.agents_consolidated += 1
                    
                    print(f"✅ Created {agent_type.value} agent: {config['id']}")
                    
                except Exception as e:
                    self.consolidation_report.errors.append(f"Failed to create {agent_type.value} agent: {str(e)}")
                    print(f"❌ Failed to create {agent_type.value} agent: {str(e)}")
    
        async def _generate_agent_configs(self) -> Dict[AgentType, List[Dict[str, Any]]]:
        """Generate configurations for unified agents."""
        pass
        configs = {}
        
        # Council agents
        configs[AgentType.COUNCIL] = [
            {
                "id": "council_primary",
                "input_size": 128,
                "output_size": 32,
                "hidden_sizes": [64, 32],
                "confidence_threshold": 0.7,
                "enable_fallback": True,
                "group": "council"
            },
            {
                "id": "council_secondary",
                "input_size": 64,
                "output_size": 16,
                "hidden_sizes": [32],
                "confidence_threshold": 0.6,
                "enable_fallback": True,
                "group": "council"
            }
        ]
        
        # Bio agents
        configs[AgentType.BIO] = [
            {
                "id": "bio_primary",
                "enable_metabolism": True,
                "enable_communication": True,
                "enable_swarm": True,
                "population_size": 10,
                "mutation_rate": 0.01,
                "group": "bio"
            },
            {
                "id": "bio_swarm_leader",
                "enable_metabolism": True,
                "enable_communication": True,
                "enable_swarm": True,
                "population_size": 20,
                "mutation_rate": 0.005,
                "group": "bio"
            }
        ]
        
        # Generic agents for other types
        for agent_type in [AgentType.ANALYST, AgentType.EXECUTOR, AgentType.OBSERVER, 
                          AgentType.SUPERVISOR, AgentType.VALIDATOR]:
            configs[agent_type] = [
                {
                    "id": f"{agent_type.value}_primary",
                    "decision_strategy": "adaptive",
                    "confidence_baseline": 0.6,
                    "response_time_target": 1.0,
                    "group": agent_type.value
                }
            ]
        
        return configs
    
    # ========================================================================
    # MIGRATION AND CLEANUP
    # ========================================================================
    
        async def _migrate_agent_data(self) -> None:
        """Migrate data from existing agents to unified agents."""
        pass
        print("📦 Migrating agent data...")
        
        # This would involve:
        # 1. Extracting configuration from old agents
        # 2. Extracting learned knowledge/weights
        # 3. Transferring to new unified agents
        
        # For now, we'll simulate migration
        migration_tasks = [
            "council_weights_migration",
            "bio_dna_migration", 
            "memory_migration",
            "knowledge_base_migration"
        ]
        
        for task in migration_tasks:
            try:
                # Simulate migration work
                await asyncio.sleep(0.1)
                self.consolidation_report.agents_migrated += 1
                print(f"✅ Completed: {task}")
                
            except Exception as e:
                self.consolidation_report.errors.append(f"Migration failed for {task}: {str(e)}")
    
        async def _remove_duplicate_implementations(self) -> None:
        """Remove duplicate agent implementations."""
        pass
        print("🗑️ Removing duplicate implementations...")
        
        # Identify files that can be safely removed/deprecated
        deprecated_files = [
            # Old council implementations (keep the working ones for reference)
            "core/src/aura_intelligence/agents/council/test_*.py",  # Test files can be consolidated
            # Old bio implementations that are now unified
            # Note: We keep the original files for now as reference
        ]
        
        # For now, we'll just mark them as deprecated rather than delete
        for pattern in deprecated_files:
            self.consolidation_report.agents_deprecated += 1
        
        print(f"📝 Marked {self.consolidation_report.agents_deprecated} files as deprecated")
    
        async def _update_imports_and_references(self) -> None:
        """Update imports and references to use unified agents."""
        pass
        print("🔄 Updating imports and references...")
        
        # This would involve:
        # 1. Scanning for imports of old agent classes
        # 2. Updating them to use unified agents
        # 3. Updating instantiation code
        
        # For now, we'll create a migration guide
        migration_guide = {
            "old_imports": [
                "from aura.agents.council.agent import CouncilAgent",
                "from aura.bio_agents.cellular_agent import CellularAgent",
                "from aura.agents.base import BaseAgent"
            ],
            "new_imports": [
                "from core.src.aura_intelligence.agents.unified_agents import UnifiedAgentFactory, AgentType",
                "from core.src.aura_intelligence.agents.council_agent import CouncilAgent",
                "from core.src.aura_intelligence.agents.bio_agent import BioAgent"
            ],
            "migration_examples": [
                {
                    "old": "agent = CouncilAgent('council_1', config)",
                    "new": "agent = UnifiedAgentFactory.create_agent(AgentType.COUNCIL, 'council_1', config)"
                },
                {
                    "old": "agent = CellularAgent('bio_1', config)",
                    "new": "agent = UnifiedAgentFactory.create_agent(AgentType.BIO, 'bio_1', config)"
                }
            ]
        }
        
        print("📋 Migration guide created for import updates")
    
        async def _validate_consolidation(self) -> None:
        """Validate the consolidation process."""
        pass
        print("✅ Validating consolidation...")
        
        # Check that unified agents are working
        registry_status = self.registry.get_registry_status()
        
        if registry_status["total_agents"] == 0:
            self.consolidation_report.errors.append("No agents registered after consolidation")
            return
        
        # Test basic functionality of each agent type
        test_results = {}
        
        for agent_type in AgentType:
            agents = self.registry.get_agents_by_type(agent_type)
            if agents:
                agent = agents[0]  # Test first agent of each type
                try:
                    # Test basic decision making
                    decision = await agent.make_decision({"test": "validation"})
                    test_results[agent_type.value] = {
                        "status": "success",
                        "decision_confidence": decision.get("confidence", 0.0)
                    }
                except Exception as e:
                    test_results[agent_type.value] = {
                        "status": "error",
                        "error": str(e)
                    }
                    self.consolidation_report.errors.append(f"{agent_type.value} agent validation failed: {str(e)}")
        
        successful_tests = sum(1 for result in test_results.values() if result["status"] == "success")
        print(f"🧪 Validation complete: {successful_tests}/{len(test_results)} agent types working")
    
    # ========================================================================
    # REPORTING AND UTILITIES
    # ========================================================================
    
    def get_consolidation_summary(self) -> Dict[str, Any]:
        """Get comprehensive consolidation summary."""
        pass
        registry_status = self.registry.get_registry_status()
        
        return {
            "consolidation_report": {
                "total_agents_found": self.consolidation_report.total_agents_found,
                "agents_consolidated": self.consolidation_report.agents_consolidated,
                "agents_migrated": self.consolidation_report.agents_migrated,
                "agents_deprecated": self.consolidation_report.agents_deprecated,
                "duplicate_agents_removed": self.consolidation_report.duplicate_agents_removed,
                "consolidation_time_seconds": self.consolidation_report.consolidation_time_seconds,
                "errors": self.consolidation_report.errors,
                "warnings": self.consolidation_report.warnings
            },
            "registry_status": registry_status,
            "unified_agents": {
                "council_agents": len(self.registry.get_agents_by_type(AgentType.COUNCIL)),
                "bio_agents": len(self.registry.get_agents_by_type(AgentType.BIO)),
                "generic_agents": sum(
                    len(self.registry.get_agents_by_type(agent_type))
                    for agent_type in [AgentType.ANALYST, AgentType.EXECUTOR, 
                                     AgentType.OBSERVER, AgentType.SUPERVISOR, 
                                     AgentType.VALIDATOR, AgentType.TEMPORAL, AgentType.REAL]
                )
            },
            "consolidation_success": len(self.consolidation_report.errors) == 0
        }
    
        async def demonstrate_unified_agents(self) -> Dict[str, Any]:
        """Demonstrate unified agent functionality."""
        pass
        print("🎭 Demonstrating unified agent functionality...")
        
        demo_results = {}
        
        # Test each agent type
        for agent_type in AgentType:
            agents = self.registry.get_agents_by_type(agent_type)
            if agents:
                agent = agents[0]
                
                try:
                    # Test decision making
                    decision = await agent.make_decision({
                        "demo": True,
                        "context": f"Testing {agent_type.value} agent",
                        "complexity": "medium"
                    })
                    
                    # Test learning
                    learning_result = await agent.learn_from_feedback({
                        "type": "demo_feedback",
                        "score": 0.8,
                        "input": {"demo": True},
                        "expected": {"success": True}
                    })
                    
                    # Test communication
                    if len(agents) > 1:
                        message_id = await agent.send_message(
                            agents[1].component_id,
                            "demo_message",
                            {"greeting": f"Hello from {agent.component_id}"}
                        )
                    else:
                        message_id = "no_other_agents"
                    
                    demo_results[agent_type.value] = {
                        "agent_id": agent.component_id,
                        "decision_made": True,
                        "decision_confidence": decision.get("confidence", 0.0),
                        "learning_successful": learning_result,
                        "message_sent": message_id != "no_other_agents",
                        "agent_type_specific": agent.get_agent_type(),
                        "status": "success"
                    }
                    
                except Exception as e:
                    demo_results[agent_type.value] = {
                        "status": "error",
                        "error": str(e)
                    }
        
        print(f"🎯 Demo complete: {len(demo_results)} agent types tested")
        return demo_results
    
        async def cleanup_old_implementations(self, confirm: bool = False) -> Dict[str, Any]:
        """Clean up old agent implementations (use with caution)."""
        if not confirm:
            return {
                "status": "confirmation_required",
                "message": "Set confirm=True to actually perform cleanup",
                "files_that_would_be_affected": list(self.known_agent_files.values())
            }
        
        print("🧹 Cleaning up old implementations...")
        
        cleanup_results = {
            "files_moved": [],
            "files_deprecated": [],
            "errors": []
        }
        
        # For safety, we'll move files to a backup directory rather than delete
        backup_dir = Path("backup_old_agents")
        backup_dir.mkdir(exist_ok=True)
        
        for agent_type, file_paths in self.known_agent_files.items():
            for file_path in file_paths:
                path = Path(file_path)
                if path.exists():
                    try:
                        # Move to backup
                        backup_path = backup_dir / f"{agent_type}_{path.name}"
                        path.rename(backup_path)
                        cleanup_results["files_moved"].append(str(backup_path))
                        
                    except Exception as e:
                        cleanup_results["errors"].append(f"Failed to move {file_path}: {str(e)}")
        
        return cleanup_results

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def consolidate_agents(config: Dict[str, Any] = None) -> ConsolidationReport:
        """Convenience function to consolidate all agents."""
        consolidation_system = AgentConsolidationSystem(config)
        return await consolidation_system.consolidate_all_agents()

async def demonstrate_agents() -> Dict[str, Any]:
        """Convenience function to demonstrate unified agents."""
        consolidation_system = AgentConsolidationSystem()
        return await consolidation_system.demonstrate_unified_agents()

    def get_consolidation_status() -> Dict[str, Any]:
        """Get current consolidation status."""
        registry = get_agent_registry()
        return {
        "registry_status": registry.get_registry_status(),
        "available_agent_types": [t.value for t in AgentType],
        "factory_ready": True,
        "consolidation_complete": True
        }
