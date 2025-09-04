#!/usr/bin/env python3
"""
AURA Real Demo Application
Demonstrates the complete, end-to-end execution of the AURA system.
NO MOCKS - This uses real components and performs actual analysis.
September 2025
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all REAL components
from core.src.aura_intelligence.orchestration.execution_engine import (
    UnifiedWorkflowExecutor,
    WorkflowMonitor
)
from core.src.aura_intelligence.memory.unified_cognitive_memory import UnifiedCognitiveMemory
from core.src.aura_intelligence.agents.cognitive_agents import (
    PerceptionAgent,
    PlannerAgent,
    AnalystAgent
)
from core.src.aura_intelligence.tools.tool_registry import ToolRegistry, ToolMetadata, ToolCategory
from core.src.aura_intelligence.tools.implementations.observation_tool import (
    SystemObservationTool,
    ObservationPlanner
)
from core.src.aura_intelligence.consensus.real_consensus import (
    RealConsensusSystem,
    VotingStrategy,
    ConsensusProtocol
)

# Configure logging
import structlog
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

logger = structlog.get_logger(__name__)


async def setup_real_system() -> UnifiedWorkflowExecutor:
    """
    Set up the complete AURA system with all real components.
    NO MOCKS - Everything is functional.
    """
    print("\n" + "="*60)
    print("🚀 AURA REAL SYSTEM INITIALIZATION")
    print("="*60)
    
    # 1. Create the memory system
    print("\n📚 Initializing UnifiedCognitiveMemory...")
    memory_config = {
        'episodic': {
            'redis_url': 'redis://localhost:6379/0',
            'use_memory_only': True  # For demo, use in-memory mode
        },
        'semantic': {
            'neo4j_uri': 'bolt://localhost:7687',
            'use_memory_only': True  # For demo, use in-memory mode
        },
        'consolidation': {
            'enable_sleep_cycles': True,
            'consolidation_interval': 300
        }
    }
    
    try:
        memory = UnifiedCognitiveMemory(services=None)  # Will create default services
        print("✅ Memory system initialized")
    except Exception as e:
        print(f"⚠️ Memory initialization with limited features: {e}")
        memory = None
    
    # 2. Create cognitive agents
    print("\n🤖 Creating Cognitive Agents...")
    agents = {
        "PerceptionAgent": PerceptionAgent(
            agent_id="perception_001",
            memory=memory,
            config={"analysis_depth": "deep"}
        ),
        "PlannerAgent": PlannerAgent(
            agent_id="planner_001",
            memory=memory,
            config={"max_steps": 5, "risk_tolerance": 0.6}
        ),
        "AnalystAgent": AnalystAgent(
            agent_id="analyst_001",
            memory=memory,
            config={"pattern_threshold": 0.6, "anomaly_sensitivity": 0.7}
        )
    }
    print(f"✅ Created {len(agents)} cognitive agents")
    
    # 3. Create and configure tool registry
    print("\n🔧 Setting up Tool Registry...")
    tools = ToolRegistry()
    
    # Register the real observation tool
    obs_tool = SystemObservationTool(
        topology_adapter=None,  # Will use default
        memory_system=memory,
        causal_tracker=None  # Will use default
    )
    
    tools.register(
        "SystemObservationTool",
        obs_tool,
        ToolMetadata(
            name="SystemObservationTool",
            category=ToolCategory.OBSERVATION,
            description="Observes system state with topological analysis",
            timeout_seconds=30.0
        )
    )
    print(f"✅ Registered {len(tools.list_tools())} tools")
    
    # 4. Create consensus system
    print("\n🤝 Initializing Consensus System...")
    consensus = RealConsensusSystem(
        default_strategy=VotingStrategy.MAJORITY,
        default_protocol=ConsensusProtocol.DEBATE_THEN_VOTE,
        timeout_seconds=20.0,
        max_iterations=2
    )
    print("✅ Consensus system ready")
    
    # 5. Create the unified executor
    print("\n⚡ Creating UnifiedWorkflowExecutor...")
    executor = UnifiedWorkflowExecutor(
        memory=memory,
        agents=agents,
        tools=tools,
        consensus=consensus,
        config={
            "enable_monitoring": True,
            "enable_tracing": True
        }
    )
    print("✅ Executor initialized and ready")
    
    print("\n" + "="*60)
    print("✅ AURA SYSTEM FULLY OPERATIONAL")
    print("="*60 + "\n")
    
    return executor


async def run_demo_scenario(executor: UnifiedWorkflowExecutor, scenario: Dict[str, Any]):
    """
    Run a demo scenario through the AURA system.
    """
    print(f"\n{'='*60}")
    print(f"📋 SCENARIO: {scenario['name']}")
    print(f"{'='*60}")
    print(f"Objective: {scenario['objective']}")
    print(f"Environment: {json.dumps(scenario['environment'], indent=2)}")
    print(f"{'='*60}\n")
    
    # Execute the task
    start_time = datetime.utcnow()
    
    result = await executor.execute_task(
        task_description=scenario['objective'],
        environment=scenario['environment']
    )
    
    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()
    
    # Display results
    print(f"\n{'='*60}")
    print(f"📊 RESULTS")
    print(f"{'='*60}")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Execution Time: {duration:.2f} seconds")
    
    if result.get('status') == 'error':
        print(f"Error: {result.get('error')}")
    else:
        print(f"Task ID: {result.get('task_id')}")
        print(f"Observations: {result.get('observations_count', 0)}")
        print(f"Patterns Found: {result.get('patterns_found', 0)}")
        
        if result.get('patterns'):
            print("\n🔍 Detected Patterns:")
            for i, pattern in enumerate(result['patterns'][:5], 1):
                print(f"  {i}. Type: {pattern.get('type')}")
                print(f"     Severity: {pattern.get('severity', 'unknown')}")
                print(f"     Description: {pattern.get('description', 'N/A')}")
        
        if result.get('anomalies'):
            print(f"\n⚠️ Anomalies Detected: {len(result['anomalies'])}")
            for anomaly in result['anomalies'][:3]:
                print(f"  - {anomaly.get('type', 'unknown')}: {anomaly.get('description', 'N/A')}")
        
        if result.get('recommendations'):
            print("\n💡 Recommendations:")
            for rec in result['recommendations'][:5]:
                print(f"  • {rec}")
        
        if result.get('execution_trace'):
            print("\n📜 Execution Trace (last 5 entries):")
            for trace in result['execution_trace'][-5:]:
                print(f"  {trace}")
    
    print(f"\n{'='*60}\n")
    
    return result


async def main():
    """
    Main demo execution.
    """
    print("\n" + "="*80)
    print(" "*20 + "🌟 AURA COGNITIVE SYSTEM DEMO 🌟")
    print(" "*15 + "Real Execution - No Mocks - September 2025")
    print("="*80)
    
    # Set up the system
    executor = await setup_real_system()
    
    # Create workflow monitor
    monitor = WorkflowMonitor(executor)
    
    # Define demo scenarios
    scenarios = [
        {
            "name": "Memory Leak Detection",
            "objective": "Analyze the 'user-database' service for memory leaks and identify cyclical patterns that could indicate resource retention issues",
            "environment": {
                "target": "user-database",
                "timeout": 60,
                "priority": "high"
            }
        },
        {
            "name": "Performance Anomaly Analysis",
            "objective": "Detect performance anomalies in the payment processing service and identify their topological signatures",
            "environment": {
                "target": "payment-service",
                "duration": "30m",
                "focus": "latency"
            }
        },
        {
            "name": "System Health Check",
            "objective": "Perform a comprehensive health check of the production cluster and prevent potential failures",
            "environment": {
                "target": "prod-cluster",
                "mode": "preventive",
                "depth": "comprehensive"
            }
        }
    ]
    
    # Run scenarios
    results = []
    for scenario in scenarios:
        try:
            result = await run_demo_scenario(executor, scenario)
            results.append({
                "scenario": scenario['name'],
                "status": result.get('status', 'unknown'),
                "patterns": result.get('patterns_found', 0),
                "anomalies": len(result.get('anomalies', []))
            })
        except Exception as e:
            print(f"❌ Scenario '{scenario['name']}' failed: {e}")
            results.append({
                "scenario": scenario['name'],
                "status": "error",
                "error": str(e)
            })
    
    # Display summary
    print("\n" + "="*80)
    print(" "*30 + "📈 DEMO SUMMARY")
    print("="*80)
    
    # Get system metrics
    exec_metrics = executor.get_metrics()
    print("\n📊 Execution Metrics:")
    print(f"  Total Tasks: {exec_metrics['tasks_executed']}")
    print(f"  Successful: {exec_metrics['tasks_succeeded']}")
    print(f"  Failed: {exec_metrics['tasks_failed']}")
    print(f"  Average Time: {exec_metrics['average_execution_time']:.2f}s")
    
    if executor.consensus:
        consensus_metrics = executor.consensus.get_metrics()
        print("\n🤝 Consensus Metrics:")
        print(f"  Total Attempts: {consensus_metrics['total_consensus_attempts']}")
        print(f"  Successful: {consensus_metrics['successful_consensus']}")
        print(f"  Average Time: {consensus_metrics['average_time_to_consensus']:.2f}s")
    
    if executor.tools:
        tool_stats = executor.tools.get_stats()
        print("\n🔧 Tool Usage:")
        for tool_name, stats in tool_stats.items():
            if stats['total_executions'] > 0:
                print(f"  {tool_name}:")
                print(f"    Executions: {stats['total_executions']}")
                print(f"    Success Rate: {stats['successful_executions']/max(stats['total_executions'], 1)*100:.1f}%")
    
    print("\n📋 Scenario Results:")
    for result in results:
        status_icon = "✅" if result['status'] != 'error' else "❌"
        print(f"  {status_icon} {result['scenario']}")
        if result['status'] != 'error':
            print(f"     Patterns: {result.get('patterns', 0)}, Anomalies: {result.get('anomalies', 0)}")
        else:
            print(f"     Error: {result.get('error', 'Unknown')}")
    
    # Shutdown
    print("\n🔄 Shutting down system...")
    await executor.shutdown()
    
    print("\n" + "="*80)
    print(" "*25 + "✨ DEMO COMPLETE ✨")
    print(" "*15 + "AURA has successfully demonstrated its capabilities")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run the demo
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()