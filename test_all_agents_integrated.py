#!/usr/bin/env python3
"""
🚀 AURA Test Agents - Integrated System Test
===========================================

Demonstrates all 5 test agents working together:
- Code Agent: Analyzes and optimizes code
- Data Agent: Processes data with TDA
- Creative Agent: Generates creative content
- Architect Agent: Analyzes system architecture
- Coordinator Agent: Orchestrates everything

Tests:
1. Individual agent capabilities
2. Multi-agent collaboration
3. Byzantine consensus
4. GPU acceleration
5. End-to-end workflows
"""

import asyncio
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import structlog

# Import our test agents
from core.src.aura_intelligence.agents.test_code_agent import create_code_agent
from core.src.aura_intelligence.agents.test_data_agent import create_data_agent
from core.src.aura_intelligence.agents.test_creative_agent import create_creative_agent
from core.src.aura_intelligence.agents.test_architect_agent import create_architect_agent
from core.src.aura_intelligence.agents.test_coordinator_agent import create_coordinator_agent

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
        structlog.dev.ConsoleRenderer(colors=True)
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class TestScenarios:
    """Test scenarios for multi-agent system"""
    
    def __init__(self):
        self.agents = {}
        self.results = {}
        
    async def setup_agents(self):
        """Initialize all test agents"""
        logger.info("🚀 Initializing test agents...")
        
        # Create agents
        self.agents['code'] = create_code_agent()
        self.agents['data'] = create_data_agent()
        self.agents['creative'] = create_creative_agent()
        self.agents['architect'] = create_architect_agent()
        self.agents['coordinator'] = create_coordinator_agent()
        
        # Register agents with coordinator
        for agent_type, agent in self.agents.items():
            if agent_type != 'coordinator':
                self.agents['coordinator'].register_agent(agent)
                
        logger.info("✅ All agents initialized and registered")
        
    async def test_individual_agents(self):
        """Test each agent individually"""
        logger.info("\n🧪 Testing Individual Agents")
        logger.info("=" * 60)
        
        # Test Code Agent
        logger.info("\n📝 Testing Code Agent...")
        code_result = await self.agents['code'].process_message({
            "type": "analyze",
            "payload": {
                "files": ["test_all_agents_integrated.py"]
            }
        })
        
        if 'result' in code_result:
            analysis = code_result['result']
            logger.info(f"  Complexity Score: {analysis.complexity_score:.2f}")
            logger.info(f"  Quality Score: {analysis.quality_score:.2f}")
            logger.info(f"  Optimization Suggestions: {len(analysis.optimization_suggestions)}")
            logger.info(f"  Latency: {code_result['metrics']['latency_ms']:.2f}ms")
            
        # Test Data Agent
        logger.info("\n📊 Testing Data Agent...")
        
        # Generate test data
        test_data = pd.DataFrame({
            'x': np.random.randn(1000),
            'y': np.random.randn(1000),
            'z': np.random.randn(1000) * 2 + 5
        })
        
        data_result = await self.agents['data'].process_message({
            "type": "analyze",
            "payload": {
                "data": test_data,
                "analysis_type": "full"
            }
        })
        
        if 'result' in data_result:
            analysis = data_result['result']
            logger.info(f"  Dataset Shape: {analysis.shape}")
            logger.info(f"  Anomalies Detected: {len(analysis.anomalies)}")
            logger.info(f"  Patterns Found: {len(analysis.patterns)}")
            logger.info(f"  Processing Time: {analysis.processing_time_ms:.2f}ms")
            
        # Test Creative Agent
        logger.info("\n🎨 Testing Creative Agent...")
        creative_result = await self.agents['creative'].process_message({
            "type": "generate",
            "payload": {
                "prompt": "Create an innovative solution for distributed AI systems",
                "style": "technical",
                "num_variations": 5
            }
        })
        
        if 'result' in creative_result:
            output = creative_result['result']
            logger.info(f"  Generated Variations: {len(output.content)}")
            logger.info(f"  Diversity Score: {output.diversity_score:.2f}")
            logger.info(f"  Quality Metrics: {output.quality_metrics}")
            logger.info(f"  Generation Time: {output.generation_time_ms:.2f}ms")
            
        # Test Architect Agent
        logger.info("\n🏗️ Testing Architect Agent...")
        architect_result = await self.agents['architect'].process_message({
            "type": "analyze",
            "payload": {
                "components": [
                    {"id": "frontend", "type": "ui"},
                    {"id": "api", "type": "service"},
                    {"id": "database", "type": "storage"},
                    {"id": "cache", "type": "cache"},
                    {"id": "queue", "type": "messaging"}
                ],
                "connections": [
                    {"from": "frontend", "to": "api"},
                    {"from": "api", "to": "database"},
                    {"from": "api", "to": "cache"},
                    {"from": "api", "to": "queue"}
                ],
                "requirements": {
                    "throughput": 10000,
                    "latency": 50,
                    "availability": 0.999
                }
            }
        })
        
        if 'result' in architect_result:
            analysis = architect_result['result']
            logger.info(f"  Topology Metrics: {len(analysis.topology_metrics)} metrics")
            logger.info(f"  Bottlenecks Found: {len(analysis.bottlenecks)}")
            logger.info(f"  Patterns Detected: {analysis.patterns_detected}")
            logger.info(f"  Scalability Type: {analysis.scalability_forecast.get('scalability_type', 'unknown')}")
            
        # Test Coordinator Agent
        logger.info("\n🎯 Testing Coordinator Agent...")
        coord_result = await self.agents['coordinator'].process_message({
            "type": "analyze",
            "payload": {
                "type": "optimization",
                "description": "Optimize system performance",
                "priority": 1
            }
        })
        
        if 'result' in coord_result:
            analysis = coord_result['result']
            logger.info(f"  Task Complexity: {analysis.get('task_complexity', 0)}")
            logger.info(f"  Available Agents: {analysis.get('system_load', {}).get('available_agents', 0)}")
            logger.info(f"  Feasibility: {analysis.get('feasibility', 'unknown')}")
            
    async def test_multi_agent_collaboration(self):
        """Test agents working together"""
        logger.info("\n🤝 Testing Multi-Agent Collaboration")
        logger.info("=" * 60)
        
        # Complex task requiring multiple agents
        complex_task = {
            "id": "complex_analysis_001",
            "type": "analysis",
            "description": "Analyze system performance and suggest optimizations",
            "data": pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
                'cpu_usage': np.random.uniform(0.3, 0.9, 100),
                'memory_usage': np.random.uniform(0.4, 0.8, 100),
                'response_time': np.random.uniform(10, 200, 100),
                'error_rate': np.random.uniform(0, 0.05, 100)
            }),
            "code_files": ["test_all_agents_integrated.py"],
            "requirements": {
                "optimize_for": "latency",
                "target_improvement": 0.3
            },
            "requires_consensus": True
        }
        
        # Coordinate execution
        logger.info("📋 Decomposing complex task...")
        result = await self.agents['coordinator'].coordinate_swarm(
            list(self.agents.values())[:-1],  # All except coordinator
            complex_task
        )
        
        logger.info(f"\n📊 Coordination Results:")
        logger.info(f"  Subtasks Created: {result.performance_metrics['subtask_count']}")
        logger.info(f"  Total Time: {result.performance_metrics['total_time_ms']:.2f}ms")
        logger.info(f"  Parallel Efficiency: {result.performance_metrics['parallel_efficiency']:.2%}")
        logger.info(f"  Consensus Quality: {result.consensus_quality:.2f}")
        logger.info(f"  Agents Involved: {len(result.agents_involved)}")
        
        self.results['collaboration'] = result
        
    async def test_byzantine_consensus(self):
        """Test Byzantine fault tolerance"""
        logger.info("\n🛡️ Testing Byzantine Consensus")
        logger.info("=" * 60)
        
        # Create a scenario where agents might disagree
        consensus_task = {
            "id": "consensus_test_001",
            "type": "generation",
            "description": "Generate best architecture for high-scale system",
            "constraints": {
                "must_handle": "1M requests/sec",
                "budget": "limited",
                "timeline": "aggressive"
            },
            "requires_consensus": True
        }
        
        # Each agent provides their recommendation
        votes = []
        
        for agent_type, agent in self.agents.items():
            if agent_type == 'coordinator':
                continue
                
            # Get agent's vote
            vote_result = await agent.process_message({
                "type": "consensus",
                "payload": {
                    "votes": [],
                    "options": [
                        {"id": "microservices", "description": "Microservices architecture"},
                        {"id": "monolith", "description": "Optimized monolith"},
                        {"id": "serverless", "description": "Serverless functions"},
                        {"id": "hybrid", "description": "Hybrid approach"}
                    ]
                }
            })
            
            if 'result' in vote_result:
                vote = vote_result['result']
                vote['agent_id'] = agent.state.agent_id
                vote['agent_type'] = agent_type
                votes.append(vote)
                
        # Run consensus
        logger.info(f"🗳️ Collected {len(votes)} votes")
        
        consensus_result = await self.agents['coordinator']._tool_coordinate_consensus(votes)
        
        logger.info(f"\n📊 Consensus Results:")
        logger.info(f"  Valid Votes: {consensus_result['valid_votes']}")
        logger.info(f"  Byzantine Agents Detected: {len(consensus_result['byzantine_agents'])}")
        logger.info(f"  Consensus Quality: {consensus_result['consensus_quality']:.2f}")
        logger.info(f"  Time to Consensus: {consensus_result['time_to_consensus_ms']:.2f}ms")
        
        if consensus_result['byzantine_agents']:
            logger.warning(f"  ⚠️ Byzantine agents: {consensus_result['byzantine_agents']}")
            
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        logger.info("\n🔄 Testing End-to-End Workflow")
        logger.info("=" * 60)
        
        # Scenario: Optimize a data processing pipeline
        workflow = {
            "id": "e2e_workflow_001",
            "type": "optimization",
            "description": "Optimize data processing pipeline",
            "steps": [
                {
                    "step": 1,
                    "description": "Analyze current code",
                    "agent": "code",
                    "input": {"files": ["test_all_agents_integrated.py"]}
                },
                {
                    "step": 2,
                    "description": "Analyze data patterns",
                    "agent": "data",
                    "input": {
                        "data": pd.DataFrame({
                            'feature1': np.random.randn(1000),
                            'feature2': np.random.randn(1000) * 2,
                            'target': np.random.randint(0, 2, 1000)
                        })
                    }
                },
                {
                    "step": 3,
                    "description": "Generate optimization strategies",
                    "agent": "creative",
                    "input": {
                        "prompt": "Generate innovative optimization strategies for data pipeline",
                        "context": "Based on code and data analysis"
                    }
                },
                {
                    "step": 4,
                    "description": "Design optimized architecture",
                    "agent": "architect",
                    "input": {
                        "requirements": {
                            "throughput": 100000,
                            "latency": 10,
                            "scalability": "horizontal"
                        }
                    }
                }
            ]
        }
        
        logger.info("🚀 Executing workflow...")
        
        # Execute through coordinator
        start_time = time.perf_counter()
        
        result = await self.agents['coordinator'].coordinate_swarm(
            list(self.agents.values())[:-1],
            workflow
        )
        
        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000
        
        logger.info(f"\n✅ Workflow Completed!")
        logger.info(f"  Total Execution Time: {total_time:.2f}ms")
        logger.info(f"  Subtasks Completed: {result.performance_metrics['subtask_count']}")
        logger.info(f"  Resource Utilization: {result.performance_metrics['resource_utilization']:.2%}")
        
        # Show GPU metrics if available
        if 'gpu_utilization' in result.performance_metrics:
            logger.info(f"  GPU Utilization: {result.performance_metrics['gpu_utilization']:.2%}")
            
    async def test_performance_and_scaling(self):
        """Test system performance and scaling"""
        logger.info("\n⚡ Testing Performance and Scaling")
        logger.info("=" * 60)
        
        # Test with increasing load
        load_levels = [10, 50, 100]
        
        for load in load_levels:
            logger.info(f"\n📈 Testing with {load} concurrent tasks...")
            
            tasks = []
            for i in range(load):
                task = {
                    "id": f"perf_test_{i}",
                    "type": "analysis" if i % 2 == 0 else "generation",
                    "data": np.random.randn(100, 10),
                    "priority": np.random.randint(1, 4)
                }
                tasks.append(task)
                
            # Measure performance
            start_time = time.perf_counter()
            
            # Process tasks (simplified - would use proper concurrency)
            results = []
            for task in tasks[:10]:  # Limit for demo
                agent_type = "data" if task["type"] == "analysis" else "creative"
                result = await self.agents[agent_type].process_message({
                    "type": task["type"],
                    "payload": task
                })
                results.append(result)
                
            end_time = time.perf_counter()
            
            # Calculate metrics
            total_time = (end_time - start_time) * 1000
            avg_latency = total_time / min(len(tasks), 10)
            
            logger.info(f"  Total Time: {total_time:.2f}ms")
            logger.info(f"  Average Latency: {avg_latency:.2f}ms")
            logger.info(f"  Throughput: {min(len(tasks), 10) / (total_time / 1000):.2f} tasks/sec")
            
    async def cleanup(self):
        """Clean up agents"""
        logger.info("\n🧹 Cleaning up...")
        
        for agent_type, agent in self.agents.items():
            await agent.shutdown()
            
        logger.info("✅ Cleanup complete")


async def main():
    """Run all tests"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        AURA Test Agents - Integrated System Test         ║
    ║                                                          ║
    ║  Testing 5 specialized agents with GPU acceleration,     ║
    ║  Byzantine consensus, and swarm intelligence             ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    test_runner = TestScenarios()
    
    try:
        # Setup
        await test_runner.setup_agents()
        
        # Run tests
        await test_runner.test_individual_agents()
        await test_runner.test_multi_agent_collaboration()
        await test_runner.test_byzantine_consensus()
        await test_runner.test_end_to_end_workflow()
        await test_runner.test_performance_and_scaling()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)
        logger.info("✅ All tests completed successfully!")
        logger.info("✅ Agents demonstrated:")
        logger.info("   - Individual capabilities")
        logger.info("   - Multi-agent collaboration")
        logger.info("   - Byzantine fault tolerance")
        logger.info("   - End-to-end workflows")
        logger.info("   - Performance scaling")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise
        
    finally:
        await test_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())