"""
AURA Edge Coordination Demo
Neuromorphic + Byzantine Consensus for IoT/Edge device coordination
"""

import asyncio
import numpy as np
import httpx
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
import random
import math


class EdgeCoordinationSystem:
    """
    Edge device coordination using:
    - Neuromorphic processing for energy efficiency
    - Byzantine consensus for fault tolerance
    - Distributed decision making
    """
    
    def __init__(self):
        self.neuromorphic_url = "http://localhost:8004"
        self.consensus_url = "http://localhost:8003"
        self.lnn_url = "http://localhost:8001"
        
        # Simulated edge devices
        self.edge_devices = self._initialize_edge_devices()
    
    def _initialize_edge_devices(self) -> List[Dict[str, Any]]:
        """Initialize simulated edge devices (e.g., drones, sensors, robots)."""
        devices = []
        
        # Create a swarm of 10 edge devices
        for i in range(10):
            devices.append({
                "id": f"edge_device_{i}",
                "type": random.choice(["drone", "sensor", "robot"]),
                "position": [random.uniform(0, 100), random.uniform(0, 100)],
                "battery": random.uniform(0.3, 1.0),
                "status": "active",
                "capabilities": self._get_device_capabilities(i),
                "spike_rate": 0.0,  # Neuromorphic activity
                "energy_consumed": 0.0
            })
        
        return devices
    
    def _get_device_capabilities(self, device_id: int) -> List[str]:
        """Get device capabilities based on type."""
        base_capabilities = ["communication", "processing"]
        
        if device_id % 3 == 0:
            base_capabilities.extend(["high_compute", "ml_inference"])
        if device_id % 2 == 0:
            base_capabilities.extend(["camera", "object_detection"])
        if device_id < 3:
            base_capabilities.extend(["long_range_comm", "coordinator"])
        
        return base_capabilities
    
    async def coordinate_swarm_mission(self, mission: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate a swarm mission using neuromorphic processing and consensus.
        
        Args:
            mission: Mission parameters including objective, constraints, etc.
        
        Returns:
            Coordination results with task assignments and energy metrics
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            print("🤖 AURA Edge Coordination System")
            print("=" * 50)
            print(f"Mission: {mission['objective']}")
            print(f"Devices: {len(self.edge_devices)} edge nodes")
            print("\n" + "-" * 50 + "\n")
            
            # Step 1: Neuromorphic processing for energy-efficient task allocation
            print("⚡ Step 1: Neuromorphic task allocation...")
            
            # Convert mission and device states to spike trains
            spike_data = self._encode_to_spikes(mission, self.edge_devices)
            
            # Process with neuromorphic system
            neuro_response = await self._process_neuromorphic(client, spike_data)
            
            # Decode task assignments from spike patterns
            task_assignments = self._decode_spike_assignments(
                neuro_response,
                self.edge_devices,
                mission
            )
            
            print(f"   Energy efficiency: {neuro_response.get('energy_efficiency', 0):.1%}")
            print(f"   Spike utilization: {neuro_response.get('spike_rate', 0):.3f} Hz")
            
            # Step 2: Byzantine consensus for fault-tolerant coordination
            print("\n🛡️ Step 2: Byzantine consensus on task allocation...")
            
            # Each device votes on the proposed allocation
            consensus_result = await self._byzantine_task_consensus(
                client,
                task_assignments,
                self.edge_devices
            )
            
            print(f"   Consensus achieved: {consensus_result['accepted']}")
            print(f"   Agreement level: {consensus_result['confidence']:.1%}")
            
            # Step 3: Optimize coordination with LNN reasoning
            print("\n🧠 Step 3: Logical optimization of coordination...")
            
            optimization_result = await self._optimize_with_lnn(
                client,
                task_assignments,
                mission,
                self.edge_devices
            )
            
            final_assignments = optimization_result['optimized_assignments']
            
            # Step 4: Simulate mission execution
            print("\n🚀 Step 4: Simulating mission execution...")
            
            execution_results = self._simulate_execution(
                final_assignments,
                mission,
                neuro_response.get('energy_per_spike', 0.001)
            )
            
            return {
                "timestamp": datetime.now().isoformat(),
                "mission": mission,
                "coordination_results": {
                    "task_assignments": final_assignments,
                    "consensus_achieved": consensus_result['accepted'],
                    "consensus_confidence": consensus_result['confidence'],
                    "energy_efficiency": neuro_response.get('energy_efficiency', 0),
                    "total_energy_consumed": execution_results['total_energy'],
                    "mission_success": execution_results['success'],
                    "completion_time": execution_results['completion_time']
                },
                "device_metrics": execution_results['device_metrics'],
                "neuromorphic_stats": {
                    "spike_rate": neuro_response.get('spike_rate', 0),
                    "energy_per_spike": neuro_response.get('energy_per_spike', 0),
                    "processing_time_ms": neuro_response.get('processing_time_ms', 0)
                }
            }
    
    def _encode_to_spikes(
        self,
        mission: Dict[str, Any],
        devices: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Encode mission and device states into spike trains."""
        # Simulate spike encoding based on device states and mission requirements
        spike_trains = []
        
        for device in devices:
            # Device state encoding
            battery_spikes = self._generate_spike_train(device['battery'], base_rate=10)
            position_spikes = self._generate_spike_train(
                np.linalg.norm(device['position']), base_rate=5
            )
            
            # Capability encoding
            capability_spikes = len(device['capabilities']) * 2
            
            spike_trains.append({
                "device_id": device['id'],
                "spike_pattern": battery_spikes + position_spikes + [capability_spikes],
                "temporal_pattern": self._generate_temporal_pattern(device)
            })
        
        return {
            "spike_trains": spike_trains,
            "mission_encoding": self._encode_mission_requirements(mission),
            "time_window_ms": 100
        }
    
    def _generate_spike_train(self, value: float, base_rate: float = 10) -> List[float]:
        """Generate spike train based on value (rate coding)."""
        rate = base_rate * value
        n_spikes = int(rate)
        return [1.0] * n_spikes + [0.0] * (10 - n_spikes)
    
    def _generate_temporal_pattern(self, device: Dict[str, Any]) -> List[float]:
        """Generate temporal spike pattern based on device state."""
        # Simulate complex temporal patterns
        pattern = []
        for i in range(20):
            # Create device-specific temporal signature
            phase = (hash(device['id']) % 10) / 10.0
            spike_prob = 0.5 + 0.3 * math.sin(2 * math.pi * i / 20 + phase)
            pattern.append(1.0 if random.random() < spike_prob else 0.0)
        return pattern
    
    def _encode_mission_requirements(self, mission: Dict[str, Any]) -> Dict[str, Any]:
        """Encode mission requirements as constraints."""
        return {
            "priority_encoding": self._priority_to_spikes(mission.get('priority', 'medium')),
            "spatial_constraints": mission.get('target_area', [[0, 0], [100, 100]]),
            "capability_requirements": mission.get('required_capabilities', []),
            "time_constraint_ms": mission.get('max_duration_ms', 60000)
        }
    
    def _priority_to_spikes(self, priority: str) -> List[float]:
        """Convert priority level to spike pattern."""
        patterns = {
            "low": [1, 0, 0, 0, 0],
            "medium": [1, 1, 0, 1, 0],
            "high": [1, 1, 1, 1, 0],
            "critical": [1, 1, 1, 1, 1]
        }
        return patterns.get(priority, patterns["medium"])
    
    async def _process_neuromorphic(
        self,
        client: httpx.AsyncClient,
        spike_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process spike data through neuromorphic system."""
        # In a real implementation, this would call the neuromorphic service
        # For demo, we simulate the response
        
        # Simulate neuromorphic processing
        total_spikes = sum(
            sum(train['spike_pattern']) + sum(train['temporal_pattern'])
            for train in spike_data['spike_trains']
        )
        
        processing_time = random.uniform(10, 50)  # ms
        energy_per_spike = 0.001  # mJ (millijoules)
        
        return {
            "spike_rate": total_spikes / (spike_data['time_window_ms'] / 1000),
            "energy_per_spike": energy_per_spike,
            "total_energy": total_spikes * energy_per_spike,
            "processing_time_ms": processing_time,
            "energy_efficiency": 0.85 + random.uniform(-0.1, 0.1),  # 75-95% efficiency
            "task_allocation_spikes": self._generate_allocation_spikes(spike_data)
        }
    
    def _generate_allocation_spikes(self, spike_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate task allocation based on neuromorphic processing."""
        allocations = []
        
        for train in spike_data['spike_trains']:
            # Simulate neuromorphic decision making
            device_score = sum(train['spike_pattern']) / len(train['spike_pattern'])
            temporal_score = sum(train['temporal_pattern']) / len(train['temporal_pattern'])
            
            allocations.append({
                "device_id": train['device_id'],
                "allocation_score": (device_score + temporal_score) / 2,
                "spike_response": random.uniform(0.3, 1.0)
            })
        
        return allocations
    
    def _decode_spike_assignments(
        self,
        neuro_response: Dict[str, Any],
        devices: List[Dict[str, Any]],
        mission: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Decode spike patterns into task assignments."""
        assignments = []
        allocation_spikes = neuro_response.get('task_allocation_spikes', [])
        
        # Sort devices by allocation score
        sorted_allocations = sorted(
            allocation_spikes,
            key=lambda x: x['allocation_score'],
            reverse=True
        )
        
        # Assign tasks based on neuromorphic output
        available_tasks = mission.get('tasks', self._generate_default_tasks(mission))
        
        for i, allocation in enumerate(sorted_allocations[:len(available_tasks)]):
            device = next(d for d in devices if d['id'] == allocation['device_id'])
            task = available_tasks[i % len(available_tasks)]
            
            assignments.append({
                "device_id": device['id'],
                "task_id": task['id'],
                "task_type": task['type'],
                "priority": allocation['allocation_score'],
                "estimated_energy": task.get('energy_cost', 1.0) * (2 - device['battery']),
                "assigned_by": "neuromorphic_processor"
            })
        
        return assignments
    
    def _generate_default_tasks(self, mission: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate default tasks based on mission objective."""
        tasks = []
        
        if "surveillance" in mission['objective']:
            tasks.extend([
                {"id": "task_scan_1", "type": "area_scan", "energy_cost": 2.0},
                {"id": "task_patrol_1", "type": "patrol", "energy_cost": 3.0},
                {"id": "task_monitor_1", "type": "monitor", "energy_cost": 1.0}
            ])
        
        if "search" in mission['objective']:
            tasks.extend([
                {"id": "task_search_1", "type": "search_pattern", "energy_cost": 2.5},
                {"id": "task_detect_1", "type": "object_detection", "energy_cost": 1.5}
            ])
        
        if "coordinate" in mission['objective']:
            tasks.extend([
                {"id": "task_comm_1", "type": "relay_communication", "energy_cost": 0.5},
                {"id": "task_sync_1", "type": "synchronize", "energy_cost": 0.3}
            ])
        
        return tasks if tasks else [
            {"id": "task_default_1", "type": "idle", "energy_cost": 0.1}
        ]
    
    async def _byzantine_task_consensus(
        self,
        client: httpx.AsyncClient,
        task_assignments: List[Dict[str, Any]],
        devices: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run Byzantine consensus on task assignments."""
        # Simulate Byzantine consensus voting
        votes = {}
        
        for device in devices:
            # Each device votes on the assignment
            # Some devices might be faulty (Byzantine)
            is_byzantine = random.random() < 0.1  # 10% Byzantine nodes
            
            if is_byzantine:
                vote = random.choice([True, False])
            else:
                # Honest nodes vote based on assignment quality
                device_assignment = next(
                    (a for a in task_assignments if a['device_id'] == device['id']),
                    None
                )
                
                if device_assignment:
                    # Vote yes if energy cost is reasonable
                    vote = device_assignment['estimated_energy'] < device['battery'] * 5
                else:
                    vote = True  # No assignment is fine
            
            votes[device['id']] = vote
        
        # Calculate consensus
        yes_votes = sum(1 for vote in votes.values() if vote)
        total_votes = len(votes)
        
        # Byzantine consensus requires > 2/3 majority
        consensus_threshold = 2/3
        accepted = yes_votes / total_votes > consensus_threshold
        
        return {
            "accepted": accepted,
            "confidence": yes_votes / total_votes,
            "yes_votes": yes_votes,
            "total_votes": total_votes,
            "byzantine_threshold": consensus_threshold
        }
    
    async def _optimize_with_lnn(
        self,
        client: httpx.AsyncClient,
        task_assignments: List[Dict[str, Any]],
        mission: Dict[str, Any],
        devices: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Optimize coordination using LNN reasoning."""
        # Prepare optimization features
        features = []
        
        for assignment in task_assignments:
            device = next(d for d in devices if d['id'] == assignment['device_id'])
            
            features.append([
                assignment['priority'],
                assignment['estimated_energy'],
                device['battery'],
                len(device['capabilities']) / 10.0,
                1.0 if assignment['task_type'] in ['patrol', 'search_pattern'] else 0.0
            ])
        
        # In real implementation, would call LNN service
        # For demo, simulate optimization
        optimized_assignments = task_assignments.copy()
        
        # Rebalance based on battery levels
        for assignment in optimized_assignments:
            device = next(d for d in devices if d['id'] == assignment['device_id'])
            
            if device['battery'] < 0.3:
                # Low battery - reduce workload
                assignment['priority'] *= 0.5
                assignment['estimated_energy'] *= 0.7
        
        return {
            "optimized_assignments": optimized_assignments,
            "optimization_score": 0.85,
            "reasoning": "Balanced workload based on battery levels and capabilities"
        }
    
    def _simulate_execution(
        self,
        assignments: List[Dict[str, Any]],
        mission: Dict[str, Any],
        energy_per_spike: float
    ) -> Dict[str, Any]:
        """Simulate mission execution and calculate metrics."""
        total_energy = 0
        device_metrics = {}
        
        for assignment in assignments:
            device_id = assignment['device_id']
            device = next(d for d in self.edge_devices if d['id'] == device_id)
            
            # Calculate energy consumption
            task_energy = assignment['estimated_energy']
            spike_energy = random.uniform(0.1, 0.5)  # Neuromorphic overhead
            total_device_energy = task_energy + spike_energy
            
            total_energy += total_device_energy
            
            # Update device metrics
            device_metrics[device_id] = {
                "task": assignment['task_type'],
                "energy_consumed": total_device_energy,
                "battery_remaining": max(0, device['battery'] - total_device_energy / 10),
                "spike_rate": random.uniform(10, 100),  # Hz
                "efficiency": random.uniform(0.7, 0.95)
            }
        
        # Determine mission success
        success_criteria = len([m for m in device_metrics.values() if m['battery_remaining'] > 0.1])
        mission_success = success_criteria >= len(self.edge_devices) * 0.7
        
        return {
            "total_energy": total_energy,
            "device_metrics": device_metrics,
            "success": mission_success,
            "completion_time": random.uniform(30, 120)  # seconds
        }


async def simulate_edge_coordination():
    """Run edge coordination simulation."""
    coordinator = EdgeCoordinationSystem()
    
    # Define mission
    mission = {
        "objective": "distributed surveillance and search",
        "priority": "high",
        "target_area": [[20, 20], [80, 80]],
        "required_capabilities": ["camera", "object_detection", "communication"],
        "max_duration_ms": 120000,  # 2 minutes
        "tasks": [
            {"id": "area_scan_north", "type": "area_scan", "energy_cost": 2.0},
            {"id": "area_scan_south", "type": "area_scan", "energy_cost": 2.0},
            {"id": "patrol_perimeter", "type": "patrol", "energy_cost": 3.0},
            {"id": "object_search", "type": "search_pattern", "energy_cost": 2.5},
            {"id": "comm_relay", "type": "relay_communication", "energy_cost": 0.5}
        ]
    }
    
    # Run coordination
    try:
        result = await coordinator.coordinate_swarm_mission(mission)
        
        print("\n📊 COORDINATION RESULTS")
        print("=" * 50)
        
        coord = result['coordination_results']
        print(f"✅ Mission Success: {coord['mission_success']}")
        print(f"⚡ Energy Efficiency: {coord['energy_efficiency']:.1%}")
        print(f"🔋 Total Energy: {coord['total_energy_consumed']:.2f} mJ")
        print(f"⏱️ Completion Time: {coord['completion_time']:.1f} seconds")
        
        print(f"\n🛡️ Byzantine Consensus:")
        print(f"   - Achieved: {coord['consensus_achieved']}")
        print(f"   - Confidence: {coord['consensus_confidence']:.1%}")
        
        print(f"\n🧠 Neuromorphic Stats:")
        neuro = result['neuromorphic_stats']
        print(f"   - Spike Rate: {neuro['spike_rate']:.1f} Hz")
        print(f"   - Energy/Spike: {neuro['energy_per_spike']:.4f} mJ")
        print(f"   - Processing Time: {neuro['processing_time_ms']:.1f} ms")
        
        print(f"\n📋 Task Assignments:")
        for assignment in coord['task_assignments'][:5]:  # Show first 5
            print(f"   - {assignment['device_id']}: {assignment['task_type']} (energy: {assignment['estimated_energy']:.2f})")
        
        print(f"\n🔋 Device Status:")
        for device_id, metrics in list(result['device_metrics'].items())[:5]:
            print(f"   - {device_id}: {metrics['task']} | Battery: {metrics['battery_remaining']:.1%} | Efficiency: {metrics['efficiency']:.1%}")
        
    except Exception as e:
        print(f"❌ Error during coordination: {e}")
        print("Make sure all AURA services are running!")


if __name__ == "__main__":
    asyncio.run(simulate_edge_coordination())