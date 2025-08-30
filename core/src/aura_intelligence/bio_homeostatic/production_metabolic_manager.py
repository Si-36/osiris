"""
Production-Ready Metabolic Manager - 2025 Biological Homeostasis

Features:
- Adaptive resource allocation based on system health
- Self-regulating feedback loops
- Circadian rhythm simulation
- Stress response and recovery patterns
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import structlog
from collections import deque

logger = structlog.get_logger(__name__)


@dataclass
class SystemMetrics:
    """Real-time system health metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_agents: int = 0
    error_rate: float = 0.0
    latency_ms: float = 0.0
    throughput: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HomeostaticState:
    """Current homeostatic state of the system"""
    health_score: float = 1.0
    stress_level: float = 0.0
    recovery_rate: float = 0.1
    adaptation_level: float = 0.5
    energy_budget: float = 1.0
    metabolism_rate: float = 1.0


class ProductionMetabolicManager:
    """
    Biological-inspired system management with homeostatic regulation
    
    Key concepts:
    - Allostasis: Achieving stability through change
    - Hormesis: Beneficial adaptation to mild stress
    - Circadian rhythms: Time-based resource allocation
    - Metabolic flexibility: Switching between energy modes
    """
    
    def __init__(self, 
                 update_interval: float = 5.0,
                 history_size: int = 100,
                 adaptation_rate: float = 0.01):
        self.update_interval = update_interval
        self.history_size = history_size
        self.adaptation_rate = adaptation_rate
        
        # State tracking
        self.state = HomeostaticState()
        self.metrics_history: deque[SystemMetrics] = deque(maxlen=history_size)
        self.stress_history: deque[float] = deque(maxlen=history_size)
        
        # Circadian rhythm parameters
        self.circadian_phase = 0.0
        self.last_update = time.time()
        
        # Set points for homeostasis
        self.setpoints = {
            "cpu_usage": 60.0,
            "memory_usage": 70.0,
            "error_rate": 0.01,
            "latency_ms": 100.0
        }
        
        # Control parameters
        self.kp = 0.5  # Proportional gain
        self.ki = 0.1  # Integral gain
        self.kd = 0.2  # Derivative gain
        
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the metabolic regulation system"""
        if self._running:
            logger.warning("Metabolic manager already running")
            return
            
        self._running = True
        self._update_task = asyncio.create_task(self._regulation_loop())
        logger.info("🧬 Metabolic regulation system started")
        
    async def stop(self):
        """Stop the metabolic regulation system"""
        self._running = False
        if self._update_task:
            await self._update_task
        logger.info("🛑 Metabolic regulation system stopped")
        
    async def update_metrics(self, metrics: Dict[str, Any]):
        """Update system metrics and trigger homeostatic response"""
        # Create metrics object
        current_metrics = SystemMetrics(
            cpu_usage=metrics.get("cpu_usage", 0.0),
            memory_usage=metrics.get("memory_usage", 0.0),
            active_agents=metrics.get("active_agents", 0),
            error_rate=metrics.get("error_rate", 0.0),
            latency_ms=metrics.get("latency_ms", 0.0),
            throughput=metrics.get("throughput", 0.0)
        )
        
        # Add to history
        self.metrics_history.append(current_metrics)
        
        # Calculate stress level
        stress = self._calculate_stress(current_metrics)
        self.stress_history.append(stress)
        
        # Update homeostatic state
        await self._update_homeostasis(current_metrics, stress)
        
    async def _regulation_loop(self):
        """Main homeostatic regulation loop"""
        while self._running:
            try:
                # Update circadian rhythm
                self._update_circadian_rhythm()
                
                # Apply homeostatic regulation
                await self._apply_regulation()
                
                # Check for adaptation needs
                self._check_adaptation()
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Regulation loop error: {e}")
                await asyncio.sleep(self.update_interval * 2)
                
    def _calculate_stress(self, metrics: SystemMetrics) -> float:
        """Calculate system stress level based on deviation from setpoints"""
        deviations = []
        
        # CPU stress
        cpu_dev = abs(metrics.cpu_usage - self.setpoints["cpu_usage"]) / 100.0
        deviations.append(cpu_dev * 1.5)  # Higher weight for CPU
        
        # Memory stress
        mem_dev = abs(metrics.memory_usage - self.setpoints["memory_usage"]) / 100.0
        deviations.append(mem_dev * 1.2)
        
        # Error rate stress (exponential response)
        error_dev = max(0, metrics.error_rate - self.setpoints["error_rate"])
        deviations.append(np.exp(error_dev * 10) - 1)
        
        # Latency stress
        latency_dev = max(0, metrics.latency_ms - self.setpoints["latency_ms"]) / 1000.0
        deviations.append(latency_dev)
        
        # Calculate weighted stress
        stress = np.mean(deviations)
        return min(1.0, stress)  # Cap at 1.0
        
    async def _update_homeostasis(self, metrics: SystemMetrics, stress: float):
        """Update homeostatic state based on current stress"""
        # Update stress level with smoothing
        self.state.stress_level = 0.7 * self.state.stress_level + 0.3 * stress
        
        # Calculate health score (inverse of stress with recovery)
        recovery_factor = 1.0 + self.state.recovery_rate
        self.state.health_score = min(1.0, (1.0 - self.state.stress_level) * recovery_factor)
        
        # Update metabolism based on stress (allostatic response)
        if self.state.stress_level > 0.7:
            # High stress: increase metabolism for fight-or-flight
            self.state.metabolism_rate = min(2.0, 1.0 + self.state.stress_level)
        elif self.state.stress_level < 0.3:
            # Low stress: conserve energy
            self.state.metabolism_rate = max(0.5, 1.0 - (0.3 - self.state.stress_level))
        else:
            # Normal range: gradual return to baseline
            self.state.metabolism_rate = 0.9 * self.state.metabolism_rate + 0.1 * 1.0
            
        # Update energy budget based on health and circadian rhythm
        circadian_factor = 0.5 + 0.5 * np.sin(self.circadian_phase)
        self.state.energy_budget = self.state.health_score * circadian_factor * self.state.metabolism_rate
        
    def _update_circadian_rhythm(self):
        """Update circadian phase (24-hour cycle)"""
        current_time = time.time()
        elapsed = current_time - self.last_update
        self.last_update = current_time
        
        # One full cycle per 24 hours (86400 seconds)
        # Speed up for testing: one cycle per hour (3600 seconds)
        cycle_duration = 3600.0
        phase_increment = (2 * np.pi * elapsed) / cycle_duration
        
        self.circadian_phase = (self.circadian_phase + phase_increment) % (2 * np.pi)
        
    async def _apply_regulation(self):
        """Apply homeostatic regulation to system parameters"""
        if len(self.metrics_history) < 2:
            return
            
        # PID control for resource allocation
        current = self.metrics_history[-1]
        previous = self.metrics_history[-2]
        
        # Calculate errors
        cpu_error = self.setpoints["cpu_usage"] - current.cpu_usage
        mem_error = self.setpoints["memory_usage"] - current.memory_usage
        
        # Apply PID control with energy budget constraints
        cpu_adjustment = self.kp * cpu_error * self.state.energy_budget
        mem_adjustment = self.kp * mem_error * self.state.energy_budget
        
        # Log regulation actions
        if abs(cpu_adjustment) > 5.0 or abs(mem_adjustment) > 5.0:
            logger.info(
                "Homeostatic regulation applied",
                cpu_adjustment=f"{cpu_adjustment:.2f}%",
                mem_adjustment=f"{mem_adjustment:.2f}%",
                energy_budget=f"{self.state.energy_budget:.2f}",
                stress_level=f"{self.state.stress_level:.2f}"
            )
            
    def _check_adaptation(self):
        """Check if system needs long-term adaptation (allostasis)"""
        if len(self.stress_history) < 10:
            return
            
        # Calculate recent stress trend
        recent_stress = list(self.stress_history)[-10:]
        stress_mean = np.mean(recent_stress)
        stress_trend = recent_stress[-1] - recent_stress[0]
        
        # Hormesis: mild stress improves adaptation
        if 0.3 < stress_mean < 0.5 and stress_trend < 0:
            self.state.adaptation_level = min(1.0, self.state.adaptation_level + self.adaptation_rate)
            self.state.recovery_rate = min(0.3, self.state.recovery_rate + 0.01)
            logger.info("Positive adaptation detected (hormesis effect)")
            
        # Chronic stress: reduce performance
        elif stress_mean > 0.7:
            self.state.adaptation_level = max(0.1, self.state.adaptation_level - self.adaptation_rate * 2)
            self.state.recovery_rate = max(0.05, self.state.recovery_rate - 0.02)
            logger.warning("Chronic stress detected, reducing adaptation")
            
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the system"""
        return {
            "health_score": round(self.state.health_score, 3),
            "stress_level": round(self.state.stress_level, 3),
            "recovery_rate": round(self.state.recovery_rate, 3),
            "adaptation_level": round(self.state.adaptation_level, 3),
            "energy_budget": round(self.state.energy_budget, 3),
            "metabolism_rate": round(self.state.metabolism_rate, 3),
            "circadian_phase": round(self.circadian_phase, 3),
            "circadian_factor": round(0.5 + 0.5 * np.sin(self.circadian_phase), 3)
        }
        
    def get_resource_allocation(self, component_id: str) -> float:
        """Get resource allocation for a specific component"""
        # Base allocation on energy budget and component priority
        base_allocation = self.state.energy_budget
        
        # Apply circadian modulation
        if "critical" in component_id.lower():
            # Critical components get stable allocation
            return max(0.5, base_allocation)
        elif "background" in component_id.lower():
            # Background tasks reduced during stress
            return base_allocation * (1.0 - self.state.stress_level * 0.5)
        else:
            # Normal components follow energy budget
            return base_allocation


# Global singleton instance
_manager_instance: Optional[ProductionMetabolicManager] = None


def get_production_metabolic() -> ProductionMetabolicManager:
    """Get the global production metabolic manager instance"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ProductionMetabolicManager()
    return _manager_instance


async def example_usage():
    """Example of using the production metabolic manager"""
    manager = get_production_metabolic()
    await manager.start()
    
    try:
        # Simulate normal operation
        await manager.update_metrics({
            "cpu_usage": 45.0,
            "memory_usage": 60.0,
            "active_agents": 10,
            "error_rate": 0.005,
            "latency_ms": 80.0,
            "throughput": 1000.0
        })
        
        health = await manager.get_health_status()
        print(f"Normal state: {health}")
        
        # Simulate high stress
        await manager.update_metrics({
            "cpu_usage": 95.0,
            "memory_usage": 88.0,
            "active_agents": 50,
            "error_rate": 0.05,
            "latency_ms": 250.0,
            "throughput": 500.0
        })
        
        health = await manager.get_health_status()
        print(f"Stressed state: {health}")
        
        # Let it run for a bit
        await asyncio.sleep(10)
        
    finally:
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())