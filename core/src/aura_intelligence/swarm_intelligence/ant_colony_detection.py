"""Production-grade Ant Colony Detection with signature-based pheromones"""
import asyncio, time, hashlib, json, random
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque

class AntColonyDetection:
    """Collective error detection using 209+ components as swarm agents"""
    
    def __init__(self, component_registry=None, coral_system=None, tda_adapter=None,
        max_ants_per_round: int = 32, round_timeout_s: float = 0.25):
        self.registry = component_registry or self._get_registry()
        self.coral = coral_system or self._get_coral()
        self.tda = tda_adapter or self._get_tda_adapter()
        self.max_ants_per_round = max_ants_per_round
        self.round_timeout_s = round_timeout_s
        self.pheromone_decay = 0.97
        self.pheromone_boost = 0.10
        
        # Error pheromones keyed by signature (not component-only)
        self.pheromone_trails = defaultdict(float)
        # Per-component health EMA (0..1, 1=healthy)
        self.comp_health = defaultdict(lambda: 1.0)
        self.alpha = 0.2
        
        # Recent anomalies queue for priority routing
        self.anomaly_queue = deque(maxlen=256)
        self._semaphore = asyncio.Semaphore(64)

    def _get_registry(self):
        try:
            from ..components.real_registry import get_real_registry
            return get_real_registry()
        except: 
            return None

    def _get_coral(self):
        try:
            from ..coral.best_coral import CoRaLSystem
            return CoRaLSystem()
        except: 
            return None

    def _get_tda_adapter(self):
        try:
            from ..tda.unified_engine_2025 import UnifiedTDAEngine
            return UnifiedTDAEngine()
        except: 
            return None

    def _signature_key(self, result: Any) -> str:
        """Build stable error signature from response"""
        try:
            if isinstance(result, dict):
                sig = {
                    "status": result.get("status"),
                    "error": result.get("error_type") or result.get("error"),
                    "tda_anomaly": round(float(result.get("tda_anomaly", 0.0)), 2),
                    "component": result.get("component")
                }
            else:
                sig = {"status": "exception", "type": str(type(result))}
            raw = json.dumps(sig, sort_keys=True)
        except:
            raw = str(result)
        return hashlib.blake2s(raw.encode()).hexdigest()[:16]

    def _is_anomaly(self, result: Any) -> bool:
        if isinstance(result, dict):
            if result.get("status") == "error": return True
        if float(result.get("confidence", 1.0)) < 0.5: return True
        if float(result.get("tda_anomaly", 0.0)) > 0.7: return True
        return False

    def _select_components(self) -> List[str]:
        """Select bounded set of components with priority routing"""
        pass
        if not (self.registry and hasattr(self.registry, "components")):
            return [f"component_{i}" for i in range(self.max_ants_per_round)]

        all_ids = list(self.registry.components.keys())
        priority = set()

        # 1) Recently anomalous components
        for comp_id in list(self.anomaly_queue):
            if comp_id in all_ids:
                priority.add(comp_id)
                if len(priority) >= self.max_ants_per_round // 3: break

        # 2) Low-health components
        low_health = sorted(all_ids, key=lambda cid: self.comp_health[cid])[:self.max_ants_per_round // 3]
        priority.update(low_health)

        # 3) Random fill
        remaining = self.max_ants_per_round - len(priority)
        if remaining > 0:
            random_fill = random.sample(all_ids, min(remaining, len(all_ids)))
            priority.update(random_fill)

        selected = list(priority)[:self.max_ants_per_round]
        random.shuffle(selected)
        return selected

        async def detect_errors(self, test_data: Any) -> Dict[str, Any]:
        """Run bounded swarm detection round"""
        components = self._select_components()
        if not components:
            return {"errors_detected": 0, "error_components": [], "healthy_components": 0}

        async def _probe(comp_id: str):
        try:
            async with self._semaphore:
        ctx = {"mode": "probe", "swarm_check": True}
        if self.registry and hasattr(self.registry, "process_data"):
            return comp_id, await self.registry.process_data(comp_id, test_data, context=ctx)
        await asyncio.sleep(0.001)
        return comp_id, {"processed": True, "component": comp_id, "confidence": 0.9}
        except Exception as e:
        return comp_id, {"status": "error", "error_type": type(e).__name__, "component": comp_id}

        # Execute with timeout
        tasks = [asyncio.create_task(_probe(cid)) for cid in components]
        done, pending = await asyncio.wait(tasks, timeout=self.round_timeout_s)
        
        # Cancel pending tasks
        for p in pending: p.cancel()

        errors, healthy = {}, {}
        for d in done:
        try:
            comp_id, result = d.result()
        except Exception as e:
        comp_id, result = "unknown", {"status": "error", "error_type": type(e).__name__}

        anomaly = self._is_anomaly(result)
        # Update component health EMA
        self.comp_health[comp_id] = self.alpha * (0.0 if anomaly else 1.0) + (1-self.alpha) * self.comp_health[comp_id]

        if anomaly:
            sig = self._signature_key(result)
        self.pheromone_trails[sig] = self.pheromone_trails[sig] * self.pheromone_decay + self.pheromone_boost
        errors[comp_id] = result
        self.anomaly_queue.appendleft(comp_id)
        else:
        healthy[comp_id] = result

        # Global pheromone decay
        for k in list(self.pheromone_trails.keys()):
        self.pheromone_trails[k] *= self.pheromone_decay
        if self.pheromone_trails[k] < 0.01:
            del self.pheromone_trails[k]

        return {
        "errors_detected": len(errors),
        "error_components": list(errors.keys()),
        "healthy_components": len(healthy),
        "detection_rate": len(errors) / len(components) if components else 0.0
        }

    def get_swarm_status(self) -> Dict[str, Any]:
        strong_trails = {k: v for k, v in self.pheromone_trails.items() if v > 0.3}
        avg_health = sum(self.comp_health.values()) / len(self.comp_health) if self.comp_health else 1.0
        
        return {
            "total_ants": len(self.registry.components) if self.registry and hasattr(self.registry, "components") else 209,
            "active_trails": len(self.pheromone_trails),
            "strong_error_trails": len(strong_trails),
            "avg_component_health": round(avg_health, 3),
            "swarm_health": "healthy" if len(strong_trails) < 10 and avg_health > 0.8 else "degraded"
        }