"""
🧠 AURA Executive Controller
============================

Executive control system for multi-agent coordination based on
Global Workspace Theory and consciousness principles.

Features:
- Global workspace for information sharing
- Executive functions (planning, monitoring, control)
- Attention mechanisms
- Multi-agent coordination
- Emergent behavior detection

Extracted from consciousness.py - focusing on executive control.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ==================== Core Types ====================

class ConsciousnessLevel(Enum):
    """Levels of system consciousness"""
    DORMANT = "dormant"          # System starting up
    REACTIVE = "reactive"        # Basic response to stimuli
    ADAPTIVE = "adaptive"        # Learning from experience
    REFLECTIVE = "reflective"    # Self-monitoring
    METACOGNITIVE = "metacognitive"  # Thinking about thinking
    EMERGENT = "emergent"        # Novel behaviors emerging


class AttentionFocus(Enum):
    """What the system is focusing on"""
    PERFORMANCE = "performance"
    ERRORS = "errors"
    LEARNING = "learning"
    COORDINATION = "coordination"
    EXPLORATION = "exploration"
    OPTIMIZATION = "optimization"


@dataclass
class WorkspaceItem:
    """Item in global workspace"""
    item_id: str
    content: Any
    source: str  # Which component
    priority: float
    timestamp: float = field(default_factory=time.time)
    ttl: float = 300.0  # Time to live in seconds
    
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


@dataclass
class ExecutiveState:
    """Current executive state"""
    consciousness_level: ConsciousnessLevel = ConsciousnessLevel.REACTIVE
    attention_focus: AttentionFocus = AttentionFocus.PERFORMANCE
    working_memory_usage: float = 0.0
    cognitive_load: float = 0.0
    decision_confidence: float = 0.5
    awareness_score: float = 0.5


# ==================== Global Workspace ====================

class GlobalWorkspace:
    """
    Global workspace for information sharing between components.
    
    Based on Global Workspace Theory - conscious information is
    globally available to all cognitive processes.
    """
    
    def __init__(self, max_items: int = 100):
        self.max_items = max_items
        self.workspace: Dict[str, WorkspaceItem] = {}
        self.subscribers: Dict[str, List[callable]] = defaultdict(list)
        self.access_log: deque = deque(maxlen=1000)
        
    async def broadcast(self, item: WorkspaceItem):
        """Broadcast information to global workspace"""
        # Add to workspace
        self.workspace[item.item_id] = item
        
        # Limit size
        if len(self.workspace) > self.max_items:
            # Remove oldest expired items
            expired = [
                k for k, v in self.workspace.items()
                if v.is_expired()
            ]
            for k in expired[:len(self.workspace) - self.max_items]:
                del self.workspace[k]
        
        # Notify subscribers
        for subscriber in self.subscribers.get(item.source, []):
            try:
                await subscriber(item)
            except Exception as e:
                logger.error(f"Subscriber error: {e}")
        
        # Log access
        self.access_log.append({
            "item_id": item.item_id,
            "source": item.source,
            "timestamp": item.timestamp
        })
    
    def subscribe(self, source: str, callback: callable):
        """Subscribe to broadcasts from a source"""
        self.subscribers[source].append(callback)
    
    def query(self, filter_func: callable) -> List[WorkspaceItem]:
        """Query workspace with filter function"""
        return [
            item for item in self.workspace.values()
            if not item.is_expired() and filter_func(item)
        ]
    
    def get_attention_items(self, n: int = 5) -> List[WorkspaceItem]:
        """Get top N items by priority"""
        valid_items = [
            item for item in self.workspace.values()
            if not item.is_expired()
        ]
        return sorted(valid_items, key=lambda x: x.priority, reverse=True)[:n]


# ==================== Executive Functions ====================

class ExecutiveFunctions:
    """
    Executive functions for high-level control.
    
    Implements:
    - Planning
    - Monitoring  
    - Inhibition
    - Task switching
    - Working memory
    """
    
    def __init__(self):
        self.plans: Dict[str, Dict[str, Any]] = {}
        self.active_goals: List[Dict[str, Any]] = []
        self.inhibited_actions: Set[str] = set()
        self.task_queue: deque = deque()
        self.working_memory: deque = deque(maxlen=7)  # Miller's law
        
    async def plan(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Create a plan to achieve goal"""
        plan_id = f"plan_{int(time.time() * 1000)}"
        
        # Simple planning - can be enhanced with more sophisticated algorithms
        plan = {
            "plan_id": plan_id,
            "goal": goal,
            "steps": self._decompose_goal(goal),
            "estimated_duration": self._estimate_duration(goal),
            "resources_needed": self._estimate_resources(goal),
            "created_at": time.time()
        }
        
        self.plans[plan_id] = plan
        self.active_goals.append(goal)
        
        logger.info(f"Created plan {plan_id} for goal: {goal.get('name', 'unknown')}")
        return plan
    
    def _decompose_goal(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose goal into steps"""
        # Simple decomposition - can be enhanced
        goal_type = goal.get("type", "generic")
        
        if goal_type == "optimization":
            return [
                {"step": "analyze_current_state", "duration": 10},
                {"step": "identify_bottlenecks", "duration": 20},
                {"step": "generate_solutions", "duration": 30},
                {"step": "evaluate_solutions", "duration": 20},
                {"step": "implement_best", "duration": 40}
            ]
        elif goal_type == "learning":
            return [
                {"step": "collect_examples", "duration": 30},
                {"step": "extract_patterns", "duration": 40},
                {"step": "build_model", "duration": 50},
                {"step": "validate_model", "duration": 30}
            ]
        else:
            return [
                {"step": "analyze", "duration": 20},
                {"step": "plan", "duration": 20},
                {"step": "execute", "duration": 40},
                {"step": "evaluate", "duration": 20}
            ]
    
    def _estimate_duration(self, goal: Dict[str, Any]) -> float:
        """Estimate duration for goal completion"""
        complexity = goal.get("complexity", 0.5)
        base_duration = 60.0  # seconds
        return base_duration * (1 + complexity)
    
    def _estimate_resources(self, goal: Dict[str, Any]) -> Dict[str, float]:
        """Estimate resources needed"""
        return {
            "cpu": goal.get("cpu_intensive", 0.5),
            "memory": goal.get("memory_intensive", 0.5),
            "agents": goal.get("agents_needed", 1)
        }
    
    async def monitor_progress(self, plan_id: str) -> Dict[str, Any]:
        """Monitor plan execution progress"""
        plan = self.plans.get(plan_id)
        if not plan:
            return {"error": "Plan not found"}
        
        # Calculate progress (simplified)
        elapsed = time.time() - plan["created_at"]
        estimated = plan["estimated_duration"]
        progress = min(elapsed / estimated, 1.0) if estimated > 0 else 0
        
        return {
            "plan_id": plan_id,
            "progress": progress,
            "elapsed": elapsed,
            "estimated_remaining": max(0, estimated - elapsed),
            "status": "completed" if progress >= 1.0 else "in_progress"
        }
    
    def inhibit_action(self, action: str, duration: float = 60.0):
        """Inhibit an action temporarily"""
        self.inhibited_actions.add(action)
        
        # Schedule removal
        async def remove_inhibition():
            await asyncio.sleep(duration)
            self.inhibited_actions.discard(action)
        
        asyncio.create_task(remove_inhibition())
        logger.info(f"Inhibited action '{action}' for {duration}s")
    
    def is_inhibited(self, action: str) -> bool:
        """Check if action is inhibited"""
        return action in self.inhibited_actions
    
    def switch_task(self, new_task: Dict[str, Any]):
        """Switch to a new task"""
        # Save current context to working memory
        if self.task_queue:
            current = self.task_queue[0]
            self.working_memory.append(current)
        
        # Add new task to front
        self.task_queue.appendleft(new_task)
        logger.info(f"Switched to task: {new_task.get('name', 'unknown')}")
    
    def update_working_memory(self, item: Any):
        """Update working memory"""
        self.working_memory.append(item)


# ==================== Attention Mechanism ====================

class AttentionMechanism:
    """
    Attention mechanism for focusing system resources.
    
    Implements:
    - Selective attention
    - Divided attention
    - Sustained attention
    - Attention switching
    """
    
    def __init__(self):
        self.attention_weights: Dict[str, float] = defaultdict(float)
        self.focus_history: deque = deque(maxlen=100)
        self.distraction_threshold = 0.3
        
    def focus_on(self, target: str, weight: float = 1.0):
        """Focus attention on target"""
        self.attention_weights[target] = min(weight, 1.0)
        self.focus_history.append({
            "target": target,
            "weight": weight,
            "timestamp": time.time()
        })
        
        # Normalize weights
        total = sum(self.attention_weights.values())
        if total > 1.0:
            for key in self.attention_weights:
                self.attention_weights[key] /= total
    
    def get_attention_distribution(self) -> Dict[str, float]:
        """Get current attention distribution"""
        return dict(self.attention_weights)
    
    def is_distracted(self) -> bool:
        """Check if attention is too divided"""
        if not self.attention_weights:
            return False
        
        max_weight = max(self.attention_weights.values())
        return max_weight < self.distraction_threshold
    
    def filter_by_attention(self, items: List[Any], key_func: callable) -> List[Any]:
        """Filter items by attention weight"""
        filtered = []
        for item in items:
            key = key_func(item)
            if self.attention_weights.get(key, 0) > 0.1:
                filtered.append(item)
        return filtered


# ==================== Executive Controller ====================

class ExecutiveController:
    """
    Main executive controller for AURA system.
    
    Coordinates multi-agent activities using consciousness principles
    and executive functions.
    """
    
    def __init__(self, memory_system=None, orchestrator=None):
        self.memory = memory_system
        self.orchestrator = orchestrator
        
        # Core components
        self.workspace = GlobalWorkspace()
        self.executive = ExecutiveFunctions()
        self.attention = AttentionMechanism()
        
        # State
        self.state = ExecutiveState()
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        
        # Emergent behavior detection
        self.behavior_patterns: deque = deque(maxlen=1000)
        self.emergent_behaviors: List[Dict[str, Any]] = []
        
        logger.info("Executive Controller initialized")
    
    async def initialize(self):
        """Initialize executive controller"""
        # Set initial attention
        self.attention.focus_on("system_health", 0.4)
        self.attention.focus_on("performance", 0.3)
        self.attention.focus_on("learning", 0.3)
        
        # Create initial goals
        await self.executive.plan({
            "name": "maintain_system_health",
            "type": "maintenance",
            "priority": 0.9
        })
        
        self.state.consciousness_level = ConsciousnessLevel.ADAPTIVE
    
    async def process_information(self, info: Dict[str, Any]):
        """Process information through executive control"""
        # Create workspace item
        item = WorkspaceItem(
            item_id=f"info_{int(time.time() * 1000)}",
            content=info,
            source=info.get("source", "unknown"),
            priority=info.get("priority", 0.5)
        )
        
        # Broadcast to workspace
        await self.workspace.broadcast(item)
        
        # Update working memory
        self.executive.update_working_memory(info)
        
        # Check if we need to switch attention
        if info.get("urgent", False):
            self.attention.focus_on(info["source"], 0.8)
            self.state.attention_focus = AttentionFocus.ERRORS
        
        # Detect patterns
        await self._detect_patterns(info)
        
        # Store in memory if available
        if self.memory:
            await self.memory.store({
                "type": "executive_processing",
                "info": info,
                "attention": self.attention.get_attention_distribution(),
                "consciousness_level": self.state.consciousness_level.value
            })
    
    async def coordinate_agents(self, agents: List[str], task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multiple agents on a task"""
        logger.info(f"Coordinating {len(agents)} agents on task: {task.get('name')}")
        
        # Create plan
        plan = await self.executive.plan(task)
        
        # Allocate agents to steps
        allocations = self._allocate_agents(agents, plan["steps"])
        
        # Execute through orchestrator if available
        if self.orchestrator:
            workflow = await self.orchestrator.create_workflow({
                "plan": plan,
                "allocations": allocations,
                "coordination_type": "executive"
            })
            
            result = await self.orchestrator.execute_workflow(workflow)
        else:
            # Simple execution
            result = {"status": "completed", "allocations": allocations}
        
        # Monitor and adapt
        progress = await self.executive.monitor_progress(plan["plan_id"])
        
        # Update consciousness level based on success
        if progress["status"] == "completed":
            self._elevate_consciousness()
        
        return {
            "plan": plan,
            "allocations": allocations,
            "result": result,
            "progress": progress
        }
    
    def _allocate_agents(self, agents: List[str], steps: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Allocate agents to steps"""
        allocations = {}
        
        # Simple round-robin allocation - can be enhanced
        for i, step in enumerate(steps):
            allocated = []
            agents_needed = max(1, len(agents) // len(steps))
            
            for j in range(agents_needed):
                agent_idx = (i * agents_needed + j) % len(agents)
                allocated.append(agents[agent_idx])
            
            allocations[step["step"]] = allocated
        
        return allocations
    
    async def _detect_patterns(self, info: Dict[str, Any]):
        """Detect emergent behavior patterns"""
        # Add to pattern history
        self.behavior_patterns.append({
            "info": info,
            "timestamp": time.time(),
            "attention": self.attention.get_attention_distribution().copy()
        })
        
        # Simple pattern detection - can be enhanced
        if len(self.behavior_patterns) >= 10:
            recent = list(self.behavior_patterns)[-10:]
            
            # Check for repeated patterns
            pattern_types = [p["info"].get("type") for p in recent]
            if len(set(pattern_types)) == 1:
                # Same type repeated - potential emergent behavior
                emergent = {
                    "type": "repetitive_behavior",
                    "pattern": pattern_types[0],
                    "count": 10,
                    "detected_at": time.time()
                }
                
                self.emergent_behaviors.append(emergent)
                logger.info(f"Detected emergent behavior: {emergent}")
                
                # Elevate consciousness
                self._elevate_consciousness()
    
    def _elevate_consciousness(self):
        """Elevate consciousness level"""
        current_idx = list(ConsciousnessLevel).index(self.state.consciousness_level)
        if current_idx < len(ConsciousnessLevel) - 1:
            self.state.consciousness_level = list(ConsciousnessLevel)[current_idx + 1]
            logger.info(f"Consciousness elevated to: {self.state.consciousness_level.value}")
    
    async def reflect(self) -> Dict[str, Any]:
        """Reflect on system state and performance"""
        logger.info("Executive reflection initiated")
        
        # Get attention items
        attention_items = self.workspace.get_attention_items()
        
        # Analyze patterns
        pattern_summary = self._analyze_patterns()
        
        # Calculate cognitive load
        self.state.cognitive_load = len(self.executive.working_memory) / 7.0
        
        # Calculate awareness
        workspace_diversity = len(set(item.source for item in attention_items))
        self.state.awareness_score = min(workspace_diversity / 5.0, 1.0)
        
        reflection = {
            "consciousness_level": self.state.consciousness_level.value,
            "attention_focus": self.state.attention_focus.value,
            "cognitive_load": self.state.cognitive_load,
            "awareness_score": self.state.awareness_score,
            "active_goals": len(self.executive.active_goals),
            "emergent_behaviors": len(self.emergent_behaviors),
            "pattern_summary": pattern_summary,
            "attention_distribution": self.attention.get_attention_distribution(),
            "is_distracted": self.attention.is_distracted()
        }
        
        # Store reflection
        if self.memory:
            await self.memory.store({
                "type": "executive_reflection",
                "reflection": reflection,
                "timestamp": time.time()
            })
        
        return reflection
    
    def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze behavior patterns"""
        if not self.behavior_patterns:
            return {"pattern_count": 0}
        
        # Count pattern types
        pattern_types = defaultdict(int)
        for pattern in self.behavior_patterns:
            ptype = pattern["info"].get("type", "unknown")
            pattern_types[ptype] += 1
        
        # Find most common
        most_common = max(pattern_types.items(), key=lambda x: x[1])
        
        return {
            "pattern_count": len(self.behavior_patterns),
            "unique_patterns": len(pattern_types),
            "most_common": most_common[0],
            "most_common_count": most_common[1]
        }
    
    async def make_decision(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make executive decision between options"""
        logger.info(f"Making decision between {len(options)} options")
        
        # Check if any actions are inhibited
        valid_options = [
            opt for opt in options
            if not self.executive.is_inhibited(opt.get("action", ""))
        ]
        
        if not valid_options:
            return {"decision": None, "reason": "all_options_inhibited"}
        
        # Score options based on current attention
        scored_options = []
        for option in valid_options:
            score = 0.0
            
            # Score by attention alignment
            for key, weight in self.attention.attention_weights.items():
                if key in option.get("benefits", []):
                    score += weight
            
            # Adjust by consciousness level
            if self.state.consciousness_level == ConsciousnessLevel.EMERGENT:
                # Prefer novel options
                if option.get("novel", False):
                    score *= 1.5
            elif self.state.consciousness_level == ConsciousnessLevel.REACTIVE:
                # Prefer safe options
                if option.get("safe", True):
                    score *= 1.2
            
            scored_options.append((option, score))
        
        # Select best option
        best_option = max(scored_options, key=lambda x: x[1])
        
        decision = {
            "decision": best_option[0],
            "score": best_option[1],
            "confidence": self.state.decision_confidence,
            "consciousness_level": self.state.consciousness_level.value,
            "reasoning": "attention_weighted_selection"
        }
        
        # Update confidence based on score difference
        if len(scored_options) > 1:
            scores = [s[1] for s in scored_options]
            score_variance = np.var(scores)
            self.state.decision_confidence = min(1.0 - score_variance, 0.95)
        
        return decision
    
    def get_executive_state(self) -> Dict[str, Any]:
        """Get current executive state"""
        return {
            "consciousness_level": self.state.consciousness_level.value,
            "attention_focus": self.state.attention_focus.value,
            "working_memory_usage": self.state.working_memory_usage,
            "cognitive_load": self.state.cognitive_load,
            "decision_confidence": self.state.decision_confidence,
            "awareness_score": self.state.awareness_score,
            "active_goals": len(self.executive.active_goals),
            "inhibited_actions": list(self.executive.inhibited_actions),
            "workspace_items": len(self.workspace.workspace),
            "emergent_behaviors": len(self.emergent_behaviors)
        }