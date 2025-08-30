"""
🧠 Executive Function System
2025 State-of-the-Art Cognitive Control for AURA Intelligence

Implements executive functions based on latest neuroscience research:
    pass
- Working Memory with capacity limits and interference
- Cognitive Flexibility for adaptive plan modification  
- Inhibitory Control for filtering inappropriate actions
- Planning System for goal-oriented behavior
- Metacognitive Monitoring for self-awareness

Based on Miyake & Friedman (2012) and Diamond (2013) frameworks.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Set, Tuple, Protocol
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque

from ..core.interfaces import SystemComponent
from ..core.types import Priority, ConfidenceScore, TaskStatus

# Import consciousness components with fallbacks
try:
    from .global_workspace import WorkspaceContent, ConsciousDecision, MetaCognitiveController, ConsciousnessLevel
except ImportError:
    WorkspaceContent = None
    ConsciousDecision = None
    MetaCognitiveController = None
    ConsciousnessLevel = None

try:
    from .attention import AttentionMechanism
except ImportError:
    AttentionMechanism = None


class ExecutiveState(Enum):
    """Executive function states"""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    INHIBITING = "inhibiting"
    SWITCHING = "switching"


class GoalStatus(Enum):
    """Goal execution status"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"


@dataclass
class Goal:
    """Represents a goal in working memory"""
    goal_id: str
    description: str
    priority: Priority
    deadline: Optional[float] = None
    status: GoalStatus = GoalStatus.PENDING
    sub_goals: List['Goal'] = field(default_factory=list)
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Check if goal has expired"""
        pass
        return self.deadline is not None and time.time() > self.deadline


@dataclass
class Plan:
    """Represents an execution plan"""
    plan_id: str
    goal_id: str
    steps: List[Dict[str, Any]]
    current_step: int = 0
    confidence: ConfidenceScore = 0.5
    estimated_duration: float = 0.0
    created_at: float = field(default_factory=time.time)
    
    def is_complete(self) -> bool:
        """Check if plan is complete"""
        pass
        return self.current_step >= len(self.steps)


class WorkingMemory:
    """
    Working Memory with capacity limits and interference effects
    Based on Baddeley's multi-component model (2012)
    """
    
    def __init__(self, capacity: int = 7):  # Miller's magic number ±2
        self.capacity = capacity
        self.active_goals: Dict[str, Goal] = {}
        self.goal_queue: deque = deque(maxlen=capacity * 2)
        self.interference_matrix: np.ndarray = np.eye(capacity)
        
    def add_goal(self, goal: Goal) -> bool:
        """Add goal to working memory with capacity management"""
        if len(self.active_goals) >= self.capacity:
            # Remove lowest priority goal if at capacity
            lowest_priority_goal = min(
                self.active_goals.values(),
                key=lambda g: (g.priority.value, -g.created_at)
            )
            self.remove_goal(lowest_priority_goal.goal_id)
        
        self.active_goals[goal.goal_id] = goal
        self.goal_queue.append(goal.goal_id)
        return True
    
    def remove_goal(self, goal_id: str) -> Optional[Goal]:
        """Remove goal from working memory"""
        return self.active_goals.pop(goal_id, None)
    
    def get_active_goals(self) -> List[Goal]:
        """Get all active goals sorted by priority"""
        pass
        return sorted(
            self.active_goals.values(),
            key=lambda g: (g.priority.value, g.created_at),
            reverse=True
        )
    
    def calculate_interference(self) -> float:
        """Calculate interference between active goals"""
        pass
        if len(self.active_goals) <= 1:
            return 0.0
        
        # Simplified interference calculation
        return min(1.0, (len(self.active_goals) - 1) / self.capacity)


class CognitiveFlexibility:
    """
    Cognitive Flexibility for adaptive plan modification
    Based on task-switching and set-shifting research
    """
    
    def __init__(self):
        self.current_context: Optional[str] = None
        self.context_history: List[Tuple[str, float]] = []
        self.switch_cost: float = 0.1  # Time penalty for context switching
        self.adaptation_rate: float = 0.1
        
        async def switch_context(self, new_context: str) -> float:
                    """Switch to new context and return switch cost"""
        if self.current_context == new_context:
            return 0.0
        
        # Calculate switch cost based on context similarity
        switch_cost = self.switch_cost
        if self.current_context:
            # Add to history
            self.context_history.append((self.current_context, time.time()))
            
            # Reduce switch cost if recently used context
            recent_contexts = [ctx for ctx, t in self.context_history[-5:]]
            if new_context in recent_contexts:
                switch_cost *= 0.5
        
        self.current_context = new_context
        return switch_cost
    
        async def adapt_plan(self, plan: Plan, new_context: Dict[str, Any]) -> Plan:
                    """Adapt plan based on new context"""
        # Simple adaptation: adjust step priorities based on context
        adapted_steps = []
        for step in plan.steps:
            adapted_step = step.copy()
            
            # Adjust step based on context
            if 'urgency' in new_context:
                adapted_step['priority'] = min(
                    1.0, 
                    step.get('priority', 0.5) + new_context['urgency'] * 0.2
                )
            
            adapted_steps.append(adapted_step)
        
        # Create new plan with adapted steps
        adapted_plan = Plan(
            plan_id=f"{plan.plan_id}_adapted_{int(time.time())}",
            goal_id=plan.goal_id,
            steps=adapted_steps,
            confidence=plan.confidence * 0.9,  # Slight confidence reduction
            estimated_duration=plan.estimated_duration * 1.1  # Slight time increase
        )
        
        return adapted_plan


class InhibitoryControl:
    """
    Inhibitory Control for filtering inappropriate actions
    Based on response inhibition and interference control research
    """
    
    def __init__(self):
        self.inhibition_threshold: float = 0.3
        self.inhibited_actions: Set[str] = set()
        self.inhibition_history: List[Tuple[str, float, str]] = []
        
        async def should_inhibit(
        self, 
        action: Dict[str, Any], 
        context: Dict[str, Any]
        ) -> Tuple[bool, str]:
                    """Determine if action should be inhibited"""
        action_type = action.get('type', 'unknown')
        
        # Check explicit inhibition rules
        if action_type in self.inhibited_actions:
            return True, f"Action type '{action_type}' is explicitly inhibited"
        
        # Check context-based inhibition
        risk_score = context.get('risk_score', 0.0)
        if risk_score > self.inhibition_threshold:
            return True, f"High risk score ({risk_score:.3f}) exceeds threshold"
        
        # Check resource constraints
        if context.get('resource_usage', 0.0) > 0.8:
            return True, "High resource usage detected"
        
        # Check temporal constraints
        if action.get('deadline') and time.time() > action['deadline']:
            return True, "Action deadline has passed"
        
        return False, "No inhibition required"
    
    def add_inhibition_rule(self, action_type: str, reason: str) -> None:
        """Add new inhibition rule"""
        self.inhibited_actions.add(action_type)
        self.inhibition_history.append((action_type, time.time(), reason))
    
    def remove_inhibition_rule(self, action_type: str) -> bool:
        """Remove inhibition rule"""
        if action_type in self.inhibited_actions:
            self.inhibited_actions.remove(action_type)
            return True
        return False


class PlanningSystem:
    """
    Goal-oriented planning system
    Based on hierarchical task networks and means-ends analysis
    """
    
    def __init__(self):
        self.planning_strategies: Dict[str, callable] = {
            'hierarchical': self._hierarchical_planning,
            'means_ends': self._means_ends_planning,
            'reactive': self._reactive_planning
        }
        self.default_strategy = 'hierarchical'
        
        async def create_plan(
        self, 
        goal: Goal, 
        context: Dict[str, Any],
        strategy: Optional[str] = None
        ) -> Plan:
                    """Create execution plan for goal"""
        strategy = strategy or self.default_strategy
        planning_func = self.planning_strategies.get(strategy, self._hierarchical_planning)
        
        steps = await planning_func(goal, context)
        
        plan = Plan(
            plan_id=f"plan_{goal.goal_id}_{int(time.time())}",
            goal_id=goal.goal_id,
            steps=steps,
            confidence=self._estimate_plan_confidence(steps, context),
            estimated_duration=self._estimate_duration(steps)
        )
        
        return plan
    
        async def _hierarchical_planning(
        self, 
        goal: Goal, 
        context: Dict[str, Any]
        ) -> List[Dict[str, Any]]:
                    """Hierarchical task decomposition"""
        steps = []
        
        # Decompose goal into sub-goals
        if goal.sub_goals:
            for sub_goal in goal.sub_goals:
                sub_steps = await self._hierarchical_planning(sub_goal, context)
                steps.extend(sub_steps)
        else:
            # Create atomic steps for leaf goals
            steps.append({
                'action': 'execute_goal',
                'goal_id': goal.goal_id,
                'description': goal.description,
                'priority': goal.priority.value,
                'estimated_time': 1.0
            })
        
        return steps
    
        async def _means_ends_planning(
        self, 
        goal: Goal, 
        context: Dict[str, Any]
        ) -> List[Dict[str, Any]]:
                    """Means-ends analysis planning"""
        steps = []
        current_state = context.get('current_state', {})
        target_state = context.get('target_state', {})
        
        # Simple means-ends analysis
        differences = self._find_differences(current_state, target_state)
        
        for diff in differences:
            steps.append({
                'action': 'reduce_difference',
                'difference': diff,
                'priority': goal.priority.value,
                'estimated_time': 0.5
            })
        
        return steps
    
        async def _reactive_planning(
        self, 
        goal: Goal, 
        context: Dict[str, Any]
        ) -> List[Dict[str, Any]]:
                    """Reactive planning based on current context"""
        return [{
            'action': 'reactive_execute',
            'goal_id': goal.goal_id,
            'context_driven': True,
            'priority': goal.priority.value,
            'estimated_time': 0.3
        }]
    
    def _find_differences(
        self, 
        current_state: Dict[str, Any], 
        target_state: Dict[str, Any]
        ) -> List[str]:
                    """Find differences between current and target states"""
        differences = []
        
        for key, target_value in target_state.items():
            current_value = current_state.get(key)
            if current_value != target_value:
                differences.append(f"{key}: {current_value} -> {target_value}")
        
        return differences
    
    def _estimate_plan_confidence(
        self, 
        steps: List[Dict[str, Any]], 
        context: Dict[str, Any]
        ) -> float:
                    """Estimate confidence in plan success"""
        if not steps:
            return 0.0
        
        # Base confidence decreases with plan complexity
        base_confidence = max(0.1, 1.0 - len(steps) * 0.1)
        
        # Adjust based on context
        context_confidence = context.get('confidence_modifier', 1.0)
        
        return min(1.0, base_confidence * context_confidence)
    
    def _estimate_duration(self, steps: List[Dict[str, Any]]) -> float:
        """Estimate total plan duration"""
        return sum(step.get('estimated_time', 1.0) for step in steps)


class ExecutiveFunction(SystemComponent):
    """
    Main Executive Function System
    Coordinates working memory, cognitive flexibility, inhibitory control, and planning
    """
    
    def __init__(self):
        super().__init__()
        self.working_memory = WorkingMemory()
        self.cognitive_flexibility = CognitiveFlexibility()
        self.inhibitory_control = InhibitoryControl()
        self.planning_system = PlanningSystem()
        
        self.current_state = ExecutiveState.IDLE
        self.active_plans: Dict[str, Plan] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.metrics = {
            'goals_completed': 0,
            'plans_created': 0,
            'inhibitions_triggered': 0,
            'context_switches': 0,
            'average_plan_confidence': 0.0,
            'working_memory_load': 0.0
        }
    
        async def start(self) -> None:
                    """Start executive function system"""
        pass
        self.current_state = ExecutiveState.IDLE
        print("🧠 Executive Function System started")
    
        async def stop(self) -> None:
                    """Stop executive function system"""
        pass
        self.current_state = ExecutiveState.IDLE
        print("🛑 Executive Function System stopped")
    
        async def process_goal(
        self, 
        goal: Goal, 
        context: Dict[str, Any] = None
        ) -> ConsciousDecision:
                    """Process a goal through the executive function system"""
        context = context or {}
        
        # Add goal to working memory
        if not self.working_memory.add_goal(goal):
            return ConsciousDecision(
                decision_id=f"exec_reject_{goal.goal_id}",
                options=[{'action': 'reject', 'reason': 'working_memory_full'}],
                chosen={'action': 'reject', 'reason': 'working_memory_full'},
                confidence=1.0,
                reasoning=["Working memory at capacity"]
            )
        
        # Create plan
        self.current_state = ExecutiveState.PLANNING
        plan = await self.planning_system.create_plan(goal, context)
        self.active_plans[plan.plan_id] = plan
        self.metrics['plans_created'] += 1
        
        # Check for inhibition
        should_inhibit, inhibition_reason = await self.inhibitory_control.should_inhibit(
            {'type': 'execute_plan', 'plan_id': plan.plan_id},
            context
        )
        
        if should_inhibit:
            self.metrics['inhibitions_triggered'] += 1
            return ConsciousDecision(
                decision_id=f"exec_inhibit_{goal.goal_id}",
                options=[{'action': 'inhibit', 'reason': inhibition_reason}],
                chosen={'action': 'inhibit', 'reason': inhibition_reason},
                confidence=0.9,
                reasoning=[f"Inhibited: {inhibition_reason}"]
            )
        
        # Execute plan
        self.current_state = ExecutiveState.EXECUTING
        execution_result = await self._execute_plan(plan, context)
        
        # Update metrics
        self._update_metrics()
        
        return ConsciousDecision(
            decision_id=f"exec_complete_{goal.goal_id}",
            options=[execution_result],
            chosen=execution_result,
            confidence=plan.confidence,
            reasoning=[f"Executed plan with {len(plan.steps)} steps"]
        )
    
        async def _execute_plan(
        self, 
        plan: Plan, 
        context: Dict[str, Any]
        ) -> Dict[str, Any]:
                    """Execute a plan step by step"""
        results = []
        
        for i, step in enumerate(plan.steps):
            plan.current_step = i
            
            # Check if context switch is needed
            step_context = step.get('context', 'default')
            switch_cost = await self.cognitive_flexibility.switch_context(step_context)
            
            if switch_cost > 0:
                self.metrics['context_switches'] += 1
                # Simulate switch cost delay
                await asyncio.sleep(switch_cost * 0.1)
            
            # Execute step
            step_result = await self._execute_step(step, context)
            results.append(step_result)
            
            # Check for plan adaptation needs
            if step_result.get('requires_adaptation', False):
                adapted_plan = await self.cognitive_flexibility.adapt_plan(plan, context)
                self.active_plans[adapted_plan.plan_id] = adapted_plan
                plan = adapted_plan
        
        # Mark goal as completed
        goal = self.working_memory.active_goals.get(plan.goal_id)
        if goal:
            goal.status = GoalStatus.COMPLETED
            goal.progress = 1.0
            self.metrics['goals_completed'] += 1
        
        return {
            'action': 'plan_executed',
            'plan_id': plan.plan_id,
            'steps_completed': len(results),
            'results': results,
            'success': True
        }
    
        async def _execute_step(
        self, 
        step: Dict[str, Any], 
        context: Dict[str, Any]
        ) -> Dict[str, Any]:
                    """Execute a single plan step"""
        # Simulate step execution
        execution_time = step.get('estimated_time', 1.0)
        await asyncio.sleep(execution_time * 0.01)  # Scaled down for demo
        
        return {
            'step': step,
            'success': True,
            'execution_time': execution_time,
            'timestamp': time.time()
        }
    
    def _update_metrics(self) -> None:
        """Update executive function metrics"""
        pass
        # Update working memory load
        self.metrics['working_memory_load'] = (
            len(self.working_memory.active_goals) / self.working_memory.capacity
        )
        
        # Update average plan confidence
        if self.active_plans:
            total_confidence = sum(plan.confidence for plan in self.active_plans.values())
            self.metrics['average_plan_confidence'] = total_confidence / len(self.active_plans)
    
    def get_executive_state(self) -> Dict[str, Any]:
        """Get current executive function state"""
        pass
        return {
            'current_state': self.current_state.value,
            'active_goals': len(self.working_memory.active_goals),
            'active_plans': len(self.active_plans),
            'working_memory_load': self.metrics['working_memory_load'],
            'metrics': self.metrics.copy(),
            'interference_level': self.working_memory.calculate_interference()
        }


# Factory functions
    def create_executive_function() -> ExecutiveFunction:
        """Create executive function system"""
        return ExecutiveFunction()


def create_goal(
    goal_id: str,
    description: str,
    priority: Priority = Priority.NORMAL,
    deadline: Optional[float] = None
) -> Goal:
    """Create a new goal"""
    return Goal(
        goal_id=goal_id,
        description=description,
        priority=priority,
        deadline=deadline
    )


def create_executive_functions(
    consciousness_controller: Optional[MetaCognitiveController] = None,
    attention_mechanism: Optional[AttentionMechanism] = None
) -> ExecutiveFunction:
    """Create executive functions system with optional integrations"""
    return ExecutiveFunction(consciousness_controller, attention_mechanism)


async def test_executive_functions():
    """Test the executive functions system"""
    
    print("🧠 Testing Executive Functions System...")
    
    # Create system
    executive = create_executive_functions()
    
    try:
        # Start system
        await executive.start()
        
        # Test goal setting
        print("\n📋 Testing Goal Setting...")
        goal_id = await executive.set_goal(
            description="Implement advanced AI reasoning system",
            level=PlanningLevel.STRATEGIC,
            priority=0.8,
            deadline=datetime.utcnow() + timedelta(hours=24)
        )
        print(f"✅ Created strategic goal: {goal_id}")
        
        # Test decision making
        print("\n🤔 Testing Decision Making...")
        options = [
            {'name': 'Option A', 'score': 0.7, 'resources_needed': {'attention': 0.3}},
            {'name': 'Option B', 'score': 0.8, 'resources_needed': {'attention': 0.5}},
            {'name': 'Option C', 'score': 0.6, 'resources_needed': {'attention': 0.2}}
        ]
        
        decision = await executive.make_decision(
            options=options,
            context={'priority': 0.8, 'urgency': 0.6},
            require_confidence=0.6
        )
        print(f"✅ Decision made: {decision['decision']['name']} (confidence: {decision['confidence']:.2f})")
        
        # Test attention allocation
        print("\n👁️ Testing Attention Allocation...")
        attention_targets = {
            'goal_monitoring': 0.4,
            'decision_making': 0.3,
            'planning': 0.3
        }
        
        allocations = await executive.allocate_attention(attention_targets, duration=30.0)
        print(f"✅ Attention allocated: {allocations}")
        
        # Test progress monitoring
        print("\n📊 Testing Progress Monitoring...")
        progress = await executive.monitor_progress(goal_id)
        print(f"✅ Goal progress: {progress['current_progress']:.1%}")
        
        # Test system state
        print("\n🔍 Testing System State...")
        state = executive.get_executive_state()
        print(f"✅ Cognitive load: {state['executive_state']['cognitive_load']:.2f}")
        print(f"✅ Meta-cognitive level: {state['executive_state']['meta_cognitive_level']:.2f}")
        print(f"✅ Active goals: {len(state['executive_state']['active_goals'])}")
        
        # Wait for monitoring
        print("\n⏱️ Running meta-cognitive monitoring for 10 seconds...")
        await asyncio.sleep(10)
        
        # Final state check
        final_state = executive.get_executive_state()
        print(f"✅ Final cognitive efficiency: {final_state['executive_metrics']['cognitive_efficiency']:.2f}")
        
        print("\n🎉 Executive Functions System test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        await executive.stop()


if __name__ == "__main__":
    asyncio.run(test_executive_functions())