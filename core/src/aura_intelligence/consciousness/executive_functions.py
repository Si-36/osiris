"""
Executive Functions with Hierarchical Active Inference - 2025 Implementation

Based on neuroscience research and active inference principles:
- Hierarchical goal management with expected free energy
- Working memory as predictive buffer
- Cognitive flexibility through model switching
- Inhibitory control via precision weighting

Key innovations:
- Goals as predictions to minimize free energy
- Planning as trajectory through belief space
- Adaptive resource allocation
- Meta-cognitive monitoring
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import structlog
from collections import deque
import uuid
import heapq

logger = structlog.get_logger(__name__)


class GoalStatus(Enum):
    """Goal execution status"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"


class ExecutiveState(Enum):
    """Executive function state"""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    ADAPTING = "adapting"


@dataclass
class Goal:
    """Goal as prediction to minimize free energy"""
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    
    # Goal as prediction
    desired_state: Dict[str, Any] = field(default_factory=dict)
    expected_free_energy: float = 0.0
    current_free_energy: float = float('inf')
    
    # Hierarchical structure
    parent_goal: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    
    # Priority and resources
    priority: float = 0.5
    required_resources: Dict[str, float] = field(default_factory=dict)
    allocated_resources: Dict[str, float] = field(default_factory=dict)
    
    # Temporal aspects
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    
    # Status tracking
    status: GoalStatus = GoalStatus.PENDING
    progress: float = 0.0
    confidence: float = 0.0
    
    def calculate_urgency(self) -> float:
        """Calculate goal urgency based on deadline and priority"""
        if not self.deadline:
            return self.priority
        
        time_remaining = (self.deadline - datetime.now()).total_seconds()
        if time_remaining <= 0:
            return 1.0
        
        # Urgency increases as deadline approaches
        urgency_from_time = 1.0 / (1.0 + time_remaining / 3600.0)  # Hours
        return (self.priority + urgency_from_time) / 2.0


@dataclass
class Plan:
    """Plan as trajectory through belief space"""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal_id: str = ""
    
    # Trajectory through belief space
    belief_trajectory: List[Dict[str, np.ndarray]] = field(default_factory=list)
    action_sequence: List[Dict[str, Any]] = field(default_factory=list)
    
    # Expected outcomes
    expected_free_energy_trajectory: List[float] = field(default_factory=list)
    expected_duration: timedelta = timedelta(seconds=0)
    
    # Execution tracking
    current_step: int = 0
    actual_free_energy: List[float] = field(default_factory=list)
    
    # Adaptability
    flexibility_score: float = 0.5
    alternative_branches: List["Plan"] = field(default_factory=list)
    
    def get_next_action(self) -> Optional[Dict[str, Any]]:
        """Get next action in sequence"""
        if self.current_step < len(self.action_sequence):
            return self.action_sequence[self.current_step]
        return None
    
    def advance(self) -> bool:
        """Advance to next step"""
        if self.current_step < len(self.action_sequence) - 1:
            self.current_step += 1
            return True
        return False


class WorkingMemory:
    """Working memory as predictive buffer with active inference"""
    
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: Dict[str, Any] = {}
        self.access_history = deque(maxlen=100)
        self.predictive_models: Dict[str, np.ndarray] = {}
        self.precision_weights: Dict[str, float] = {}
        
    def store(self, key: str, value: Any, precision: float = 1.0):
        """Store item with precision weighting"""
        # Capacity limit with precision-weighted removal
        if len(self.items) >= self.capacity and key not in self.items:
            # Remove lowest precision item
            if self.precision_weights:
                min_key = min(self.precision_weights, key=self.precision_weights.get)
                if self.precision_weights.get(min_key, 0) < precision:
                    del self.items[min_key]
                    del self.precision_weights[min_key]
                else:
                    return  # Don't store if precision too low
        
        self.items[key] = value
        self.precision_weights[key] = precision
        self.access_history.append((key, datetime.now()))
        
        # Update predictive model
        self._update_prediction(key, value)
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve with prediction if not found"""
        if key in self.items:
            self.access_history.append((key, datetime.now()))
            return self.items[key]
        
        # Try to predict from similar items
        return self._predict_value(key)
    
    def _update_prediction(self, key: str, value: Any):
        """Update predictive model for key"""
        # Simple predictive model - can be replaced with more sophisticated
        if isinstance(value, (int, float)):
            if key not in self.predictive_models:
                self.predictive_models[key] = np.array([value])
            else:
                # Exponential moving average
                alpha = 0.3
                self.predictive_models[key] = (
                    alpha * np.array([value]) + 
                    (1 - alpha) * self.predictive_models[key]
                )
    
    def _predict_value(self, key: str) -> Optional[Any]:
        """Predict value based on similar keys"""
        # Find similar keys
        similar_keys = [k for k in self.items.keys() if key in k or k in key]
        if similar_keys:
            # Return value of most similar key
            return self.items[similar_keys[0]]
        
        # Check predictive models
        if key in self.predictive_models:
            return float(self.predictive_models[key][0])
        
        return None
    
    def get_state(self) -> Dict[str, Any]:
        """Get working memory state"""
        return {
            "items": len(self.items),
            "capacity": self.capacity,
            "utilization": len(self.items) / self.capacity,
            "avg_precision": np.mean(list(self.precision_weights.values())) if self.precision_weights else 0
        }


class CognitiveFlexibility:
    """Cognitive flexibility through model switching"""
    
    def __init__(self):
        self.models: Dict[str, Callable] = {}
        self.model_performance: Dict[str, float] = {}
        self.current_model: Optional[str] = None
        self.switch_cost = 0.1
        self.adaptation_rate = 0.05
        
    def register_model(self, name: str, model: Callable):
        """Register a cognitive model"""
        self.models[name] = model
        self.model_performance[name] = 0.5  # Initial performance
    
    async def select_model(self, context: Dict[str, Any]) -> str:
        """Select best model for context"""
        if not self.models:
            return ""
        
        # Calculate expected performance for each model
        model_scores = {}
        
        for name, model in self.models.items():
            # Base performance
            base_score = self.model_performance[name]
            
            # Context fit (simplified)
            context_fit = self._calculate_context_fit(name, context)
            
            # Switch cost if changing models
            switch_penalty = self.switch_cost if name != self.current_model else 0
            
            model_scores[name] = base_score * context_fit - switch_penalty
        
        # Select best model
        best_model = max(model_scores, key=model_scores.get)
        
        # Update current model
        if best_model != self.current_model:
            logger.info(f"Switching cognitive model from {self.current_model} to {best_model}")
            self.current_model = best_model
        
        return best_model
    
    def _calculate_context_fit(self, model_name: str, context: Dict[str, Any]) -> float:
        """Calculate how well model fits context"""
        # Simplified context matching
        if "task_type" in context:
            task_type = context["task_type"]
            
            # Some heuristics
            if "planning" in model_name and task_type == "planning":
                return 1.2
            elif "reactive" in model_name and task_type == "immediate":
                return 1.3
            elif "analytical" in model_name and task_type == "analysis":
                return 1.1
        
        return 1.0  # Default fit
    
    def update_performance(self, model_name: str, success: bool):
        """Update model performance based on outcome"""
        if model_name in self.model_performance:
            # Simple performance update
            delta = self.adaptation_rate if success else -self.adaptation_rate
            self.model_performance[model_name] = np.clip(
                self.model_performance[model_name] + delta,
                0.1, 0.9
            )


class InhibitoryControl:
    """Inhibitory control through precision weighting"""
    
    def __init__(self):
        self.inhibition_threshold = 0.3
        self.conflict_history = deque(maxlen=50)
        self.inhibited_actions: Set[str] = set()
        
    async def should_inhibit(self, 
                           action: Dict[str, Any],
                           context: Dict[str, Any]) -> Tuple[bool, str]:
        """Determine if action should be inhibited"""
        
        # Check if action is currently inhibited
        action_id = action.get("id", str(action))
        if action_id in self.inhibited_actions:
            return True, "Action currently inhibited"
        
        # Calculate conflict with goals
        conflict_score = self._calculate_conflict(action, context)
        
        # Check safety constraints
        if self._violates_safety(action, context):
            self.inhibited_actions.add(action_id)
            return True, "Safety constraint violation"
        
        # Check resource conflicts
        if self._has_resource_conflict(action, context):
            return True, "Resource conflict"
        
        # Precision-weighted inhibition
        action_precision = action.get("precision", 1.0)
        if conflict_score > self.inhibition_threshold * action_precision:
            self.conflict_history.append((action_id, conflict_score))
            return True, f"High conflict score: {conflict_score:.2f}"
        
        return False, "No inhibition needed"
    
    def _calculate_conflict(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate conflict between action and current goals"""
        goals = context.get("active_goals", [])
        if not goals:
            return 0.0
        
        conflicts = []
        for goal in goals:
            # Simple conflict detection
            if action.get("opposes", "") == goal.get("target", ""):
                conflicts.append(goal.get("priority", 0.5))
        
        return max(conflicts) if conflicts else 0.0
    
    def _violates_safety(self, action: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if action violates safety constraints"""
        safety_rules = context.get("safety_rules", [])
        
        for rule in safety_rules:
            if rule.get("prohibits") == action.get("type"):
                return True
        
        return False
    
    def _has_resource_conflict(self, action: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check for resource conflicts"""
        required = action.get("resources", {})
        available = context.get("available_resources", {})
        
        for resource, amount in required.items():
            if available.get(resource, 0) < amount:
                return True
        
        return False
    
    def release_inhibition(self, action_id: str):
        """Release inhibition on action"""
        self.inhibited_actions.discard(action_id)


class PlanningSystem:
    """Hierarchical planning with active inference"""
    
    def __init__(self):
        self.planning_horizon = 10
        self.beam_width = 5
        self.models = {
            "means_ends": self._means_ends_planning,
            "hierarchical": self._hierarchical_planning,
            "opportunistic": self._opportunistic_planning
        }
        
    async def create_plan(self,
                         goal: Goal,
                         current_beliefs: Dict[str, np.ndarray],
                         context: Dict[str, Any]) -> Plan:
        """Create plan to achieve goal"""
        
        # Select planning strategy based on goal type
        strategy = self._select_strategy(goal, context)
        planning_func = self.models.get(strategy, self._means_ends_planning)
        
        # Generate plan
        plan = await planning_func(goal, current_beliefs, context)
        
        # Add flexibility through alternative branches
        if strategy != "opportunistic":
            alternatives = await self._generate_alternatives(plan, goal, current_beliefs)
            plan.alternative_branches = alternatives[:2]  # Keep top 2 alternatives
        
        return plan
    
    def _select_strategy(self, goal: Goal, context: Dict[str, Any]) -> str:
        """Select planning strategy based on goal and context"""
        # Use hierarchical for complex goals with subgoals
        if goal.subgoals:
            return "hierarchical"
        
        # Use opportunistic for urgent goals
        if goal.calculate_urgency() > 0.8:
            return "opportunistic"
        
        # Default to means-ends analysis
        return "means_ends"
    
    async def _means_ends_planning(self,
                                  goal: Goal,
                                  current_beliefs: Dict[str, np.ndarray],
                                  context: Dict[str, Any]) -> Plan:
        """Means-ends analysis planning"""
        plan = Plan(goal_id=goal.goal_id)
        
        # Calculate difference between current and desired state
        state_diff = self._calculate_state_difference(current_beliefs, goal.desired_state)
        
        # Generate actions to reduce difference
        actions = []
        belief_trajectory = [current_beliefs.copy()]
        expected_fe = [goal.current_free_energy]
        
        for step in range(self.planning_horizon):
            # Find action that maximally reduces free energy
            action = self._find_best_action(state_diff, context)
            if not action:
                break
            
            actions.append(action)
            
            # Predict belief update
            next_beliefs = self._predict_belief_update(belief_trajectory[-1], action)
            belief_trajectory.append(next_beliefs)
            
            # Update state difference
            state_diff = self._calculate_state_difference(next_beliefs, goal.desired_state)
            
            # Estimate free energy
            fe = self._estimate_free_energy(next_beliefs, goal.desired_state)
            expected_fe.append(fe)
            
            # Stop if goal reached
            if fe < goal.expected_free_energy:
                break
        
        plan.action_sequence = actions
        plan.belief_trajectory = belief_trajectory
        plan.expected_free_energy_trajectory = expected_fe
        plan.expected_duration = timedelta(seconds=len(actions) * 2)  # 2 sec per action
        
        return plan
    
    async def _hierarchical_planning(self,
                                   goal: Goal,
                                   current_beliefs: Dict[str, np.ndarray],
                                   context: Dict[str, Any]) -> Plan:
        """Hierarchical task decomposition"""
        plan = Plan(goal_id=goal.goal_id)
        
        # Decompose into subgoals
        subgoals = self._decompose_goal(goal, context)
        
        # Plan for each subgoal
        subplans = []
        cumulative_beliefs = current_beliefs.copy()
        
        for subgoal in subgoals:
            subplan = await self._means_ends_planning(subgoal, cumulative_beliefs, context)
            subplans.append(subplan)
            
            # Update beliefs based on subplan
            if subplan.belief_trajectory:
                cumulative_beliefs = subplan.belief_trajectory[-1]
        
        # Combine subplans
        for subplan in subplans:
            plan.action_sequence.extend(subplan.action_sequence)
            plan.belief_trajectory.extend(subplan.belief_trajectory)
            plan.expected_free_energy_trajectory.extend(subplan.expected_free_energy_trajectory)
        
        return plan
    
    async def _opportunistic_planning(self,
                                    goal: Goal,
                                    current_beliefs: Dict[str, np.ndarray],
                                    context: Dict[str, Any]) -> Plan:
        """Quick opportunistic planning for urgent goals"""
        plan = Plan(goal_id=goal.goal_id)
        
        # Generate minimal action sequence
        available_actions = context.get("available_actions", [])
        
        # Greedy selection of actions
        selected_actions = []
        temp_beliefs = current_beliefs.copy()
        
        for _ in range(min(3, self.planning_horizon)):  # Max 3 actions for quick plan
            best_action = None
            best_fe_reduction = 0
            
            for action in available_actions:
                predicted_beliefs = self._predict_belief_update(temp_beliefs, action)
                fe_reduction = self._estimate_free_energy_reduction(
                    temp_beliefs, predicted_beliefs, goal.desired_state
                )
                
                if fe_reduction > best_fe_reduction:
                    best_action = action
                    best_fe_reduction = fe_reduction
            
            if best_action:
                selected_actions.append(best_action)
                temp_beliefs = self._predict_belief_update(temp_beliefs, best_action)
            else:
                break
        
        plan.action_sequence = selected_actions
        plan.flexibility_score = 0.8  # High flexibility for adaptation
        
        return plan
    
    def _calculate_state_difference(self,
                                  current: Dict[str, np.ndarray],
                                  desired: Dict[str, Any]) -> Dict[str, float]:
        """Calculate difference between states"""
        diff = {}
        
        for key, desired_value in desired.items():
            if key in current:
                current_value = current[key]
                if isinstance(desired_value, (int, float)):
                    diff[key] = abs(desired_value - np.mean(current_value))
                else:
                    # For non-numeric, use binary difference
                    diff[key] = 0.0 if np.array_equal(current_value, desired_value) else 1.0
            else:
                diff[key] = 1.0  # Maximum difference if not present
        
        return diff
    
    def _find_best_action(self,
                         state_diff: Dict[str, float],
                         context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find action that best reduces state difference"""
        available_actions = context.get("available_actions", [])
        if not available_actions:
            return None
        
        # Score actions by expected difference reduction
        action_scores = []
        
        for action in available_actions:
            effects = action.get("effects", {})
            score = 0.0
            
            for key, reduction in effects.items():
                if key in state_diff:
                    score += state_diff[key] * reduction
            
            action_scores.append((score, action))
        
        # Return highest scoring action
        if action_scores:
            action_scores.sort(key=lambda x: x[0], reverse=True)
            return action_scores[0][1]
        
        return None
    
    def _predict_belief_update(self,
                             beliefs: Dict[str, np.ndarray],
                             action: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Predict belief update after action"""
        updated = beliefs.copy()
        
        effects = action.get("effects", {})
        for key, effect in effects.items():
            if key in updated:
                # Simple linear update
                updated[key] = updated[key] + effect * 0.1
            else:
                # Create new belief
                updated[key] = np.array([effect * 0.1])
        
        return updated
    
    def _estimate_free_energy(self,
                            beliefs: Dict[str, np.ndarray],
                            desired_state: Dict[str, Any]) -> float:
        """Estimate free energy of beliefs relative to desired state"""
        total_fe = 0.0
        
        for key, desired in desired_state.items():
            if key in beliefs:
                belief = beliefs[key]
                if isinstance(desired, (int, float)):
                    # Squared error as free energy proxy
                    error = (desired - np.mean(belief)) ** 2
                    total_fe += error
                else:
                    # Binary difference
                    total_fe += 0.0 if np.array_equal(belief, desired) else 1.0
            else:
                total_fe += 1.0  # Penalty for missing belief
        
        return total_fe
    
    def _estimate_free_energy_reduction(self,
                                      current: Dict[str, np.ndarray],
                                      predicted: Dict[str, np.ndarray],
                                      desired: Dict[str, Any]) -> float:
        """Estimate reduction in free energy"""
        current_fe = self._estimate_free_energy(current, desired)
        predicted_fe = self._estimate_free_energy(predicted, desired)
        return current_fe - predicted_fe
    
    def _decompose_goal(self, goal: Goal, context: Dict[str, Any]) -> List[Goal]:
        """Decompose goal into subgoals"""
        subgoals = []
        
        # Simple decomposition based on goal structure
        if "steps" in goal.desired_state:
            for i, step in enumerate(goal.desired_state["steps"]):
                subgoal = Goal(
                    description=f"Step {i+1} of {goal.description}",
                    desired_state={"step": step},
                    priority=goal.priority,
                    parent_goal=goal.goal_id
                )
                subgoals.append(subgoal)
        else:
            # If no explicit steps, create synthetic subgoals
            num_parts = min(3, len(goal.desired_state))
            for i, (key, value) in enumerate(list(goal.desired_state.items())[:num_parts]):
                subgoal = Goal(
                    description=f"Achieve {key}",
                    desired_state={key: value},
                    priority=goal.priority * (1 - i * 0.1),
                    parent_goal=goal.goal_id
                )
                subgoals.append(subgoal)
        
        return subgoals
    
    async def _generate_alternatives(self,
                                   main_plan: Plan,
                                   goal: Goal,
                                   beliefs: Dict[str, np.ndarray]) -> List[Plan]:
        """Generate alternative plans"""
        alternatives = []
        
        # Try different planning strategies
        for strategy in ["opportunistic", "means_ends"]:
            if strategy != self._select_strategy(goal, {}):
                alt_plan = await self.models[strategy](goal, beliefs, {})
                alternatives.append(alt_plan)
        
        return alternatives


class ExecutiveFunctions:
    """Main executive functions system with active inference"""
    
    def __init__(self):
        # Core components
        self.working_memory = WorkingMemory()
        self.cognitive_flexibility = CognitiveFlexibility()
        self.inhibitory_control = InhibitoryControl()
        self.planning_system = PlanningSystem()
        
        # Goal management
        self.goals: Dict[str, Goal] = {}
        self.active_goals: List[str] = []
        self.goal_queue = asyncio.PriorityQueue()
        
        # Execution state
        self.current_state = ExecutiveState.IDLE
        self.current_plan: Optional[Plan] = None
        self.execution_history = deque(maxlen=100)
        
        # Resource management
        self.total_resources = {"attention": 1.0, "working_memory": 1.0, "processing": 1.0}
        self.allocated_resources: Dict[str, Dict[str, float]] = {}
        
        # Meta-cognitive monitoring
        self.performance_metrics = {
            "goals_completed": 0,
            "goals_failed": 0,
            "avg_completion_time": 0.0,
            "resource_efficiency": 1.0
        }
        
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        logger.info("Executive Functions initialized")
    
    async def start(self):
        """Start executive function system"""
        if self._running:
            return
        
        self._running = True
        self.current_state = ExecutiveState.IDLE
        
        # Start background tasks
        self._tasks.append(asyncio.create_task(self._goal_processing_loop()))
        self._tasks.append(asyncio.create_task(self._execution_loop()))
        self._tasks.append(asyncio.create_task(self._monitoring_loop()))
        
        logger.info("Executive Functions started")
    
    async def stop(self):
        """Stop executive function system"""
        self._running = False
        self.current_state = ExecutiveState.IDLE
        
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        logger.info("Executive Functions stopped")
    
    def add_goal(self, goal: Goal) -> str:
        """Add a new goal"""
        self.goals[goal.goal_id] = goal
        
        # Add to priority queue (negative priority for max heap)
        priority = -goal.calculate_urgency()
        self.goal_queue.put_nowait((priority, goal.goal_id))
        
        logger.info(f"Added goal: {goal.description}")
        return goal.goal_id
    
    async def _goal_processing_loop(self):
        """Process goals from queue"""
        while self._running:
            try:
                # Get highest priority goal
                priority, goal_id = await asyncio.wait_for(
                    self.goal_queue.get(),
                    timeout=0.5
                )
                
                if goal_id not in self.goals:
                    continue
                
                goal = self.goals[goal_id]
                
                # Check if resources available
                if self._can_allocate_resources(goal):
                    # Allocate resources
                    self._allocate_resources(goal)
                    
                    # Add to active goals
                    self.active_goals.append(goal_id)
                    goal.status = GoalStatus.ACTIVE
                    
                    # Start planning
                    self.current_state = ExecutiveState.PLANNING
                    
                    logger.info(f"Activated goal: {goal.description}")
                else:
                    # Put back in queue
                    self.goal_queue.put_nowait((priority, goal_id))
                    await asyncio.sleep(0.1)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Goal processing error: {e}")
    
    def _can_allocate_resources(self, goal: Goal) -> bool:
        """Check if resources available for goal"""
        for resource, required in goal.required_resources.items():
            available = self.total_resources.get(resource, 0)
            allocated = sum(
                alloc.get(resource, 0) 
                for alloc in self.allocated_resources.values()
            )
            
            if available - allocated < required:
                return False
        
        return True
    
    def _allocate_resources(self, goal: Goal):
        """Allocate resources to goal"""
        allocated = {}
        
        for resource, required in goal.required_resources.items():
            allocated[resource] = required
        
        self.allocated_resources[goal.goal_id] = allocated
        goal.allocated_resources = allocated
    
    def _release_resources(self, goal_id: str):
        """Release resources from goal"""
        if goal_id in self.allocated_resources:
            del self.allocated_resources[goal_id]
    
    async def _execution_loop(self):
        """Execute active plans"""
        while self._running:
            try:
                if self.current_state == ExecutiveState.PLANNING:
                    # Plan for active goals
                    for goal_id in self.active_goals:
                        if goal_id in self.goals:
                            goal = self.goals[goal_id]
                            
                            # Get current beliefs from working memory
                            beliefs = self._get_current_beliefs()
                            
                            # Create plan
                            plan = await self.planning_system.create_plan(
                                goal, beliefs, self._get_context()
                            )
                            
                            self.current_plan = plan
                            self.current_state = ExecutiveState.EXECUTING
                            
                            logger.info(f"Created plan for goal: {goal.description}")
                            break
                
                elif self.current_state == ExecutiveState.EXECUTING:
                    # Execute current plan
                    if self.current_plan:
                        success = await self._execute_plan_step()
                        
                        if not success:
                            # Plan failed, try adaptation
                            self.current_state = ExecutiveState.ADAPTING
                
                elif self.current_state == ExecutiveState.ADAPTING:
                    # Adapt plan based on current situation
                    if self.current_plan:
                        adapted = await self._adapt_plan()
                        
                        if adapted:
                            self.current_state = ExecutiveState.EXECUTING
                        else:
                            # Cannot adapt, fail goal
                            await self._fail_current_goal()
                            self.current_state = ExecutiveState.IDLE
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Execution loop error: {e}")
    
    def _get_current_beliefs(self) -> Dict[str, np.ndarray]:
        """Get current beliefs from working memory"""
        beliefs = {}
        
        for key, value in self.working_memory.items.items():
            if isinstance(value, (int, float)):
                beliefs[key] = np.array([value])
            elif isinstance(value, list):
                beliefs[key] = np.array(value)
            elif isinstance(value, np.ndarray):
                beliefs[key] = value
        
        return beliefs
    
    def _get_context(self) -> Dict[str, Any]:
        """Get current execution context"""
        return {
            "active_goals": [self.goals[gid] for gid in self.active_goals if gid in self.goals],
            "available_resources": self._get_available_resources(),
            "working_memory": self.working_memory.get_state(),
            "cognitive_state": self.current_state.value,
            "available_actions": self._get_available_actions()
        }
    
    def _get_available_resources(self) -> Dict[str, float]:
        """Calculate available resources"""
        available = {}
        
        for resource, total in self.total_resources.items():
            allocated = sum(
                alloc.get(resource, 0) 
                for alloc in self.allocated_resources.values()
            )
            available[resource] = total - allocated
        
        return available
    
    def _get_available_actions(self) -> List[Dict[str, Any]]:
        """Get currently available actions"""
        # This would be populated based on the system's capabilities
        return [
            {"id": "store_memory", "type": "cognitive", "effects": {"knowledge": 0.1}},
            {"id": "retrieve_memory", "type": "cognitive", "effects": {"recall": 0.2}},
            {"id": "analyze", "type": "cognitive", "effects": {"understanding": 0.3}},
            {"id": "synthesize", "type": "cognitive", "effects": {"integration": 0.2}},
            {"id": "focus_attention", "type": "executive", "effects": {"attention": 0.5}},
            {"id": "switch_task", "type": "executive", "effects": {"flexibility": 0.1}}
        ]
    
    async def _execute_plan_step(self) -> bool:
        """Execute one step of the current plan"""
        if not self.current_plan:
            return False
        
        action = self.current_plan.get_next_action()
        if not action:
            # Plan completed
            await self._complete_current_goal()
            return False
        
        # Check inhibitory control
        should_inhibit, reason = await self.inhibitory_control.should_inhibit(
            action, self._get_context()
        )
        
        if should_inhibit:
            logger.warning(f"Action inhibited: {reason}")
            return False
        
        # Execute action
        try:
            # Simulate action execution
            await asyncio.sleep(0.1)
            
            # Update working memory based on action effects
            for key, value in action.get("effects", {}).items():
                current = self.working_memory.retrieve(key) or 0
                self.working_memory.store(key, current + value)
            
            # Record execution
            self.execution_history.append({
                "action": action,
                "timestamp": datetime.now(),
                "success": True
            })
            
            # Advance plan
            self.current_plan.advance()
            
            return True
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False
    
    async def _adapt_plan(self) -> bool:
        """Adapt current plan to new situation"""
        if not self.current_plan or not self.current_plan.alternative_branches:
            return False
        
        # Try alternative branch
        self.current_plan = self.current_plan.alternative_branches[0]
        logger.info("Switched to alternative plan branch")
        
        return True
    
    async def _complete_current_goal(self):
        """Mark current goal as completed"""
        if self.active_goals and self.current_plan:
            goal_id = self.current_plan.goal_id
            
            if goal_id in self.goals:
                goal = self.goals[goal_id]
                goal.status = GoalStatus.COMPLETED
                goal.progress = 1.0
                
                # Update metrics
                self.performance_metrics["goals_completed"] += 1
                
                # Release resources
                self._release_resources(goal_id)
                
                # Remove from active goals
                self.active_goals.remove(goal_id)
                
                logger.info(f"Completed goal: {goal.description}")
        
        self.current_plan = None
        self.current_state = ExecutiveState.IDLE
    
    async def _fail_current_goal(self):
        """Mark current goal as failed"""
        if self.active_goals and self.current_plan:
            goal_id = self.current_plan.goal_id
            
            if goal_id in self.goals:
                goal = self.goals[goal_id]
                goal.status = GoalStatus.FAILED
                
                # Update metrics
                self.performance_metrics["goals_failed"] += 1
                
                # Release resources
                self._release_resources(goal_id)
                
                # Remove from active goals
                self.active_goals.remove(goal_id)
                
                logger.warning(f"Failed goal: {goal.description}")
        
        self.current_plan = None
        self.current_state = ExecutiveState.IDLE
    
    async def _monitoring_loop(self):
        """Monitor executive function performance"""
        while self._running:
            try:
                await asyncio.sleep(1.0)
                
                # Update resource efficiency
                total_allocated = sum(
                    sum(alloc.values()) 
                    for alloc in self.allocated_resources.values()
                )
                total_available = sum(self.total_resources.values())
                
                self.performance_metrics["resource_efficiency"] = (
                    total_allocated / total_available if total_available > 0 else 0
                )
                
                # Check for stalled goals
                for goal_id in self.active_goals:
                    if goal_id in self.goals:
                        goal = self.goals[goal_id]
                        
                        # Check deadline
                        if goal.deadline and datetime.now() > goal.deadline:
                            logger.warning(f"Goal deadline exceeded: {goal.description}")
                            await self._fail_current_goal()
                
                # Cognitive flexibility adaptation
                if self.cognitive_flexibility.current_model:
                    # Update model performance based on recent executions
                    recent_success = sum(
                        1 for ex in list(self.execution_history)[-10:]
                        if ex.get("success", False)
                    )
                    
                    success_rate = recent_success / 10 if self.execution_history else 0.5
                    self.cognitive_flexibility.update_performance(
                        self.cognitive_flexibility.current_model,
                        success_rate > 0.7
                    )
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def get_executive_state(self) -> Dict[str, Any]:
        """Get comprehensive executive state"""
        return {
            "state": self.current_state.value,
            "active_goals": len(self.active_goals),
            "total_goals": len(self.goals),
            "working_memory": self.working_memory.get_state(),
            "resource_utilization": {
                resource: sum(
                    alloc.get(resource, 0) 
                    for alloc in self.allocated_resources.values()
                ) / total
                for resource, total in self.total_resources.items()
            },
            "performance": self.performance_metrics,
            "current_plan": self.current_plan.plan_id if self.current_plan else None
        }


# Factory functions
def create_executive_function() -> ExecutiveFunctions:
    """Create executive function system"""
    return ExecutiveFunctions()


def create_goal(description: str,
               desired_state: Dict[str, Any],
               priority: float = 0.5,
               deadline: Optional[datetime] = None) -> Goal:
    """Create a new goal"""
    goal = Goal(
        description=description,
        desired_state=desired_state,
        priority=priority,
        deadline=deadline
    )
    
    # Estimate required resources based on complexity
    complexity = len(desired_state)
    goal.required_resources = {
        "attention": min(0.5, complexity * 0.1),
        "working_memory": min(0.7, complexity * 0.15),
        "processing": min(0.6, complexity * 0.12)
    }
    
    return goal


# Aliases for compatibility
ExecutiveFunction = ExecutiveFunctions
GoalManager = ExecutiveFunctions


# Example usage
async def example_executive_functions():
    """Example of executive functions in action"""
    executive = ExecutiveFunctions()
    await executive.start()
    
    # Create a goal
    goal = create_goal(
        description="Learn new concept",
        desired_state={
            "knowledge": 0.8,
            "understanding": 0.7,
            "integration": 0.6
        },
        priority=0.8,
        deadline=datetime.now() + timedelta(minutes=5)
    )
    
    # Add goal
    executive.add_goal(goal)
    
    # Let it process
    await asyncio.sleep(2)
    
    # Check state
    state = executive.get_executive_state()
    print(f"Executive State: {state}")
    
    await executive.stop()


if __name__ == "__main__":
    asyncio.run(example_executive_functions())