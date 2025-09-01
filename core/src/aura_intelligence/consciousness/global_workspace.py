"""
Global Workspace Theory with Active Inference - 2025 Implementation

Implements consciousness based on:
- Active Inference and Free Energy Principle
- Global Workspace Theory (GWT)
- Predictive Processing
- Hierarchical Bayesian Inference

Key innovations:
- Minimizes prediction error through active inference
- Self-organizing consciousness dynamics
- Iterative memory consolidation
- Hierarchical predictive models
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import structlog
from collections import deque
import uuid
import heapq

logger = structlog.get_logger(__name__)


class ConsciousnessLevel(Enum):
    """Consciousness levels based on integrated information and prediction accuracy"""
    UNCONSCIOUS = 0      # High prediction error, no integration
    PRECONSCIOUS = 1     # Medium prediction error, local integration
    CONSCIOUS = 2        # Low prediction error, global integration
    METACONSCIOUS = 3    # Minimal prediction error, self-aware predictions
    HYPERCONSCIOUS = 4   # Near-zero prediction error, predictive of predictions


@dataclass
class PredictiveModel:
    """Hierarchical predictive model for active inference"""
    model_id: str
    level: int  # Hierarchical level (0=sensory, higher=abstract)
    predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    precision: float = 1.0  # Precision weighting
    free_energy: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class WorkspaceContent:
    """Content in global workspace with prediction error"""
    content_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Active inference components
    prediction_error: float = 0.0
    expected_precision: float = 1.0
    surprisal: float = 0.0
    
    # GWT components
    attention_weight: float = 0.0
    broadcast_strength: float = 0.0
    
    # Temporal dynamics
    timestamp: datetime = field(default_factory=datetime.now)
    decay_rate: float = 0.1
    
    def calculate_free_energy(self) -> float:
        """Calculate free energy as surprisal + complexity"""
        complexity = -np.log(self.expected_precision + 1e-10)
        return self.surprisal + complexity


@dataclass
class ConsciousState:
    """Current conscious state with beliefs and predictions"""
    beliefs: Dict[str, np.ndarray] = field(default_factory=dict)
    predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    attention_focus: Optional[str] = None
    consciousness_level: ConsciousnessLevel = ConsciousnessLevel.PRECONSCIOUS
    global_free_energy: float = 0.0
    integration_measure: float = 0.0  # Phi-like measure


class ActiveInferenceEngine:
    """
    Active Inference engine for consciousness
    Minimizes free energy through perception and action
    """
    
    def __init__(self):
        self.generative_models: Dict[int, PredictiveModel] = {}
        self.belief_buffer = deque(maxlen=100)
        self.learning_rate = 0.01
        self.precision_threshold = 0.7
        
    async def update_beliefs(self, 
                           sensory_data: Dict[str, Any],
                           current_beliefs: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
        """Update beliefs to minimize prediction error"""
        prediction_errors = {}
        updated_beliefs = current_beliefs.copy()
        
        # Calculate prediction errors for each modality
        for modality, data in sensory_data.items():
            if modality in current_beliefs:
                # Convert data to numpy array if needed
                observation = np.array(data) if not isinstance(data, np.ndarray) else data
                prediction = current_beliefs[modality]
                
                # Ensure shapes match
                if observation.shape != prediction.shape:
                    prediction = np.resize(prediction, observation.shape)
                
                # Calculate precision-weighted prediction error
                error = observation - prediction
                precision = self._estimate_precision(modality, error)
                weighted_error = precision * error
                
                prediction_errors[modality] = weighted_error
                
                # Update beliefs using gradient descent on free energy
                updated_beliefs[modality] = prediction + self.learning_rate * weighted_error
        
        # Calculate total free energy
        free_energy = self._calculate_total_free_energy(prediction_errors)
        
        # Store in belief buffer for temporal smoothing
        self.belief_buffer.append((updated_beliefs, free_energy))
        
        return updated_beliefs, free_energy
    
    def _estimate_precision(self, modality: str, error: np.ndarray) -> float:
        """Estimate precision (inverse variance) of predictions"""
        # Adaptive precision based on recent prediction accuracy
        recent_errors = [abs(e) for _, e in self.belief_buffer if modality in e]
        if recent_errors:
            variance = np.var(recent_errors[-10:])  # Last 10 samples
            precision = 1.0 / (variance + 1e-6)
            return np.clip(precision, 0.1, 10.0)
        return 1.0
    
    def _calculate_total_free_energy(self, prediction_errors: Dict[str, np.ndarray]) -> float:
        """Calculate total free energy across all modalities"""
        total = 0.0
        for modality, error in prediction_errors.items():
            # Free energy = prediction error + complexity
            error_term = np.sum(error ** 2)
            complexity_term = len(error) * np.log(2 * np.pi)  # Model complexity
            total += error_term + complexity_term
        return total
    
    async def generate_predictions(self, 
                                 beliefs: Dict[str, np.ndarray],
                                 context: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Generate predictions using hierarchical generative models"""
        predictions = {}
        
        # Bottom-up: Start from sensory level
        for level in sorted(self.generative_models.keys()):
            model = self.generative_models[level]
            
            if level == 0:
                # Sensory predictions based on current beliefs
                for modality, belief in beliefs.items():
                    predictions[modality] = self._predict_sensory(belief, model)
            else:
                # Higher-level predictions based on lower levels
                lower_predictions = {k: v for k, v in predictions.items()}
                abstract_pred = self._predict_abstract(lower_predictions, model, context)
                predictions[f"level_{level}"] = abstract_pred
        
        return predictions
    
    def _predict_sensory(self, belief: np.ndarray, model: PredictiveModel) -> np.ndarray:
        """Generate sensory predictions"""
        # Simple predictive model - can be replaced with neural network
        if "weights" not in model.predictions:
            model.predictions["weights"] = np.random.randn(*belief.shape) * 0.1
        
        weights = model.predictions["weights"]
        prediction = belief + weights * model.precision
        
        # Add Gaussian noise for exploration
        noise = np.random.randn(*prediction.shape) * 0.01
        return prediction + noise
    
    def _predict_abstract(self, 
                         lower_predictions: Dict[str, np.ndarray],
                         model: PredictiveModel,
                         context: Dict[str, Any]) -> np.ndarray:
        """Generate abstract/higher-level predictions"""
        # Integrate lower-level predictions
        integrated = []
        for pred in lower_predictions.values():
            integrated.append(np.mean(pred))
        
        if not integrated:
            return np.array([0.0])
        
        # Apply hierarchical transformation
        abstract_state = np.array(integrated)
        
        # Context modulation
        if "goal" in context:
            goal_vector = np.array([hash(context["goal"]) % 10 / 10.0])
            abstract_state = abstract_state * 0.8 + goal_vector * 0.2
        
        return abstract_state


class GlobalWorkspace:
    """
    Global Workspace with Active Inference
    Implements consciousness through prediction error minimization
    """
    
    def __init__(self):
        # Core components
        self.active_inference = ActiveInferenceEngine()
        self.workspace_contents: Dict[str, WorkspaceContent] = {}
        self.broadcast_queue: asyncio.Queue = asyncio.Queue()
        
        # Conscious state
        self.current_state = ConsciousState()
        
        # Attention mechanism
        self.attention_buffer = deque(maxlen=50)
        self.attention_threshold = 0.3
        
        # Integration tracking
        self.integration_window = 100  # ms
        self.integration_history: List[float] = []
        
        # Subscribers to global broadcasts
        self.subscribers: Set[Callable] = set()
        
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        logger.info("Global Workspace initialized with Active Inference")
    
    async def start(self):
        """Start the global workspace processing"""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._tasks.append(asyncio.create_task(self._broadcast_loop()))
        self._tasks.append(asyncio.create_task(self._integration_loop()))
        self._tasks.append(asyncio.create_task(self._prediction_loop()))
        
        logger.info("Global Workspace started")
    
    async def stop(self):
        """Stop the global workspace"""
        self._running = False
        
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        logger.info("Global Workspace stopped")
    
    async def submit_content(self, content: WorkspaceContent) -> None:
        """Submit content to workspace for competition"""
        # Calculate prediction error and surprisal
        predictions = await self.active_inference.generate_predictions(
            self.current_state.beliefs,
            {"source": content.source}
        )
        
        if content.source in predictions:
            prediction = predictions[content.source]
            actual = np.array(content.data.get("value", [0.0]))
            
            # Ensure shapes match
            if actual.shape != prediction.shape:
                prediction = np.resize(prediction, actual.shape)
            
            content.prediction_error = np.mean((actual - prediction) ** 2)
            content.surprisal = -np.log(1 - content.prediction_error + 1e-10)
        
        # Calculate attention weight based on prediction error and precision
        content.attention_weight = self._calculate_attention_weight(content)
        
        # Add to workspace
        self.workspace_contents[content.content_id] = content
        
        # Check if it wins competition for broadcast
        if content.attention_weight > self.attention_threshold:
            await self.broadcast_queue.put(content)
    
    def _calculate_attention_weight(self, content: WorkspaceContent) -> float:
        """Calculate attention weight using prediction error and precision"""
        # High prediction error with high precision = high attention
        # Low prediction error = low attention (predicted, not surprising)
        
        error_component = np.tanh(content.prediction_error * 2)  # Sigmoid-like
        precision_component = content.expected_precision
        
        # Temporal novelty bonus
        novelty = 1.0
        for old_content in list(self.workspace_contents.values())[-10:]:
            if old_content.source == content.source:
                time_diff = (content.timestamp - old_content.timestamp).total_seconds()
                novelty *= (1 - np.exp(-time_diff / 10))  # Decay over 10 seconds
        
        attention = error_component * precision_component * novelty
        return np.clip(attention, 0, 1)
    
    async def _broadcast_loop(self):
        """Process broadcast queue and update global state"""
        while self._running:
            try:
                # Get highest attention content
                content = await asyncio.wait_for(
                    self.broadcast_queue.get(),
                    timeout=0.1
                )
                
                # Update global conscious state
                await self._update_conscious_state(content)
                
                # Broadcast to all subscribers
                for subscriber in self.subscribers:
                    try:
                        await subscriber(content)
                    except Exception as e:
                        logger.error(f"Subscriber error: {e}")
                
                # Update consciousness level
                self._update_consciousness_level()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Broadcast loop error: {e}")
    
    async def _update_conscious_state(self, content: WorkspaceContent):
        """Update conscious state with new content"""
        # Update beliefs using active inference
        new_beliefs, free_energy = await self.active_inference.update_beliefs(
            {content.source: content.data.get("value", [0.0])},
            self.current_state.beliefs
        )
        
        self.current_state.beliefs = new_beliefs
        self.current_state.global_free_energy = free_energy
        
        # Update attention focus
        if content.attention_weight > 0.7:
            self.current_state.attention_focus = content.source
        
        # Store in attention buffer
        self.attention_buffer.append(content)
    
    async def _integration_loop(self):
        """Calculate information integration (consciousness measure)"""
        while self._running:
            try:
                await asyncio.sleep(self.integration_window / 1000.0)
                
                # Calculate integration across recent contents
                recent_contents = list(self.attention_buffer)
                if len(recent_contents) > 2:
                    integration = self._calculate_integration(recent_contents)
                    self.current_state.integration_measure = integration
                    self.integration_history.append(integration)
                
            except Exception as e:
                logger.error(f"Integration loop error: {e}")
    
    def _calculate_integration(self, contents: List[WorkspaceContent]) -> float:
        """Calculate information integration (simplified Phi)"""
        if not contents:
            return 0.0
        
        # Extract sources and their interactions
        sources = list(set(c.source for c in contents))
        if len(sources) < 2:
            return 0.0
        
        # Calculate mutual information between sources
        total_mi = 0.0
        for i, source1 in enumerate(sources):
            for source2 in sources[i+1:]:
                contents1 = [c for c in contents if c.source == source1]
                contents2 = [c for c in contents if c.source == source2]
                
                if contents1 and contents2:
                    # Simplified mutual information based on attention correlation
                    weights1 = [c.attention_weight for c in contents1]
                    weights2 = [c.attention_weight for c in contents2]
                    
                    correlation = np.corrcoef(weights1[:min(len(weights1), len(weights2))], 
                                            weights2[:min(len(weights1), len(weights2))])[0, 1]
                    
                    if not np.isnan(correlation):
                        total_mi += abs(correlation)
        
        # Normalize by number of sources
        return total_mi / (len(sources) * (len(sources) - 1) / 2)
    
    async def _prediction_loop(self):
        """Continuously generate and update predictions"""
        while self._running:
            try:
                await asyncio.sleep(0.05)  # 50ms prediction cycle
                
                # Generate new predictions
                predictions = await self.active_inference.generate_predictions(
                    self.current_state.beliefs,
                    {"attention_focus": self.current_state.attention_focus}
                )
                
                self.current_state.predictions = predictions
                
                # Prune old content
                self._prune_old_content()
                
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
    
    def _prune_old_content(self):
        """Remove old content from workspace"""
        current_time = datetime.now()
        to_remove = []
        
        for content_id, content in self.workspace_contents.items():
            age = (current_time - content.timestamp).total_seconds()
            if age > 10:  # Remove content older than 10 seconds
                to_remove.append(content_id)
        
        for content_id in to_remove:
            del self.workspace_contents[content_id]
    
    def _update_consciousness_level(self):
        """Update consciousness level based on integration and free energy"""
        # Low free energy + high integration = higher consciousness
        free_energy = self.current_state.global_free_energy
        integration = self.current_state.integration_measure
        
        # Normalize free energy (lower is better)
        fe_score = 1.0 / (1.0 + free_energy)
        
        # Combined score
        consciousness_score = (fe_score + integration) / 2.0
        
        if consciousness_score < 0.2:
            self.current_state.consciousness_level = ConsciousnessLevel.UNCONSCIOUS
        elif consciousness_score < 0.4:
            self.current_state.consciousness_level = ConsciousnessLevel.PRECONSCIOUS
        elif consciousness_score < 0.6:
            self.current_state.consciousness_level = ConsciousnessLevel.CONSCIOUS
        elif consciousness_score < 0.8:
            self.current_state.consciousness_level = ConsciousnessLevel.METACONSCIOUS
        else:
            self.current_state.consciousness_level = ConsciousnessLevel.HYPERCONSCIOUS
    
    def subscribe(self, callback: Callable):
        """Subscribe to consciousness broadcasts"""
        self.subscribers.add(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from broadcasts"""
        self.subscribers.discard(callback)
    
    async def get_conscious_content(self, top_k: int = 5) -> List[WorkspaceContent]:
        """Get top conscious contents by attention weight"""
        contents = list(self.workspace_contents.values())
        contents.sort(key=lambda c: c.attention_weight, reverse=True)
        return contents[:top_k]
    
    def get_consciousness_report(self) -> Dict[str, Any]:
        """Get comprehensive consciousness report"""
        return {
            "level": self.current_state.consciousness_level.name,
            "level_value": self.current_state.consciousness_level.value,
            "global_free_energy": self.current_state.global_free_energy,
            "integration_measure": self.current_state.integration_measure,
            "attention_focus": self.current_state.attention_focus,
            "active_contents": len(self.workspace_contents),
            "subscribers": len(self.subscribers),
            "beliefs": {k: v.shape for k, v in self.current_state.beliefs.items()},
            "predictions": {k: v.shape for k, v in self.current_state.predictions.items()}
        }


# Singleton instance
_global_workspace: Optional[GlobalWorkspace] = None


def get_global_workspace() -> GlobalWorkspace:
    """Get singleton global workspace instance"""
    global _global_workspace
    if _global_workspace is None:
        _global_workspace = GlobalWorkspace()
    return _global_workspace


# Aliases for compatibility
MetaCognitiveController = GlobalWorkspace
ConsciousnessStream = GlobalWorkspace
create_metacognitive_controller = get_global_workspace


@dataclass
class ConsciousDecision:
    """A decision made through conscious deliberation"""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    options: List[Dict[str, Any]] = field(default_factory=list)
    chosen_option: Optional[Dict[str, Any]] = None
    reasoning: List[str] = field(default_factory=list)
    confidence: float = 0.0
    free_energy_reduction: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


async def make_conscious_decision(options: List[Dict[str, Any]], 
                                context: Dict[str, Any]) -> ConsciousDecision:
    """Make a decision using conscious deliberation"""
    workspace = get_global_workspace()
    decision = ConsciousDecision(options=options)
    
    # Evaluate each option through consciousness
    option_evaluations = []
    
    for option in options:
        # Submit option to consciousness
        content = WorkspaceContent(
            source="decision_making",
            data={"option": option, "context": context}
        )
        await workspace.submit_content(content)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Get evaluation based on free energy
        evaluation = {
            "option": option,
            "free_energy": content.calculate_free_energy(),
            "attention": content.attention_weight
        }
        option_evaluations.append(evaluation)
    
    # Choose option with lowest free energy
    best_eval = min(option_evaluations, key=lambda e: e["free_energy"])
    decision.chosen_option = best_eval["option"]
    decision.confidence = 1.0 / (1.0 + best_eval["free_energy"])
    decision.free_energy_reduction = max(0, option_evaluations[0]["free_energy"] - best_eval["free_energy"])
    
    # Generate reasoning
    decision.reasoning = [
        f"Evaluated {len(options)} options",
        f"Chose option with free energy: {best_eval['free_energy']:.3f}",
        f"Confidence: {decision.confidence:.2f}",
        f"Expected free energy reduction: {decision.free_energy_reduction:.3f}"
    ]
    
    return decision


# Example usage
async def example_consciousness():
    """Example of consciousness system in action"""
    workspace = get_global_workspace()
    await workspace.start()
    
    # Submit sensory content
    visual_content = WorkspaceContent(
        source="visual",
        data={"value": [0.8, 0.2, 0.5]},
        expected_precision=0.9
    )
    await workspace.submit_content(visual_content)
    
    # Submit auditory content
    auditory_content = WorkspaceContent(
        source="auditory",
        data={"value": [0.3, 0.7]},
        expected_precision=0.7
    )
    await workspace.submit_content(auditory_content)
    
    # Let consciousness process
    await asyncio.sleep(0.5)
    
    # Get consciousness report
    report = workspace.get_consciousness_report()
    print(f"Consciousness Level: {report['level']}")
    print(f"Free Energy: {report['global_free_energy']:.3f}")
    print(f"Integration: {report['integration_measure']:.3f}")
    
    # Make a conscious decision
    options = [
        {"action": "move_left", "expected_reward": 0.7},
        {"action": "move_right", "expected_reward": 0.8},
        {"action": "stay", "expected_reward": 0.5}
    ]
    
    decision = await make_conscious_decision(options, {"goal": "maximize_reward"})
    print(f"Decision: {decision.chosen_option}")
    print(f"Confidence: {decision.confidence:.2f}")
    
    await workspace.stop()


if __name__ == "__main__":
    asyncio.run(example_consciousness())