"""
Continuous Learning Service for LNN
Online learning without stopping inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import asyncio
from typing import Optional, Dict, Any, Tuple, List
import time
from collections import deque
import numpy as np
import structlog

logger = structlog.get_logger()


class ExperienceReplayBuffer:
    """Experience replay for continuous learning"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def add(self, experience: Tuple[torch.Tensor, torch.Tensor, float]):
        """Add experience (input, target, priority)"""
        self.buffer.append(experience)
        self.priorities.append(experience[2])
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample batch with priority"""
        if len(self.buffer) < batch_size:
            indices = list(range(len(self.buffer)))
        else:
            # Priority sampling
            priorities = np.array(self.priorities)
            probabilities = priorities / priorities.sum()
            indices = np.random.choice(
                len(self.buffer),
                size=batch_size,
                p=probabilities,
                replace=False
            )
        
        inputs = torch.stack([self.buffer[i][0] for i in indices])
        targets = torch.stack([self.buffer[i][1] for i in indices])
        
        return inputs, targets
    
    def __len__(self):
        return len(self.buffer)


class ContinuousLearningService:
    """
    Service for continuous learning in LNNs
    
    Features:
    - Online learning during inference
    - Experience replay
    - Catastrophic forgetting prevention
    - Performance monitoring
    """
    
    def __init__(self, 
                 replay_capacity: int = 10000,
                 replay_ratio: float = 0.5):
        self.replay_buffer = ExperienceReplayBuffer(replay_capacity)
        self.replay_ratio = replay_ratio
        self.learning_metrics = {
            "total_samples": 0,
            "total_updates": 0,
            "average_loss": 0.0,
            "learning_rate_adjustments": 0
        }
        self.logger = logger.bind(service="continuous_learning")
        
    async def train_online(self,
                          model: nn.Module,
                          new_data: torch.Tensor,
                          new_labels: Optional[torch.Tensor] = None,
                          learning_rate: float = 0.001,
                          epochs: int = 1,
                          batch_size: int = 32) -> Dict[str, Any]:
        """
        Train model online with new data
        
        Args:
            model: LNN model to train
            new_data: New training data
            new_labels: Labels (if supervised)
            learning_rate: Learning rate
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Training results
        """
        start_time = time.perf_counter()
        
        try:
            # Prepare data
            if new_labels is None:
                # Self-supervised: predict next timestep
                new_labels = self._create_self_supervised_labels(new_data)
            
            # Add to replay buffer with high priority
            for i in range(len(new_data)):
                self.replay_buffer.add((
                    new_data[i],
                    new_labels[i],
                    1.0  # High priority for new data
                ))
            
            # Setup optimizer
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=0.01
            )
            
            # Cosine annealing scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs * (len(new_data) // batch_size + 1)
            )
            
            # Loss function
            criterion = nn.MSELoss()
            
            # Track losses
            loss_before = await self._evaluate_loss(model, new_data, new_labels, criterion)
            
            # Training loop
            model.train()
            total_loss = 0
            adaptations = 0
            
            for epoch in range(epochs):
                # Mix new data with replay
                if len(self.replay_buffer) > batch_size:
                    replay_size = int(batch_size * self.replay_ratio)
                    new_size = batch_size - replay_size
                    
                    # Sample from replay buffer
                    replay_inputs, replay_targets = self.replay_buffer.sample(replay_size)
                    
                    # Sample from new data
                    indices = torch.randperm(len(new_data))[:new_size]
                    new_inputs = new_data[indices]
                    new_targets = new_labels[indices]
                    
                    # Combine
                    inputs = torch.cat([new_inputs, replay_inputs])
                    targets = torch.cat([new_targets, replay_targets])
                else:
                    inputs = new_data
                    targets = new_labels
                
                # Forward pass
                outputs, state, info = model(inputs)
                loss = criterion(outputs, targets)
                
                # Check for adaptations
                if "adaptations" in info and info["adaptations"]:
                    adaptations += 1
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            # Evaluate after training
            model.eval()
            loss_after = await self._evaluate_loss(model, new_data, new_labels, criterion)
            
            # Update metrics
            self.learning_metrics["total_samples"] += len(new_data)
            self.learning_metrics["total_updates"] += epochs
            self.learning_metrics["average_loss"] = (
                self.learning_metrics["average_loss"] * 0.9 + total_loss / epochs * 0.1
            )
            
            training_time = (time.perf_counter() - start_time) * 1000
            
            return {
                "samples_processed": len(new_data),
                "loss_before": loss_before,
                "loss_after": loss_after,
                "improvement": (loss_before - loss_after) / loss_before,
                "training_time_ms": training_time,
                "adaptations": adaptations,
                "replay_buffer_size": len(self.replay_buffer),
                "final_lr": scheduler.get_last_lr()[0]
            }
            
        except Exception as e:
            self.logger.error("Online training failed", error=str(e))
            return {
                "samples_processed": 0,
                "loss_before": 0,
                "loss_after": 0,
                "training_time_ms": 0,
                "error": str(e)
            }
    
    async def train_with_validation(self,
                                   model: nn.Module,
                                   train_data: torch.Tensor,
                                   train_labels: torch.Tensor,
                                   val_split: float = 0.2,
                                   **kwargs) -> Dict[str, Any]:
        """Train with validation split"""
        # Split data
        n_samples = len(train_data)
        n_val = int(n_samples * val_split)
        
        indices = torch.randperm(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        val_data = train_data[val_indices]
        val_labels = train_labels[val_indices]
        train_data = train_data[train_indices]
        train_labels = train_labels[train_indices]
        
        # Train
        result = await self.train_online(
            model, train_data, train_labels, **kwargs
        )
        
        # Validate
        val_metrics = await self._validate(model, val_data, val_labels)
        result["validation_metrics"] = val_metrics
        
        return result
    
    async def _evaluate_loss(self,
                           model: nn.Module,
                           data: torch.Tensor,
                           labels: torch.Tensor,
                           criterion: nn.Module) -> float:
        """Evaluate loss on data"""
        model.eval()
        with torch.no_grad():
            outputs, _, _ = model(data)
            loss = criterion(outputs, labels)
        return loss.item()
    
    async def _validate(self,
                       model: nn.Module,
                       val_data: torch.Tensor,
                       val_labels: torch.Tensor) -> Dict[str, float]:
        """Validate model performance"""
        model.eval()
        
        with torch.no_grad():
            outputs, _, info = model(val_data)
            
            # MSE
            mse = nn.functional.mse_loss(outputs, val_labels).item()
            
            # MAE
            mae = nn.functional.l1_loss(outputs, val_labels).item()
            
            # R-squared
            ss_tot = torch.sum((val_labels - val_labels.mean()) ** 2)
            ss_res = torch.sum((val_labels - outputs) ** 2)
            r2 = 1 - ss_res / ss_tot
            
            # Accuracy (for classification-like outputs)
            if outputs.size(1) > 1:
                pred_classes = outputs.argmax(dim=1)
                true_classes = val_labels.argmax(dim=1)
                accuracy = (pred_classes == true_classes).float().mean().item()
            else:
                accuracy = 0.0
        
        return {
            "mse": mse,
            "mae": mae,
            "r2": r2.item(),
            "accuracy": accuracy
        }
    
    def _create_self_supervised_labels(self, data: torch.Tensor) -> torch.Tensor:
        """Create self-supervised labels (predict next timestep)"""
        if data.dim() == 3:  # Sequence data
            return data[:, 1:, :].reshape(-1, data.size(-1))
        else:
            # For non-sequence data, use reconstruction
            return data.clone()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get learning metrics"""
        return {
            **self.learning_metrics,
            "replay_buffer_size": len(self.replay_buffer)
        }
    
    async def adaptive_learning_rate(self,
                                   model: nn.Module,
                                   performance_history: List[float]) -> float:
        """Adaptively adjust learning rate based on performance"""
        if len(performance_history) < 5:
            return 0.001  # Default
        
        # Check if performance is plateauing
        recent = performance_history[-5:]
        variance = np.var(recent)
        
        if variance < 0.0001:  # Plateaued
            self.learning_metrics["learning_rate_adjustments"] += 1
            return 0.0001  # Reduce learning rate
        elif all(recent[i] > recent[i+1] for i in range(4)):  # Degrading
            self.learning_metrics["learning_rate_adjustments"] += 1
            return 0.00001  # Significantly reduce
        else:
            return 0.001  # Keep standard rate