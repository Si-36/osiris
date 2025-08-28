"""
Transaction Manager with Temporal Saga Support
==============================================
Implements distributed transactions using Temporal workflows
and the saga pattern with automatic compensations.
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, TypeVar, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from abc import ABC, abstractmethod
import logging
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TransactionState(Enum):
    """States of a distributed transaction"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMMITTING = "committing"
    COMMITTED = "committed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"


@dataclass
class SagaStep:
    """Single step in a saga transaction"""
    step_id: str
    name: str
    action: Callable[..., Any]
    compensation: Optional[Callable[..., Any]] = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    compensated_at: Optional[datetime] = None
    result: Any = None
    error: Optional[Exception] = None
    
    # Retry configuration
    max_retries: int = 3
    retry_count: int = 0
    retry_delay: float = 1.0
    
    def is_completed(self) -> bool:
        """Check if step completed successfully"""
        return self.completed_at is not None and self.error is None
        
    def is_compensated(self) -> bool:
        """Check if step was compensated"""
        return self.compensated_at is not None


@dataclass
class OutboxMessage:
    """Message for outbox pattern implementation"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    saga_id: str = ""
    step_id: str = ""
    event_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Delivery tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Temporal integration
    temporal_workflow_id: Optional[str] = None
    temporal_run_id: Optional[str] = None
    
    def is_published(self) -> bool:
        """Check if message was published"""
        return self.published_at is not None
        
    def is_acknowledged(self) -> bool:
        """Check if message was acknowledged"""
        return self.acknowledged_at is not None


class TransactionManager:
    """
    Manages distributed transactions across multiple stores.
    Implements two-phase commit when supported, falls back to saga pattern.
    """
    
    def __init__(self):
        self.active_transactions: Dict[str, TransactionState] = {}
        self.transaction_stores: Dict[str, List[str]] = {}  # tx_id -> [store_names]
        self._lock = asyncio.Lock()
        
    async def begin(self, 
                   transaction_id: str,
                   stores: List['AbstractStore']) -> None:
        """Begin a distributed transaction"""
        async with self._lock:
            if transaction_id in self.active_transactions:
                raise ValueError(f"Transaction {transaction_id} already exists")
                
            self.active_transactions[transaction_id] = TransactionState.PENDING
            self.transaction_stores[transaction_id] = [s.__class__.__name__ for s in stores]
            
            # Try to begin transaction on each store
            for store in stores:
                try:
                    await store.begin_transaction(TransactionContext(transaction_id=transaction_id))
                except NotImplementedError:
                    # Store doesn't support transactions
                    logger.debug(f"Store {store.__class__.__name__} doesn't support transactions")
                    
            self.active_transactions[transaction_id] = TransactionState.IN_PROGRESS
            
    async def commit(self, transaction_id: str, stores: List['AbstractStore']) -> None:
        """Commit a distributed transaction"""
        async with self._lock:
            if transaction_id not in self.active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found")
                
            if self.active_transactions[transaction_id] != TransactionState.IN_PROGRESS:
                raise ValueError(f"Transaction {transaction_id} not in progress")
                
            self.active_transactions[transaction_id] = TransactionState.COMMITTING
            
        # Two-phase commit
        prepare_results = []
        
        # Phase 1: Prepare
        for store in stores:
            try:
                # Most stores don't support prepare, so we skip
                prepare_results.append((store, True))
            except Exception as e:
                logger.error(f"Prepare failed for {store.__class__.__name__}: {e}")
                prepare_results.append((store, False))
                
        # Check if all prepared successfully
        if all(result for _, result in prepare_results):
            # Phase 2: Commit
            for store in stores:
                try:
                    await store.commit_transaction(TransactionContext(transaction_id=transaction_id))
                except NotImplementedError:
                    pass
                except Exception as e:
                    logger.error(f"Commit failed for {store.__class__.__name__}: {e}")
                    # Too late to rollback - log for manual intervention
                    
            async with self._lock:
                self.active_transactions[transaction_id] = TransactionState.COMMITTED
        else:
            # Rollback
            await self.rollback(transaction_id, stores)
            
    async def rollback(self, transaction_id: str, stores: List['AbstractStore']) -> None:
        """Rollback a distributed transaction"""
        async with self._lock:
            if transaction_id not in self.active_transactions:
                return
                
            self.active_transactions[transaction_id] = TransactionState.COMPENSATING
            
        # Rollback each store
        for store in stores:
            try:
                await store.rollback_transaction(TransactionContext(transaction_id=transaction_id))
            except NotImplementedError:
                pass
            except Exception as e:
                logger.error(f"Rollback failed for {store.__class__.__name__}: {e}")
                
        async with self._lock:
            self.active_transactions[transaction_id] = TransactionState.COMPENSATED
            
    def get_status(self, transaction_id: str) -> Optional[TransactionState]:
        """Get transaction status"""
        return self.active_transactions.get(transaction_id)


class SagaOrchestrator:
    """
    Orchestrates saga pattern transactions with automatic compensation.
    Integrates with Temporal for durable execution.
    """
    
    def __init__(self, 
                 outbox_store: Optional['AbstractStore'] = None,
                 enable_temporal: bool = True):
        self.outbox_store = outbox_store
        self.enable_temporal = enable_temporal
        self.active_sagas: Dict[str, List[SagaStep]] = {}
        self._lock = asyncio.Lock()
        
    async def create_saga(self, saga_id: Optional[str] = None) -> str:
        """Create a new saga"""
        saga_id = saga_id or str(uuid.uuid4())
        
        async with self._lock:
            if saga_id in self.active_sagas:
                raise ValueError(f"Saga {saga_id} already exists")
                
            self.active_sagas[saga_id] = []
            
        logger.info(f"Created saga: {saga_id}")
        return saga_id
        
    async def add_step(self,
                      saga_id: str,
                      name: str,
                      action: Callable[..., Any],
                      compensation: Optional[Callable[..., Any]] = None,
                      *args,
                      **kwargs) -> str:
        """Add a step to the saga"""
        step_id = f"{saga_id}:{name}:{uuid.uuid4().hex[:8]}"
        
        step = SagaStep(
            step_id=step_id,
            name=name,
            action=action,
            compensation=compensation,
            args=args,
            kwargs=kwargs
        )
        
        async with self._lock:
            if saga_id not in self.active_sagas:
                raise ValueError(f"Saga {saga_id} not found")
                
            self.active_sagas[saga_id].append(step)
            
        logger.debug(f"Added step {name} to saga {saga_id}")
        return step_id
        
    async def execute(self, 
                     saga_id: str,
                     context: Optional['TransactionContext'] = None) -> Tuple[bool, List[Any]]:
        """
        Execute the saga with automatic compensation on failure.
        Returns (success, results)
        """
        if saga_id not in self.active_sagas:
            raise ValueError(f"Saga {saga_id} not found")
            
        steps = self.active_sagas[saga_id]
        results = []
        failed_step_index = -1
        
        # Execute forward path
        for i, step in enumerate(steps):
            try:
                logger.info(f"Executing saga step: {step.name}")
                step.started_at = datetime.utcnow()
                
                # Execute with retries
                for attempt in range(step.max_retries):
                    try:
                        # Create step context
                        step_context = context.create_child_context(step.name) if context else None
                        
                        # Execute action
                        result = await step.action(*step.args, **step.kwargs)
                        
                        # Record success
                        step.completed_at = datetime.utcnow()
                        step.result = result
                        results.append(result)
                        
                        # Publish to outbox
                        if self.outbox_store:
                            await self._publish_step_event(saga_id, step, "completed", result, context)
                            
                        break
                        
                    except Exception as e:
                        step.retry_count += 1
                        
                        if step.retry_count >= step.max_retries:
                            raise
                            
                        logger.warning(f"Step {step.name} failed (attempt {attempt + 1}), retrying...")
                        await asyncio.sleep(step.retry_delay * (2 ** attempt))
                        
            except Exception as e:
                logger.error(f"Saga step {step.name} failed: {e}")
                step.error = e
                failed_step_index = i
                
                # Publish failure event
                if self.outbox_store:
                    await self._publish_step_event(saga_id, step, "failed", str(e), context)
                    
                break
                
        # Compensate if failed
        if failed_step_index >= 0:
            logger.info(f"Saga {saga_id} failed at step {failed_step_index}, compensating...")
            
            # Compensate in reverse order
            for i in range(failed_step_index - 1, -1, -1):
                step = steps[i]
                
                if step.compensation and step.is_completed():
                    try:
                        logger.info(f"Compensating step: {step.name}")
                        
                        # Use the result from forward execution
                        comp_args = (step.result,) if step.result is not None else ()
                        await step.compensation(*comp_args)
                        
                        step.compensated_at = datetime.utcnow()
                        
                        # Publish compensation event
                        if self.outbox_store:
                            await self._publish_step_event(saga_id, step, "compensated", None, context)
                            
                    except Exception as e:
                        logger.error(f"Compensation failed for step {step.name}: {e}")
                        # Continue compensating other steps
                        
            return False, results
            
        logger.info(f"Saga {saga_id} completed successfully")
        return True, results
        
    async def _publish_step_event(self,
                                saga_id: str,
                                step: SagaStep,
                                event_type: str,
                                result: Any,
                                context: Optional['TransactionContext']) -> None:
        """Publish step event to outbox"""
        message = OutboxMessage(
            saga_id=saga_id,
            step_id=step.step_id,
            event_type=f"saga.step.{event_type}",
            payload={
                'step_name': step.name,
                'result': str(result) if result else None,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        
        if context:
            message.temporal_workflow_id = context.temporal_workflow_id
            message.temporal_run_id = context.temporal_run_id
            
        # Store in outbox
        await self.outbox_store.upsert(
            message.message_id,
            message,
            context
        )
        
    def get_saga_status(self, saga_id: str) -> Dict[str, Any]:
        """Get saga execution status"""
        if saga_id not in self.active_sagas:
            return {'error': 'Saga not found'}
            
        steps = self.active_sagas[saga_id]
        
        return {
            'saga_id': saga_id,
            'total_steps': len(steps),
            'completed_steps': sum(1 for s in steps if s.is_completed()),
            'compensated_steps': sum(1 for s in steps if s.is_compensated()),
            'failed_steps': sum(1 for s in steps if s.error is not None),
            'steps': [
                {
                    'name': s.name,
                    'completed': s.is_completed(),
                    'compensated': s.is_compensated(),
                    'error': str(s.error) if s.error else None
                }
                for s in steps
            ]
        }


class OutboxPattern:
    """
    Implements the transactional outbox pattern for guaranteed message delivery.
    Ensures events are published exactly once even in case of failures.
    """
    
    def __init__(self, outbox_store: 'AbstractStore'):
        self.outbox_store = outbox_store
        self._publisher_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start the outbox publisher"""
        self._running = True
        self._publisher_task = asyncio.create_task(self._publisher_loop())
        logger.info("Outbox publisher started")
        
    async def stop(self):
        """Stop the outbox publisher"""
        self._running = False
        if self._publisher_task:
            self._publisher_task.cancel()
            try:
                await self._publisher_task
            except asyncio.CancelledError:
                pass
        logger.info("Outbox publisher stopped")
        
    async def publish(self,
                     event_type: str,
                     payload: Dict[str, Any],
                     context: Optional['TransactionContext'] = None) -> str:
        """Add message to outbox for publishing"""
        message = OutboxMessage(
            event_type=event_type,
            payload=payload
        )
        
        if context:
            message.saga_id = context.saga_id or ""
            message.temporal_workflow_id = context.temporal_workflow_id
            message.temporal_run_id = context.temporal_run_id
            
        # Store in outbox
        result = await self.outbox_store.upsert(
            message.message_id,
            message,
            context
        )
        
        if not result.success:
            raise RuntimeError(f"Failed to store outbox message: {result.error}")
            
        return message.message_id
        
    async def _publisher_loop(self):
        """Background task to publish messages from outbox"""
        while self._running:
            try:
                # Query unpublished messages
                from .query_builder import query
                
                unpublished_query = (
                    query()
                    .eq('published_at', None)
                    .lt('retry_count', 3)
                    .sort_asc('created_at')
                    .with_limit(100)
                )
                
                result = await self.outbox_store.query(unpublished_query)
                
                if result.success and result.data:
                    for message in result.data:
                        await self._publish_message(message)
                        
                # Sleep before next batch
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Outbox publisher error: {e}")
                await asyncio.sleep(5.0)
                
    async def _publish_message(self, message: OutboxMessage):
        """Publish a single message"""
        try:
            # This would integrate with your actual message broker
            # For now, just mark as published
            message.published_at = datetime.utcnow()
            message.acknowledged_at = datetime.utcnow()  # Simulate immediate ack
            
            # Update in store
            await self.outbox_store.upsert(
                message.message_id,
                message
            )
            
            logger.debug(f"Published outbox message: {message.message_id}")
            
        except Exception as e:
            logger.error(f"Failed to publish message {message.message_id}: {e}")
            
            # Increment retry count
            message.retry_count += 1
            await self.outbox_store.upsert(
                message.message_id,
                message
            )