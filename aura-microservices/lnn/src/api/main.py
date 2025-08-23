"""
Liquid Neural Network Service API
Adaptive, continuous-time neural networks for real-time learning
Based on AURA Intelligence research and MIT's latest LNN technology
"""

from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import asyncio
import time
import torch
import numpy as np
import json
from typing import Optional, List, Dict, Any
import structlog
from opentelemetry import trace, metrics
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import make_asgi_app

from ..models.liquid.liquid_neural_network_2025 import (
    LiquidNeuralNetwork, LNNConfig, LNNState, LiquidMode, ODESolver,
    EdgeOptimizedLNN, DistributedLNN, create_lnn_for_mode
)
from ..services.adaptation_service import AdaptationService
from ..services.continuous_learning import ContinuousLearningService
from ..services.model_registry import ModelRegistryService
from ..schemas.requests import (
    InferenceRequest, TrainRequest, AdaptRequest,
    ModelConfigRequest, StreamingInferenceRequest, ConsensusInferenceRequest
)
from ..schemas.responses import (
    InferenceResponse, ModelStatusResponse, AdaptationResponse,
    TrainingResponse, ModelInfoResponse, DynamicsResponse
)
from ..middleware.observability import ObservabilityMiddleware
from ..middleware.security import SecurityMiddleware
from ..middleware.circuit_breaker import CircuitBreakerMiddleware
from ..utils.observability import setup_telemetry

# Initialize structured logging
logger = structlog.get_logger()

# Setup telemetry
setup_telemetry()

# WebSocket connections for streaming
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.model_subscriptions: Dict[str, List[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        # Remove from subscriptions
        for model_id, connections in self.model_subscriptions.items():
            if websocket in connections:
                connections.remove(websocket)
                
    async def broadcast_adaptation(self, model_id: str, adaptation_info: Dict[str, Any]):
        """Broadcast adaptation events to subscribed clients"""
        if model_id in self.model_subscriptions:
            for connection in self.model_subscriptions[model_id]:
                try:
                    await connection.send_json({
                        "type": "adaptation",
                        "model_id": model_id,
                        "data": adaptation_info
                    })
                except:
                    pass

manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management
    Handles initialization and cleanup of LNN models
    """
    logger.info("Starting Liquid Neural Network Service")
    
    # Initialize model registry
    app.state.model_registry = ModelRegistryService()
    
    # Initialize default models
    default_models = {
        "standard": LNNConfig(
            input_size=128,
            hidden_size=256,
            output_size=64,
            mode=LiquidMode.STANDARD
        ),
        "adaptive": LNNConfig(
            input_size=128,
            hidden_size=256,
            output_size=64,
            mode=LiquidMode.ADAPTIVE,
            enable_growth=True,
            max_neurons=512
        ),
        "edge": LNNConfig(
            input_size=128,
            hidden_size=128,
            output_size=32,
            mode=LiquidMode.EDGE,
            quantization_bits=8
        )
    }
    
    for model_id, config in default_models.items():
        model = create_lnn_for_mode(config.mode.value, **config.__dict__)
        app.state.model_registry.register_model(model_id, model, config)
        logger.info(f"Initialized {model_id} LNN model")
    
    # Initialize services
    app.state.adaptation_service = AdaptationService()
    app.state.learning_service = ContinuousLearningService()
    
    # Model states storage
    app.state.model_states = {}
    
    # Start background tasks
    app.state.adaptation_monitor = asyncio.create_task(
        monitor_adaptations(app.state.model_registry, manager)
    )
    
    logger.info("Liquid Neural Network Service ready")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Liquid Neural Network Service")
    app.state.adaptation_monitor.cancel()
    
    # Save model states
    for model_id, state in app.state.model_states.items():
        logger.info(f"Saving state for model {model_id}")


async def monitor_adaptations(registry: ModelRegistryService, conn_manager: ConnectionManager):
    """Monitor models for adaptations and broadcast updates"""
    while True:
        try:
            await asyncio.sleep(5)  # Check every 5 seconds
            
            for model_id, model in registry.get_all_models().items():
                if hasattr(model, 'get_info'):
                    info = model.get_info()
                    
                    # Check for adaptations in adaptive models
                    if info.get('total_adaptations', 0) > 0:
                        await conn_manager.broadcast_adaptation(model_id, info)
                        
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Adaptation monitor error", error=str(e))


# Create FastAPI app
app = FastAPI(
    title="AURA Liquid Neural Network Service",
    description="Adaptive, continuous-time neural networks with real-time learning capabilities",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add middleware
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(SecurityMiddleware)
app.add_middleware(CircuitBreakerMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Model-ID", "X-Adaptation-Count"]
)

# Instrument with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with service information"""
    return {
        "service": "AURA Liquid Neural Network Service",
        "version": "2.0.0",
        "features": [
            "MIT's official ncps library integration",
            "Continuous-time neural dynamics",
            "Self-modifying architecture",
            "Edge-optimized variants",
            "Real-time parameter adaptation",
            "Distributed learning with consensus",
            "ODE-based neural computation",
            "Dynamic neuron allocation"
        ],
        "status": "operational"
    }


@app.get("/api/v1/health", response_model=ModelStatusResponse, tags=["health"])
async def health_check(request: Request):
    """
    Comprehensive health check with model statistics
    """
    registry = request.app.state.model_registry
    
    # Get model statistics
    model_stats = {}
    total_inferences = 0
    total_adaptations = 0
    
    for model_id, model in registry.get_all_models().items():
        info = model.get_info()
        model_stats[model_id] = {
            "parameters": info.get("parameters", 0),
            "inference_count": info.get("inference_count", 0),
            "adaptations": info.get("total_adaptations", 0)
        }
        total_inferences += info.get("inference_count", 0)
        total_adaptations += info.get("total_adaptations", 0)
    
    return ModelStatusResponse(
        status="healthy",
        total_models=len(model_stats),
        model_stats=model_stats,
        total_inferences=total_inferences,
        total_adaptations=total_adaptations,
        uptime_seconds=time.time() - request.app.state.get("start_time", time.time())
    )


@app.post("/api/v1/inference", response_model=InferenceResponse, tags=["inference"])
async def run_inference(
    request: InferenceRequest,
    background_tasks: BackgroundTasks,
    registry: ModelRegistryService = Depends(lambda r: r.app.state.model_registry),
    states: Dict = Depends(lambda r: r.app.state.model_states)
):
    """
    Run inference through a Liquid Neural Network
    
    Features:
    - Continuous-time dynamics
    - Stateful processing
    - Real-time adaptation
    - Multiple model variants
    """
    try:
        # Get model
        model = registry.get_model(request.model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
        
        # Convert input to tensor
        input_tensor = torch.tensor(request.input_data, dtype=torch.float32)
        if request.batch_mode and input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Get or initialize state
        state_key = f"{request.model_id}:{request.session_id or 'default'}"
        hidden_state = states.get(state_key)
        
        # Run inference
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output, new_state, info = model(
                input_tensor,
                hidden_state,
                return_dynamics=request.return_dynamics
            )
        
        # Update state
        states[state_key] = new_state
        
        # Convert output
        output_data = output.numpy().tolist()
        
        # Background task to log inference
        background_tasks.add_task(
            logger.info,
            "Inference completed",
            model_id=request.model_id,
            latency_ms=info.get("latency_ms", 0)
        )
        
        # Prepare response
        response = InferenceResponse(
            output=output_data,
            model_id=request.model_id,
            session_id=request.session_id or "default",
            latency_ms=info.get("latency_ms", 0),
            inference_count=info.get("inference_count", 0),
            adaptations=info.get("adaptations", {})
        )
        
        if request.return_dynamics:
            response.dynamics = info.get("dynamics", {})
        
        return response
        
    except Exception as e:
        logger.error("Inference failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/adapt", response_model=AdaptationResponse, tags=["adaptation"])
async def adapt_model(
    request: AdaptRequest,
    registry: ModelRegistryService = Depends(lambda r: r.app.state.model_registry),
    adaptation_service: AdaptationService = Depends(lambda r: r.app.state.adaptation_service)
):
    """
    Adapt model parameters based on feedback
    
    Features:
    - Real-time parameter adjustment
    - No retraining required
    - Continuous learning
    """
    try:
        model = registry.get_model(request.model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
        
        # Perform adaptation
        adaptation_result = await adaptation_service.adapt_model(
            model,
            request.feedback_signal,
            request.adaptation_strength,
            request.target_metrics
        )
        
        return AdaptationResponse(
            model_id=request.model_id,
            adaptation_type=adaptation_result["type"],
            parameters_changed=adaptation_result["parameters_changed"],
            old_performance=adaptation_result.get("old_performance", {}),
            new_performance=adaptation_result.get("new_performance", {}),
            success=adaptation_result["success"]
        )
        
    except Exception as e:
        logger.error("Adaptation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/train/continuous", response_model=TrainingResponse, tags=["training"])
async def continuous_training(
    request: TrainRequest,
    registry: ModelRegistryService = Depends(lambda r: r.app.state.model_registry),
    learning_service: ContinuousLearningService = Depends(lambda r: r.app.state.learning_service)
):
    """
    Continuous learning without stopping inference
    
    Features:
    - Online learning
    - Experience replay
    - Minimal disruption
    """
    try:
        model = registry.get_model(request.model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
        
        # Prepare training data
        train_data = torch.tensor(request.training_data, dtype=torch.float32)
        train_labels = torch.tensor(request.training_labels, dtype=torch.float32) if request.training_labels else None
        
        # Run continuous training
        training_result = await learning_service.train_online(
            model,
            train_data,
            train_labels,
            learning_rate=request.learning_rate,
            epochs=request.epochs
        )
        
        return TrainingResponse(
            model_id=request.model_id,
            samples_processed=training_result["samples_processed"],
            loss_before=training_result["loss_before"],
            loss_after=training_result["loss_after"],
            training_time_ms=training_result["training_time_ms"],
            adaptations_triggered=training_result.get("adaptations", 0)
        )
        
    except Exception as e:
        logger.error("Training failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models", tags=["models"])
async def list_models(
    registry: ModelRegistryService = Depends(lambda r: r.app.state.model_registry)
):
    """List all available LNN models"""
    models = {}
    
    for model_id, model in registry.get_all_models().items():
        info = model.get_info()
        config = registry.get_config(model_id)
        
        models[model_id] = {
            "implementation": info.get("implementation", "unknown"),
            "mode": config.mode.value if config else "unknown",
            "parameters": info.get("parameters", 0),
            "inference_count": info.get("inference_count", 0),
            "adaptations": info.get("total_adaptations", 0),
            "features": {
                "adaptive": config.enable_growth if config else False,
                "edge_optimized": config.mode == LiquidMode.EDGE if config else False,
                "distributed": config.mode == LiquidMode.DISTRIBUTED if config else False
            }
        }
    
    return {"models": models, "total": len(models)}


@app.post("/api/v1/models/create", tags=["models"])
async def create_model(
    request: ModelConfigRequest,
    registry: ModelRegistryService = Depends(lambda r: r.app.state.model_registry)
):
    """Create a new LNN model with custom configuration"""
    try:
        # Create config
        config = LNNConfig(
            input_size=request.input_size,
            hidden_size=request.hidden_size,
            output_size=request.output_size,
            mode=LiquidMode(request.mode),
            num_layers=request.num_layers,
            enable_growth=request.enable_growth,
            ode_solver=ODESolver(request.ode_solver) if request.ode_solver else ODESolver.DOPRI5
        )
        
        # Create model
        model = create_lnn_for_mode(request.mode, **config.__dict__)
        
        # Register model
        model_id = request.model_id or f"custom_{int(time.time()*1000)}"
        registry.register_model(model_id, model, config)
        
        return {
            "model_id": model_id,
            "status": "created",
            "config": {
                "mode": config.mode.value,
                "parameters": sum(p.numel() for p in model.parameters())
            }
        }
        
    except Exception as e:
        logger.error("Model creation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models/{model_id}/info", response_model=ModelInfoResponse, tags=["models"])
async def get_model_info(
    model_id: str,
    registry: ModelRegistryService = Depends(lambda r: r.app.state.model_registry)
):
    """Get detailed information about a specific model"""
    model = registry.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    info = model.get_info()
    config = registry.get_config(model_id)
    
    return ModelInfoResponse(
        model_id=model_id,
        implementation=info.get("implementation", "unknown"),
        config=info.get("config", {}),
        parameters=info.get("parameters", 0),
        inference_count=info.get("inference_count", 0),
        total_adaptations=info.get("total_adaptations", 0),
        current_neurons=info.get("current_neurons", config.hidden_size if config else 0),
        mode=config.mode.value if config else "unknown",
        features={
            "continuous_time": True,
            "adaptive": config.enable_growth if config else False,
            "edge_optimized": config.mode == LiquidMode.EDGE if config else False,
            "distributed": config.mode == LiquidMode.DISTRIBUTED if config else False,
            "ode_solver": config.ode_solver.value if config else "unknown"
        }
    )


@app.websocket("/ws/inference/{model_id}")
async def websocket_inference(websocket: WebSocket, model_id: str):
    """
    WebSocket endpoint for streaming inference
    
    Enables:
    - Real-time continuous inference
    - Low-latency processing
    - Adaptation notifications
    """
    await manager.connect(websocket)
    
    # Subscribe to model adaptations
    if model_id not in manager.model_subscriptions:
        manager.model_subscriptions[model_id] = []
    manager.model_subscriptions[model_id].append(websocket)
    
    registry = app.state.model_registry
    states = app.state.model_states
    
    try:
        model = registry.get_model(model_id)
        if not model:
            await websocket.close(code=4004, reason="Model not found")
            return
        
        # Initialize state for this connection
        session_id = f"ws_{id(websocket)}"
        state_key = f"{model_id}:{session_id}"
        
        while True:
            # Receive input
            data = await websocket.receive_json()
            
            if data["type"] == "inference":
                # Run inference
                input_tensor = torch.tensor(data["input"], dtype=torch.float32)
                
                with torch.no_grad():
                    output, new_state, info = model(
                        input_tensor,
                        states.get(state_key)
                    )
                
                states[state_key] = new_state
                
                # Send result
                await websocket.send_json({
                    "type": "result",
                    "output": output.numpy().tolist(),
                    "latency_ms": info.get("latency_ms", 0),
                    "adaptations": info.get("adaptations", {})
                })
                
            elif data["type"] == "adapt":
                # Trigger adaptation
                if hasattr(model, 'adapt_parameters'):
                    model.adapt_parameters(data.get("feedback", 0.0))
                    await websocket.send_json({
                        "type": "adaptation_complete",
                        "status": "success"
                    })
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        if model_id in manager.model_subscriptions:
            manager.model_subscriptions[model_id].remove(websocket)
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        await websocket.close(code=4000, reason=str(e))


@app.post("/api/v1/inference/consensus", tags=["distributed"])
async def consensus_inference(
    request: ConsensusInferenceRequest,
    registry: ModelRegistryService = Depends(lambda r: r.app.state.model_registry)
):
    """
    Run inference with consensus across multiple models
    
    For distributed decision making
    """
    try:
        # Get all requested models
        models = []
        for model_id in request.model_ids:
            model = registry.get_model(model_id)
            if model:
                models.append((model_id, model))
        
        if not models:
            raise HTTPException(status_code=404, detail="No valid models found")
        
        # Run inference on all models
        input_tensor = torch.tensor(request.input_data, dtype=torch.float32)
        predictions = {}
        
        for model_id, model in models:
            with torch.no_grad():
                output, _, info = model(input_tensor)
                predictions[model_id] = {
                    "output": output.numpy().tolist(),
                    "confidence": info.get("confidence", 1.0)
                }
        
        # Simple consensus (average)
        # In production, this would use Byzantine consensus service
        all_outputs = [torch.tensor(p["output"]) for p in predictions.values()]
        consensus_output = torch.mean(torch.stack(all_outputs), dim=0)
        
        return {
            "consensus_output": consensus_output.numpy().tolist(),
            "individual_predictions": predictions,
            "participants": len(models),
            "consensus_method": "weighted_average"
        }
        
    except Exception as e:
        logger.error("Consensus inference failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/demo/adaptation", tags=["demo"])
async def demo_adaptation(
    steps: int = 10,
    registry: ModelRegistryService = Depends(lambda r: r.app.state.model_registry)
):
    """
    Demo: Self-modifying LNN in action
    
    Shows how the network adapts its architecture
    """
    try:
        # Use adaptive model
        model = registry.get_model("adaptive")
        if not model:
            raise HTTPException(status_code=404, detail="Adaptive model not found")
        
        results = []
        
        for i in range(steps):
            # Generate input with increasing complexity
            complexity = i / steps
            input_size = 128
            
            # Create pattern with varying complexity
            if complexity < 0.3:
                # Simple pattern
                input_data = torch.sin(torch.linspace(0, 2*np.pi, input_size))
            elif complexity < 0.7:
                # Medium complexity
                input_data = torch.sin(torch.linspace(0, 4*np.pi, input_size)) * \
                           torch.cos(torch.linspace(0, 2*np.pi, input_size))
            else:
                # High complexity
                input_data = torch.randn(input_size) * complexity
            
            # Run inference
            output, state, info = model(input_data.unsqueeze(0))
            
            step_result = {
                "step": i,
                "complexity": complexity,
                "output_mean": float(torch.mean(output)),
                "output_std": float(torch.std(output)),
                "adaptations": info.get("adaptations", []),
                "current_neurons": info.get("current_neurons", 0)
            }
            
            results.append(step_result)
        
        # Get final model info
        final_info = model.get_info()
        
        return {
            "demo": "Self-modifying LNN",
            "steps": results,
            "final_state": {
                "total_adaptations": final_info.get("total_adaptations", 0),
                "final_neurons": final_info.get("current_neurons", 0),
                "parameters": final_info.get("parameters", 0)
            }
        }
        
    except Exception as e:
        logger.error("Demo failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with correlation ID"""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    logger.error(
        "Unhandled exception",
        correlation_id=correlation_id,
        error=str(exc),
        path=request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "correlation_id": correlation_id,
            "message": str(exc) if request.app.debug else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
                }
            },
            "handlers": {
                "default": {
                    "formatter": "json",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )