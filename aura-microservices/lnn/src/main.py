"""
AURA-LNN Microservice
FastAPI service for Logical Neural Networks with Byzantine consensus
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import torch
import numpy as np
import asyncio
from datetime import datetime, timezone
import structlog

from .models import LNNConfig, LiquidNeuralNetwork, EdgeLNN, LNNDecision
from .consensus import ByzantineConsensus, Decision, ConsensusLevel

# Configure structured logging
logger = structlog.get_logger()

# Create FastAPI app
app = FastAPI(
    title="AURA-LNN Service",
    description="Logical Neural Networks with Byzantine Consensus for distributed AI reasoning",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
lnn_instances: Dict[str, LiquidNeuralNetwork] = {}
consensus_manager: Optional[ByzantineConsensus] = None


# Pydantic models for API
class LNNCreateRequest(BaseModel):
    model_id: str
    input_size: int = Field(default=128, ge=1, le=4096)
    hidden_size: int = Field(default=256, ge=1, le=4096)
    output_size: int = Field(default=64, ge=1, le=4096)
    tau: float = Field(default=2.0, ge=0.1, le=10.0)
    edge_optimized: bool = False


class LNNInferenceRequest(BaseModel):
    model_id: str
    input_data: List[List[float]]
    use_consensus: bool = True
    consensus_threshold: float = Field(default=0.67, ge=0.5, le=1.0)


class LNNInferenceResponse(BaseModel):
    model_id: str
    output: List[List[float]]
    confidence: float
    inference_time_ms: float
    consensus_achieved: bool
    consensus_details: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the LNN service."""
    global consensus_manager
    
    logger.info("Starting AURA-LNN Service")
    
    # Initialize Byzantine consensus manager
    consensus_manager = ByzantineConsensus(node_id="lnn-primary", byzantine_tolerance=1)
    
    # Create default LNN model
    default_config = LNNConfig(
        input_size=128,
        hidden_size=256,
        output_size=64,
        tau=2.0
    )
    lnn_instances["default"] = LiquidNeuralNetwork(default_config)
    
    logger.info("AURA-LNN Service started successfully")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "aura-lnn",
        "models_loaded": len(lnn_instances),
        "consensus_active": consensus_manager is not None
    }


@app.post("/lnn/create", response_model=Dict[str, Any])
async def create_lnn_model(request: LNNCreateRequest):
    """Create a new LNN model instance."""
    try:
        if request.model_id in lnn_instances:
            raise HTTPException(status_code=400, detail=f"Model {request.model_id} already exists")
        
        config = LNNConfig(
            input_size=request.input_size,
            hidden_size=request.hidden_size,
            output_size=request.output_size,
            tau=request.tau,
            edge_optimized=request.edge_optimized
        )
        
        # Create appropriate model type
        if request.edge_optimized:
            model = EdgeLNN(config)
        else:
            model = LiquidNeuralNetwork(config)
        
        lnn_instances[request.model_id] = model
        
        logger.info(f"Created LNN model", model_id=request.model_id, edge_optimized=request.edge_optimized)
        
        return {
            "model_id": request.model_id,
            "status": "created",
            "config": {
                "input_size": config.input_size,
                "hidden_size": config.hidden_size,
                "output_size": config.output_size,
                "tau": config.tau,
                "edge_optimized": config.edge_optimized
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to create LNN model", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lnn/inference", response_model=LNNInferenceResponse)
async def lnn_inference(request: LNNInferenceRequest):
    """Perform inference with LNN, optionally using Byzantine consensus."""
    try:
        if request.model_id not in lnn_instances:
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
        
        model = lnn_instances[request.model_id]
        
        # Convert input to tensor
        input_tensor = torch.tensor(request.input_data, dtype=torch.float32)
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(1)  # Add sequence dimension
        
        # Perform inference
        start_time = datetime.now(timezone.utc)
        
        with torch.no_grad():
            output, _ = model(input_tensor)
            output_numpy = output.squeeze().numpy()
        
        inference_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        # Calculate confidence based on output distribution
        confidence = float(torch.softmax(output.squeeze(), dim=-1).max().item())
        
        # Create LNN decision
        lnn_decision = LNNDecision(
            model_id=request.model_id,
            output=output.squeeze(),
            confidence=confidence,
            inference_time_ms=inference_time,
            adaptation_count=0,
            timestamp=datetime.now(timezone.utc),
            metadata={"input_shape": list(input_tensor.shape)}
        )
        
        consensus_achieved = False
        consensus_details = None
        
        if request.use_consensus and consensus_manager:
            # Create consensus decision
            decision = Decision(
                id=f"lnn-inference-{datetime.now().timestamp()}",
                type="lnn_inference",
                data={
                    "output": output_numpy.tolist(),
                    "confidence": confidence,
                    "model_id": request.model_id
                },
                proposer="lnn-primary",
                timestamp=datetime.now(timezone.utc)
            )
            
            # Run consensus
            consensus_result = await consensus_manager.propose_decision(decision)
            consensus_achieved = consensus_result.accepted
            
            consensus_details = {
                "accepted": consensus_result.accepted,
                "confidence": consensus_result.confidence,
                "consensus_time_ms": consensus_result.consensus_time_ms,
                "votes": len(consensus_result.votes),
                "threshold": request.consensus_threshold
            }
        
        return LNNInferenceResponse(
            model_id=request.model_id,
            output=output_numpy.tolist(),
            confidence=confidence,
            inference_time_ms=inference_time,
            consensus_achieved=consensus_achieved,
            consensus_details=consensus_details
        )
        
    except Exception as e:
        logger.error(f"Inference failed", error=str(e), model_id=request.model_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/lnn/models")
async def list_models():
    """List all available LNN models."""
    models = []
    for model_id, model in lnn_instances.items():
        models.append({
            "model_id": model_id,
            "input_size": model.config.input_size,
            "hidden_size": model.config.hidden_size,
            "output_size": model.config.output_size,
            "edge_optimized": model.config.edge_optimized
        })
    
    return {"models": models, "count": len(models)}


@app.delete("/lnn/{model_id}")
async def delete_model(model_id: str):
    """Delete an LNN model."""
    if model_id not in lnn_instances:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    if model_id == "default":
        raise HTTPException(status_code=400, detail="Cannot delete default model")
    
    del lnn_instances[model_id]
    logger.info(f"Deleted LNN model", model_id=model_id)
    
    return {"status": "deleted", "model_id": model_id}


@app.get("/consensus/stats")
async def get_consensus_stats():
    """Get Byzantine consensus statistics."""
    if not consensus_manager:
        return {"error": "Consensus not initialized"}
    
    return consensus_manager.get_consensus_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)