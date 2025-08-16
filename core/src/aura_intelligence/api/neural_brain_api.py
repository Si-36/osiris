"""
Neural Brain API - 2025 State-of-the-Art

API for agentic systems to connect and get analyzed by AURA Intelligence.
Provides real-time topology analysis, health assessment, and counseling.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

from ..tda.unified_engine_2025 import get_unified_tda_engine, AgentSystemHealth


class AgentSystemData(BaseModel):
    """Input data for agentic system analysis."""
    system_id: str = Field(..., description="Unique identifier for the agentic system")
    agents: List[Dict[str, Any]] = Field(..., description="List of agents in the system")
    communications: List[Dict[str, Any]] = Field(..., description="Communication patterns between agents")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="System performance metrics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional system metadata")


class HealthAssessmentResponse(BaseModel):
    """Response containing system health assessment."""
    system_id: str
    topology_score: float
    bottlenecks: List[str]
    recommendations: List[str]
    risk_level: str
    analysis_timestamp: datetime
    persistence_summary: Dict[str, Any]
    causal_relationships: Dict[str, List[str]]


class DashboardData(BaseModel):
    """Dashboard data for monitoring multiple systems."""
    total_systems: int
    total_analyses: int
    avg_analysis_time: float
    system_overview: Dict[str, Any]
    risk_distribution: Dict[str, int]
    common_issues: Dict[str, int]


# Create FastAPI app
app = FastAPI(
    title="AURA Neural Brain API",
    description="2025 State-of-the-Art API for Agentic System Analysis and Counseling",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze", response_model=HealthAssessmentResponse)
async def analyze_agentic_system(system_data: AgentSystemData) -> HealthAssessmentResponse:
    """
    Analyze an agentic system's topology and health.
    
    This is the main endpoint that agentic systems call to get analyzed.
    """
    try:
        # Get the unified TDA engine
        tda_engine = get_unified_tda_engine()
        
        # Perform analysis
        health_assessment = await tda_engine.analyze_agentic_system(system_data.dict())
        
        # Convert to response format
        return HealthAssessmentResponse(
            system_id=health_assessment.system_id,
            topology_score=health_assessment.topology_score,
            bottlenecks=health_assessment.bottlenecks,
            recommendations=health_assessment.recommendations,
            risk_level=health_assessment.risk_level,
            analysis_timestamp=datetime.now(),
            persistence_summary={
                "num_features": len(health_assessment.persistence_diagram),
                "max_persistence": float(health_assessment.persistence_diagram.max()) if len(health_assessment.persistence_diagram) > 0 else 0.0
            },
            causal_relationships=health_assessment.causal_graph
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/recommendations/{system_id}")
async def get_system_recommendations(system_id: str) -> Dict[str, Any]:
    """Get detailed recommendations for a specific system."""
    try:
        tda_engine = get_unified_tda_engine()
        recommendations = await tda_engine.get_system_recommendations(system_id)
        
        if "error" in recommendations:
            raise HTTPException(status_code=404, detail=recommendations["error"])
        
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@app.get("/dashboard", response_model=DashboardData)
async def get_dashboard_data() -> DashboardData:
    """Get dashboard data for monitoring all analyzed systems."""
    try:
        tda_engine = get_unified_tda_engine()
        dashboard_data = await tda_engine.get_dashboard_data()
        
        return DashboardData(
            total_systems=dashboard_data["total_systems_analyzed"],
            total_analyses=dashboard_data["total_analyses"],
            avg_analysis_time=dashboard_data["avg_analysis_time"],
            system_overview=dashboard_data["system_health_overview"],
            risk_distribution=dashboard_data["risk_distribution"],
            common_issues=dashboard_data["common_bottlenecks"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "AURA Neural Brain API", "version": "1.0.0"}


@app.post("/systems/{system_id}/monitor")
async def start_continuous_monitoring(system_id: str, background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Start continuous monitoring of a system."""
    # Add background task for continuous monitoring
    background_tasks.add_task(continuous_monitor, system_id)
    return {"message": f"Started continuous monitoring for system {system_id}"}


async def continuous_monitor(system_id: str):
    """Background task for continuous system monitoring."""
    # This would implement continuous monitoring logic
    # For now, just a placeholder
    await asyncio.sleep(60)  # Monitor every minute


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)