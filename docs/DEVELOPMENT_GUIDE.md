# 🛠️ AURA Intelligence Development Guide

## 🚀 **Getting Started**

### **Prerequisites**

#### **System Requirements**
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.9+
- **Memory**: 4GB+ RAM
- **Storage**: 10GB+ available space

#### **Required Services**
```bash
# Redis (required)
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Neo4j (optional, for knowledge graphs)
# Installation instructions in Neo4j section
```

#### **Python Dependencies**
```bash
# Install core dependencies
pip install -r requirements.txt

# Key packages
pip install fastapi uvicorn redis pydantic torch numpy
```

### **Quick Setup**

1. **Clone and Setup**
```bash
git clone <repository-url>
cd aura-intelligence
pip install -r requirements.txt
```

2. **Start Services**
```bash
# Start Redis
sudo systemctl start redis-server

# Verify Redis
redis-cli ping  # Should return PONG
```

3. **Run System**
```bash
# Start the comprehensive system
python3 comprehensive_working_system.py

# API available at http://localhost:8087
```

4. **Test System**
```bash
# Health check
curl http://localhost:8087/health

# System status
curl http://localhost:8087/
```

## 🏗️ **Development Environment**

### **Project Structure**
```
aura-intelligence/
├── core/src/aura_intelligence/     # Core AI components
│   ├── consciousness/              # Consciousness systems
│   ├── memory/                     # Memory and storage
│   ├── lnn/                        # Neural networks
│   ├── agents/                     # Agent systems
│   ├── tda/                        # Topological analysis
│   ├── communication/              # Communication systems
│   ├── orchestration/              # Orchestration systems
│   ├── observability/              # Monitoring and logging
│   ├── resilience/                 # Fault tolerance
│   ├── security/                   # Security systems
│   └── ...                         # 32+ component categories
├── aura_intelligence_api/          # API layer
├── ultimate_api_system/            # Ultimate API integration
├── docs/                           # Documentation
├── archive/                        # Development history
├── tests/                          # Test suites
├── requirements.txt                # Python dependencies
└── README.md                       # Main documentation
```

### **Development Workflow**

#### **1. Component Development**
```python
# Create new component
class NewAIComponent:
    def __init__(self):
        self.ready = True
        self.component_id = f"new-component-{int(time.time())}"
    
    async def process(self, data):
        """Main processing method"""
        # Component logic here
        return {
            "processed": True,
            "data": data,
            "component": self.component_id
        }
    
    def health_check(self):
        """Health check method"""
        return self.ready
```

#### **2. Integration Testing**
```python
def test_new_component():
    component = NewAIComponent()
    
    # Test initialization
    assert component.ready == True
    
    # Test processing
    result = await component.process({"test": "data"})
    assert result["processed"] == True
    
    # Test health check
    assert component.health_check() == True
    
    print("✅ Component tests passed")
```

#### **3. System Integration**
```python
# Add to comprehensive system
def integrate_new_component(system):
    component = NewAIComponent()
    
    system.add_working_component(
        "New AI Component",
        component,
        lambda c: c.health_check()
    )
    
    # Add to data flow if needed
    system.data_flow_stages.append("new_component_processing")
```

## 🧪 **Testing Framework**

### **Unit Testing**
```python
import pytest
import asyncio

class TestAIComponent:
    @pytest.fixture
    def component(self):
        return NewAIComponent()
    
    def test_initialization(self, component):
        assert component.ready == True
        assert component.component_id is not None
    
    @pytest.mark.asyncio
    async def test_processing(self, component):
        data = {"test": "input"}
        result = await component.process(data)
        
        assert result["processed"] == True
        assert result["data"] == data
    
    def test_health_check(self, component):
        assert component.health_check() == True
```

### **Integration Testing**
```python
class TestSystemIntegration:
    @pytest.fixture
    def system(self):
        return ComprehensiveWorkingSystem()
    
    @pytest.mark.asyncio
    async def test_complete_pipeline(self, system):
        request_data = {
            "data": {"values": [1, 2, 3, 4, 5]},
            "query": "test",
            "context": {}
        }
        
        result = await system.process_comprehensive_request(request_data)
        
        assert result["success"] == True
        assert result["stages_completed"] > 0
        assert result["total_processing_time"] < 1.0  # < 1 second
```

### **Performance Testing**
```python
import time
import statistics

def test_performance():
    system = ComprehensiveWorkingSystem()
    processing_times = []
    
    for i in range(100):
        start_time = time.time()
        
        # Process request
        result = await system.process_comprehensive_request({
            "data": {"values": list(range(10))},
            "query": f"test_{i}",
            "context": {}
        })
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
    
    # Performance assertions
    avg_time = statistics.mean(processing_times)
    max_time = max(processing_times)
    
    assert avg_time < 0.1  # Average < 100ms
    assert max_time < 0.5  # Max < 500ms
    
    print(f"Average processing time: {avg_time:.3f}s")
    print(f"Max processing time: {max_time:.3f}s")
```

## 🔧 **Component Development**

### **Component Template**
```python
import time
import asyncio
from typing import Dict, Any, Optional

class ComponentTemplate:
    """Template for new AURA Intelligence components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.component_id = f"{self.__class__.__name__.lower()}-{int(time.time())}"
        self.ready = False
        self.metrics = {
            "requests_processed": 0,
            "total_processing_time": 0.0,
            "errors": 0
        }
        
        # Initialize component
        self._initialize()
    
    def _initialize(self):
        """Initialize component resources"""
        try:
            # Component-specific initialization
            self._setup_resources()
            self.ready = True
        except Exception as e:
            self.ready = False
            raise Exception(f"Component initialization failed: {e}")
    
    def _setup_resources(self):
        """Setup component-specific resources"""
        # Override in subclasses
        pass
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method"""
        if not self.ready:
            raise Exception("Component not ready")
        
        start_time = time.time()
        
        try:
            # Component-specific processing
            result = await self._process_data(data)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics["requests_processed"] += 1
            self.metrics["total_processing_time"] += processing_time
            
            return {
                "success": True,
                "result": result,
                "component_id": self.component_id,
                "processing_time": processing_time
            }
            
        except Exception as e:
            self.metrics["errors"] += 1
            return {
                "success": False,
                "error": str(e),
                "component_id": self.component_id,
                "processing_time": time.time() - start_time
            }
    
    async def _process_data(self, data: Dict[str, Any]) -> Any:
        """Override this method in subclasses"""
        raise NotImplementedError("Subclasses must implement _process_data")
    
    def health_check(self) -> bool:
        """Health check method"""
        return self.ready
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics"""
        return {
            **self.metrics,
            "component_id": self.component_id,
            "ready": self.ready,
            "average_processing_time": (
                self.metrics["total_processing_time"] / 
                max(self.metrics["requests_processed"], 1)
            )
        }
    
    def reset_metrics(self):
        """Reset component metrics"""
        self.metrics = {
            "requests_processed": 0,
            "total_processing_time": 0.0,
            "errors": 0
        }
    
    def shutdown(self):
        """Cleanup component resources"""
        self.ready = False
        # Override in subclasses for specific cleanup
```

### **Example Component Implementation**
```python
class ExampleAIComponent(ComponentTemplate):
    """Example AI component implementation"""
    
    def _setup_resources(self):
        """Setup example-specific resources"""
        self.model = self._load_model()
        self.cache = {}
    
    def _load_model(self):
        """Load AI model (placeholder)"""
        # In real implementation, load actual model
        return {"model": "example_ai_model", "version": "1.0"}
    
    async def _process_data(self, data: Dict[str, Any]) -> Any:
        """Process data through AI model"""
        # Simulate AI processing
        await asyncio.sleep(0.001)  # 1ms processing time
        
        # Example processing logic
        input_values = data.get("values", [])
        processed_values = [x * 2 for x in input_values]  # Simple transformation
        
        return {
            "input_values": input_values,
            "processed_values": processed_values,
            "model_used": self.model["model"],
            "processing_method": "example_ai_processing"
        }
    
    def health_check(self) -> bool:
        """Enhanced health check"""
        if not self.ready:
            return False
        
        # Check model availability
        if not self.model:
            return False
        
        # Check error rate
        if self.metrics["requests_processed"] > 0:
            error_rate = self.metrics["errors"] / self.metrics["requests_processed"]
            if error_rate > 0.1:  # > 10% error rate
                return False
        
        return True
```

## 🔌 **API Development**

### **Adding New Endpoints**
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Create router
component_router = APIRouter(prefix="/component", tags=["component"])

# Request/Response models
class ComponentRequest(BaseModel):
    data: Dict[str, Any]
    options: Optional[Dict[str, Any]] = {}

class ComponentResponse(BaseModel):
    success: bool
    result: Any
    processing_time: float
    component_id: str

# Endpoint implementation
@component_router.post("/process", response_model=ComponentResponse)
async def process_component_data(request: ComponentRequest):
    """Process data through specific component"""
    try:
        component = get_component("example_ai_component")
        result = await component.process(request.data)
        return ComponentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add to main app
app.include_router(component_router)
```

### **API Testing**
```python
from fastapi.testclient import TestClient

def test_component_endpoint():
    client = TestClient(app)
    
    response = client.post("/component/process", json={
        "data": {"values": [1, 2, 3, 4, 5]},
        "options": {"timeout": 30}
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "result" in data
    assert data["processing_time"] > 0
```

## 📊 **Monitoring and Debugging**

### **Logging Setup**
```python
import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Use in components
logger = structlog.get_logger(__name__)

class LoggingComponent(ComponentTemplate):
    async def _process_data(self, data: Dict[str, Any]) -> Any:
        logger.info("Processing started", 
                   component=self.component_id, 
                   data_size=len(str(data)))
        
        try:
            result = await self._actual_processing(data)
            logger.info("Processing completed successfully",
                       component=self.component_id,
                       result_size=len(str(result)))
            return result
        except Exception as e:
            logger.error("Processing failed",
                        component=self.component_id,
                        error=str(e))
            raise
```

### **Performance Monitoring**
```python
import time
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def record_processing_time(self, component_name: str, processing_time: float):
        self.metrics[component_name].append(processing_time)
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for component, times in self.metrics.items():
            summary[component] = {
                "count": len(times),
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "total_time": sum(times)
            }
        return summary

# Global monitor instance
performance_monitor = PerformanceMonitor()

# Use in components
class MonitoredComponent(ComponentTemplate):
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        result = await super().process(data)
        processing_time = time.time() - start_time
        
        performance_monitor.record_processing_time(
            self.__class__.__name__, 
            processing_time
        )
        
        return result
```

## 🚀 **Deployment**

### **Local Development**
```bash
# Start development server
python3 comprehensive_working_system.py

# With auto-reload
uvicorn comprehensive_working_system:app --reload --host 0.0.0.0 --port 8087
```

### **Production Deployment**
```bash
# Install production dependencies
pip install gunicorn

# Start with Gunicorn
gunicorn comprehensive_working_system:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8087

# With Docker
docker build -t aura-intelligence .
docker run -p 8087:8087 aura-intelligence
```

### **Docker Configuration**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Start services and application
CMD service redis-server start && \
    python3 comprehensive_working_system.py
```

## 🔍 **Troubleshooting**

### **Common Issues**

#### **Redis Connection Issues**
```bash
# Check Redis status
sudo systemctl status redis-server

# Start Redis
sudo systemctl start redis-server

# Test connection
redis-cli ping
```

#### **Component Import Issues**
```python
# Check Python path
import sys
print(sys.path)

# Add paths if needed
sys.path.insert(0, '/path/to/core/src')
```

#### **Performance Issues**
```python
# Profile component performance
import cProfile
import pstats

def profile_component():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run component processing
    result = await component.process(test_data)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
```

### **Debugging Tools**

#### **Component Status Check**
```python
def debug_system_status():
    system = ComprehensiveWorkingSystem()
    
    print("System Status:")
    print(f"Working components: {len(system.working_components)}")
    print(f"Failed components: {len(system.failed_components)}")
    
    for name, component in system.working_components.items():
        health = component.health_check() if hasattr(component, 'health_check') else "unknown"
        print(f"  {name}: {health}")
    
    for name, error in system.failed_components.items():
        print(f"  {name}: FAILED - {error}")
```

#### **Performance Analysis**
```python
def analyze_performance():
    summary = performance_monitor.get_performance_summary()
    
    print("Performance Summary:")
    for component, metrics in summary.items():
        print(f"{component}:")
        print(f"  Requests: {metrics['count']}")
        print(f"  Avg Time: {metrics['avg_time']:.3f}s")
        print(f"  Min Time: {metrics['min_time']:.3f}s")
        print(f"  Max Time: {metrics['max_time']:.3f}s")
```

## 📚 **Best Practices**

### **Code Quality**
- Use type hints for all function parameters and return values
- Follow PEP 8 style guidelines
- Write comprehensive docstrings
- Implement proper error handling
- Use async/await for I/O operations

### **Testing**
- Write unit tests for all components
- Include integration tests for component interactions
- Add performance tests for critical paths
- Use fixtures for test data and setup
- Maintain high test coverage (>80%)

### **Documentation**
- Document all public APIs
- Include usage examples
- Maintain up-to-date README files
- Document configuration options
- Provide troubleshooting guides

### **Performance**
- Profile code regularly
- Optimize critical paths
- Use appropriate data structures
- Implement caching where beneficial
- Monitor resource usage

---

This development guide provides comprehensive information for developing, testing, and deploying AURA Intelligence components and systems.