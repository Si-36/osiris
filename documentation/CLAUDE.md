# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Repository Overview

AURA Intelligence is a comprehensive AI platform that integrates 200+ AI components across neural networks, consciousness systems, memory management, multi-agent orchestration, and advanced AI capabilities. It represents a production-ready artificial intelligence system built with modular architecture and real-time processing capabilities.

The system is organized into several key subsystems:

- **Core AI Engine**: Advanced neural networks, consciousness systems, and memory management (`core/src/aura_intelligence/`)
- **Ultimate API System**: High-performance API layer with MAX Python acceleration (`ultimate_api_system/`)
- **Multi-Agent Systems**: Council-based agent coordination and workflows (`core/src/aura_intelligence/agents/`)
- **Neural Components**: Liquid Neural Networks, Mamba2, and advanced architectures (`core/src/aura_intelligence/neural/`)
- **Memory Systems**: Redis-based storage, hybrid memory pools, and pattern recognition (`core/src/aura_intelligence/memory/`)
- **Distributed Computing**: Ray integration, consensus algorithms, and orchestration (`core/src/aura_intelligence/distributed/`)

## Essential Build Commands

### Development Setup
```bash
# Install Redis server (required for memory systems)
sudo apt install redis-server
sudo systemctl start redis-server

# Install Python dependencies
pip install -r requirements.txt

# Alternative: Install with optional dependencies
pip install -e ".[dev,docs,performance]"

# Install MAX Python for acceleration (optional)
pip install modular --index-url https://dl.modular.com/public/nightly/python/simple/
```

### Running the System
```bash
# Main entry point - starts complete AURA system
python3 main.py

# Start with shell script
./start.sh

# Start production system
python3 production_aura_2025.py

# Start ultimate API system directly
python3 ultimate_api_system/max_aura_api.py

# Docker deployment
docker-compose up -d
```

### Testing Commands
```bash
# Run comprehensive system tests
python3 test_enhanced_systems.py

# Run all component tests
python3 test_all_components.py

# Run real system status tests
python3 test_real_system_status.py

# Run integration tests
python3 comprehensive_integration_test.py

# Run specific component tests
python3 -m pytest core/src/aura_intelligence/agents/council/ -v

# Run performance tests
python3 test_bio_enhanced_system.py
```

### Development Tools
```bash
# Format code with black
black core/ ultimate_api_system/ *.py

# Lint with ruff
ruff check core/ ultimate_api_system/ *.py

# Type checking with mypy
mypy core/src/aura_intelligence/

# Run pre-commit hooks
pre-commit run --all-files
```

## High-Level Architecture

### Repository Structure

```text
osiris-2/
├── core/src/aura_intelligence/    # Core AI components (200+ modules)
│   ├── agents/                    # Multi-agent systems and councils
│   │   ├── council/               # LNN-based agent council
│   │   ├── temporal/              # Temporal workflow agents
│   │   └── v2/                    # Next-gen agent architectures
│   ├── consciousness/             # Global Workspace Theory implementation
│   ├── memory/                    # Redis-based storage and hybrid memory
│   ├── neural/                    # Liquid Neural Networks and Mamba2
│   ├── tda/                       # Topological Data Analysis
│   ├── lnn/                       # Liquid Neural Network implementations
│   ├── orchestration/             # Workflow orchestration and coordination
│   ├── distributed/               # Ray integration and consensus algorithms
│   ├── observability/             # Monitoring, metrics, and tracing
│   ├── governance/                # Constitutional AI and policy engines
│   └── coral/                     # Advanced communication protocols
├── ultimate_api_system/           # MAX-accelerated API layer
│   ├── components/                # MAX-optimized components
│   ├── api/                       # FastAPI endpoints
│   ├── core/                      # Core system integration
│   └── monitoring/                # Performance monitoring
├── aura_intelligence_api/         # Alternative API implementation
├── docs/                          # System documentation
├── tests/                         # Test suites
└── archive/                       # Development history and backups
```

### Key Architectural Patterns

1. **Modular Component Design**:
   - Each component implements standard interfaces for health checks and processing
   - Components can be independently developed, tested, and deployed
   - Graceful degradation when components fail

2. **Unified System Coordination**:
   - Central coordination through `core.unified_system.py`
   - Standardized component registration and lifecycle management
   - Consistent data flow patterns across all components

3. **Memory-First Architecture**:
   - Redis-based persistent storage for all system state
   - Pattern recognition and storage in memory systems
   - Hybrid memory pools combining different storage tiers

4. **Agent Council Pattern**:
   - LNN-based agent councils for complex decision making
   - Multi-agent coordination with confidence scoring
   - Fallback mechanisms and consensus algorithms

5. **Real-time Processing**:
   - Sub-millisecond response times for complete pipelines
   - Asynchronous processing with async/await patterns
   - Efficient resource utilization and connection pooling

### Data Flow Architecture

The system processes requests through a 7-stage pipeline:

1. **Input Processing**: Request validation and data normalization
2. **Memory Storage**: Pattern storage in Redis with key generation
3. **Consciousness Processing**: Global Workspace activation and attention
4. **System Coordination**: Unified System orchestration and state management
5. **Communication Routing**: Message distribution and event propagation
6. **Pattern Integration**: Memory pattern retrieval and relevance scoring
7. **Output Generation**: Response assembly and final validation

Total processing time: < 1ms for complete pipeline

## Development Workflow

### Component Development

When adding new components:

1. **Create component directory**: `core/src/aura_intelligence/your_component/`
2. **Implement standard interfaces**: Inherit from `aura_common.atomic.base.Component`
3. **Add health checks**: Implement `health_check()` method
4. **Register with unified system**: Use `core.unified_interfaces.register_component()`
5. **Add tests**: Create comprehensive tests in `tests/` or inline
6. **Update documentation**: Add to relevant docs in `docs/`

### Agent Development

For multi-agent systems:

1. **Use council pattern**: Extend `agents.council.core_agent.py`
2. **Implement LNN integration**: Use `agents.council.lnn/` modules
3. **Add confidence scoring**: Use `agents.council.confidence_scoring.py`
4. **Configure workflows**: Use `orchestration.workflows/` for complex flows
5. **Add temporal support**: Use `agents.temporal/` for durable workflows

### API Development

For API endpoints:

1. **Ultimate API System**: Use `ultimate_api_system/` for MAX-accelerated APIs
2. **Standard APIs**: Use `aura_intelligence_api/` for standard FastAPI endpoints
3. **Component integration**: Register components with unified system
4. **Add monitoring**: Use `observability/` modules for metrics and tracing
5. **Performance optimization**: Leverage MAX Python APIs for 100-1000x speedup

### Testing Strategy

The system uses comprehensive testing patterns:

- **Component Tests**: Individual component functionality
- **Integration Tests**: Component interaction and data flow
- **System Tests**: End-to-end system functionality
- **Performance Tests**: Latency, throughput, and resource utilization
- **Real System Tests**: Validate actual functionality (not mocks)

### Memory System Integration

All components should integrate with the memory system:

1. **Use Redis adapter**: `adapters.redis_adapter.RedisAdapter`
2. **Pattern storage**: Store data with `pattern_{timestamp}_{index}` keys
3. **Health monitoring**: Implement Redis connection health checks
4. **Memory patterns**: Use structured JSON for data storage
5. **Cleanup policies**: Implement appropriate data retention

## Critical Development Notes

### Performance Considerations

- **Redis Connection Pooling**: Use connection pools for Redis operations
- **Async Operations**: Use async/await for I/O operations
- **Memory Efficiency**: Minimize memory allocations in hot paths
- **GPU Acceleration**: Use MAX Python APIs for computation-heavy operations
- **Connection Management**: Properly close connections and handle timeouts

### Common Patterns

1. **Component Health Checks**:
```python
async def health_check(self) -> Dict[str, Any]:
    try:
        # Component-specific check
        await self.ping()
        return {"status": "healthy", "component": self.name}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

2. **Memory Pattern Storage**:
```python
pattern_key = f"pattern_{int(time.time())}_{index}"
pattern_data = {
    "original_data": data,
    "processed_at": time.time(),
    "status": "processed"
}
await redis_adapter.store_pattern(pattern_key, pattern_data)
```

3. **Component Registration**:
```python
from core.src.aura_intelligence.core.unified_interfaces import register_component
component = YourComponent(component_id="your_component", config={})
register_component(component, "your_category")
```

### Security Considerations

- **Input Validation**: Validate all external inputs using Pydantic models
- **Redis Security**: Use Redis AUTH when available
- **Error Handling**: Never expose internal errors to external APIs
- **Resource Limits**: Implement proper timeout and resource management
- **Component Isolation**: Components should not directly access others' internals

### Common Pitfalls

- **Redis Connection Leaks**: Always properly close Redis connections
- **Memory Leaks**: Be careful with circular references in agent systems
- **Blocking Operations**: Use async versions of all I/O operations
- **Component Dependencies**: Avoid tight coupling between components
- **Error Propagation**: Handle errors gracefully with fallback mechanisms

### Environment Variables

Key environment variables for system configuration:

- `REDIS_URL`: Redis connection string (default: redis://localhost:6379)
- `AURA_LOG_LEVEL`: Logging level (default: INFO)
- `AURA_MAX_WORKERS`: Number of worker processes
- `AURA_GPU_ENABLED`: Enable GPU acceleration (true/false)

## API Endpoints

### Core API Endpoints

- **Health Check**: `GET /health` - System and component health status
- **Component Status**: `GET /components` - Detailed component information
- **Process Request**: `POST /process` - Main processing endpoint
- **System Status**: `GET /` - Overall system status and metrics

### Ultimate API System

- **Process Request**: `POST /api/v2/process` - MAX-accelerated processing
- **Batch Processing**: `POST /api/v2/batch` - Batch request processing
- **WebSocket Stream**: `WS /ws/stream` - Real-time streaming
- **Model Status**: `GET /api/v2/models` - Available models and status
- **Performance Metrics**: `GET /api/v2/metrics` - System performance data

## Testing Requirements

Before committing:

```bash
# Run system health tests
curl http://localhost:8080/health

# Run comprehensive component tests
python3 test_all_components.py

# Validate memory system integration
python3 core/src/aura_intelligence/memory/smoke_test.py

# Test API endpoints
curl -X POST http://localhost:8080/process \
  -H "Content-Type: application/json" \
  -d '{"data": {"values": [1,2,3,4,5]}, "query": "test"}'

# Performance validation
python3 test_bio_enhanced_system.py
```

## Production Deployment

### Docker Deployment
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f aura-system

# Scale services
docker-compose up -d --scale aura-system=3
```

### Monitoring Stack
- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Visualization dashboard (port 3000)
- **Redis**: Memory system backend (port 6379)
- **AURA System**: Main application (port 8098)

### Performance Targets
- **Response Time**: < 1ms for complete processing pipeline
- **Throughput**: 1000+ requests/second
- **Memory Usage**: < 100MB baseline
- **Component Health**: 99.9%+ uptime
- **Redis Performance**: Sub-millisecond read/write operations

This comprehensive AI system represents the state-of-the-art in integrated artificial intelligence platforms, combining advanced neural architectures, consciousness systems, and multi-agent coordination in a production-ready framework.