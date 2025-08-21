# AURA Intelligence Integration Testing & Demo Framework

## 🚀 Overview

This is a production-grade integration testing and demo framework for AURA Intelligence microservices, built with 2025 best practices:

- **TestContainers** for isolated test environments
- **Contract Testing** with OpenAPI/AsyncAPI validation  
- **Chaos Engineering** for resilience testing
- **Interactive Demos** with real-time visualization
- **Performance Benchmarking** with detailed metrics

## 📁 Structure

```
integration/
├── framework/                    # Core testing frameworks
│   ├── testcontainers/          # Container orchestration
│   │   └── container_manager.py # Advanced TestContainers management
│   ├── contracts/               # Contract testing
│   │   └── contract_framework.py # CDC validation
│   └── chaos/                   # Chaos engineering
│       └── chaos_framework_2025.py # Fault injection
│
├── tests/                       # Test suites
│   ├── e2e/                    # End-to-end tests
│   │   └── test_full_stack_integration.py
│   ├── contracts/              # Contract definitions
│   ├── chaos/                  # Chaos experiments
│   └── performance/            # Performance tests
│
├── demos/                      # Demo applications
│   ├── interactive/            # Interactive UI
│   │   └── demo_framework.py   # Real-time demo runner
│   ├── scenarios/              # Pre-defined scenarios
│   └── data/                   # Sample data
│
├── run_integration_tests.py    # Main test orchestrator
└── requirements.txt            # Python dependencies
```

## 🛠️ Installation

```bash
# Install dependencies
cd /workspace/aura-microservices/integration
pip install -r requirements.txt

# Ensure Docker is running
docker --version
```

## 🧪 Running Tests

### Quick Start

```bash
# Run all integration tests
./run_integration_tests.py test

# Run specific test suites
./run_integration_tests.py test --skip-chaos      # Skip chaos tests
./run_integration_tests.py test --skip-contracts  # Skip contract tests
./run_integration_tests.py test --sequential      # Run sequentially

# Save results
./run_integration_tests.py test --output results.json
```

### Test Suites

1. **Contract Tests**: Validate API contracts between services
2. **Functional Tests**: End-to-end workflow validation
3. **Performance Tests**: Latency, throughput benchmarks
4. **Chaos Tests**: Resilience under failure conditions

## 🎮 Interactive Demos

### Launch Demo UI

```bash
# Start interactive demo server
./run_integration_tests.py demo

# Access at http://localhost:8888
```

### Available Scenarios

1. **Energy-Efficient Intelligence**
   - Demonstrates neuromorphic processing
   - Shows energy optimization
   - Real-time metrics visualization

2. **Real-Time Adaptation**
   - Liquid Neural Networks adapting live
   - No retraining required
   - Performance improvements

3. **Fault-Tolerant Decisions**
   - Byzantine consensus demonstration
   - Handling node failures
   - Maintaining quorum

## 🔥 Chaos Engineering

### Run Chaos Experiments

```bash
# Run chaos game day
./run_integration_tests.py chaos --experiments 3

# Dry run (no actual faults)
./run_integration_tests.py chaos --dry-run
```

### Fault Types

- Network partitions
- Service crashes
- Resource exhaustion
- Clock skew
- Cascading failures

## 📊 Key Features

### 1. TestContainers Management

```python
# Warm container pools for fast tests
manager = AuraTestContainers(use_compose=False)
await manager.prepare_warm_pools()

# Start services with health monitoring
async with manager.start_aura_stack() as service_urls:
    # Services available with automatic cleanup
```

### 2. Contract Validation

```python
# Consumer-driven contracts
runner = ContractTestRunner("./contracts")
results = await runner.run_consumer_tests(
    consumer_name="moe-router",
    provider_url="http://localhost:8000"
)
```

### 3. Chaos Experiments

```python
# Define hypothesis
hypothesis = ChaosHypothesis(
    description="System recovers from network partition",
    recovery_time_slo=timedelta(seconds=30),
    blast_radius=["neuromorphic", "moe"]
)

# Run experiment
result = await orchestrator.run_experiment(experiment)
```

### 4. Real-Time Demos

```python
# Run interactive scenario
scenario = DemoScenario(
    name="energy_efficient_intelligence",
    services=["neuromorphic", "memory", "moe"],
    steps=[...],
    visualizations=["energy_consumption", "latency_timeline"]
)
```

## 📈 Performance Benchmarks

The framework automatically collects:

- **Latency**: P50, P95, P99 percentiles
- **Throughput**: Requests per second
- **Energy**: Picojoules per operation
- **Success Rate**: % of successful requests
- **Recovery Time**: Time to restore steady state

## 🔧 Advanced Usage

### Custom Test Scenarios

```python
# Add to test_full_stack_integration.py
@pytest.mark.asyncio
async def test_custom_workflow(aura_stack):
    service_urls, _ = aura_stack
    # Your test logic here
```

### Custom Chaos Experiments

```python
# Define custom experiment
experiment = ChaosExperiment(
    name="custom_failure",
    hypothesis=ChaosHypothesis(...),
    fault_specs=[{
        "target": "memory",
        "type": "network_delay",
        "duration": 30,
        "intensity": 0.8
    }]
)
```

### Custom Demo Scenarios

```python
# Add to demo_framework.py
custom_scenario = DemoScenario(
    name="custom_demo",
    steps=[{
        "type": "custom",
        "function": your_custom_function,
        "params": {...}
    }]
)
```

## 🏆 Best Practices

1. **Isolation**: Each test runs in isolated containers
2. **Idempotency**: Tests can be run multiple times
3. **Observability**: All tests emit metrics and traces
4. **Resilience**: Automatic retries and cleanup
5. **Performance**: Parallel execution by default

## 🐛 Troubleshooting

### Common Issues

1. **Docker not running**
   ```bash
   sudo systemctl start docker
   ```

2. **Port conflicts**
   ```bash
   # Check for conflicts
   lsof -i :8000-8005
   ```

3. **Container cleanup**
   ```bash
   # Force cleanup
   docker container prune -f
   ```

## 📚 References

- [TestContainers Docs](https://testcontainers.com/)
- [Chaos Engineering Principles](https://principlesofchaos.org/)
- [Contract Testing](https://pact.io/)
- [OpenTelemetry](https://opentelemetry.io/)

## 🤝 Contributing

1. Add tests to appropriate directories
2. Follow existing patterns
3. Include documentation
4. Run full test suite before PR

---

Built with ❤️ for AURA Intelligence - August 2025