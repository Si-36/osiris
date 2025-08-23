# AURA Intelligence Production Scripts

This directory contains production-ready scripts for deploying, monitoring, and testing the AURA Intelligence system.

## Scripts Overview

### 🚀 `deploy-production.sh`
Complete production deployment script with GPU support, monitoring, and health checks.

```bash
# Full production deployment
./deploy-production.sh deploy

# Rollback to previous version
./deploy-production.sh rollback

# Health checks only
./deploy-production.sh health

# View application logs
./deploy-production.sh logs
```

**Features:**
- Automated Kubernetes deployment
- GPU node detection and configuration
- Redis cluster setup
- Monitoring stack deployment (Prometheus, Grafana)
- Comprehensive health checks
- Automatic rollback capability
- Production logging and error handling

### 📊 `monitor-system.sh`
Real-time monitoring script with automatic alerts and self-healing capabilities.

```bash
# Continuous monitoring
./monitor-system.sh monitor

# Single health check
./monitor-system.sh check

# Performance benchmarks
./monitor-system.sh benchmark

# Generate monitoring report
./monitor-system.sh report

# Force healing actions
./monitor-system.sh heal
```

**Features:**
- Pod health monitoring
- Resource usage tracking (CPU, Memory, GPU)
- Redis connectivity checks
- Performance benchmarking
- Self-healing capabilities
- Alert webhook integration
- Detailed monitoring reports

### 🔥 `load-test.sh`
Comprehensive load testing suite for production validation.

```bash
# Complete load test suite
./load-test.sh all

# Health endpoint tests only
./load-test.sh health

# GPU processing tests
./load-test.sh gpu

# System processing tests
./load-test.sh system

# Stress testing
./load-test.sh stress
```

**Environment Variables:**
```bash
export ENDPOINT=http://your-aura-instance:8080
export CONCURRENT_USERS=100
export DURATION=300
./load-test.sh all
```

**Features:**
- Multi-endpoint load testing
- GPU processing validation
- Concurrent user simulation
- Performance metrics collection
- Stress testing capabilities
- Detailed performance reports

## Quick Start

### 1. Production Deployment
```bash
# Set your Docker registry
export DOCKER_REGISTRY=your-registry.com

# Deploy to production
./deploy-production.sh deploy

# Monitor deployment
./monitor-system.sh monitor &

# Run load tests
export ENDPOINT=http://your-production-endpoint:8080
./load-test.sh all
```

### 2. Development Setup
```bash
# Build local image without pushing
export BUILD_IMAGE=false
./deploy-production.sh deploy

# Quick health check
./monitor-system.sh check
```

### 3. Performance Validation
```bash
# Quick performance test
./load-test.sh gpu

# Stress test with custom settings
export CONCURRENT_USERS=200
export DURATION=600
./load-test.sh stress
```

## Configuration

### Environment Variables

#### Deployment Configuration
- `DOCKER_REGISTRY`: Docker registry URL (default: your-registry.com)
- `IMAGE_TAG`: Docker image tag (default: latest)
- `BUILD_IMAGE`: Whether to build image (default: true)
- `GPU_ENABLED`: Enable GPU support (auto-detected)

#### Monitoring Configuration
- `NAMESPACE`: Kubernetes namespace (default: aura-production)
- `ALERT_WEBHOOK`: Slack/Teams webhook for alerts
- `AUTO_HEAL`: Enable self-healing (default: true)
- `CHECK_INTERVAL`: Monitoring interval in seconds (default: 30)

#### Load Testing Configuration
- `ENDPOINT`: Target endpoint URL (default: http://localhost:8080)
- `CONCURRENT_USERS`: Number of concurrent users (default: 50)
- `DURATION`: Test duration in seconds (default: 60)
- `RAMP_UP`: Ramp up time in seconds (default: 10)

## Prerequisites

### Required Tools
```bash
# Kubernetes
kubectl version

# Docker
docker version

# Basic utilities
curl --version
jq --version
bc --version
```

### Optional Tools
```bash
# For advanced monitoring
helm version

# For GPU monitoring
nvidia-smi
```

### Kubernetes Requirements
- Cluster with GPU nodes (for GPU acceleration)
- StorageClass for persistent volumes
- Ingress controller (for external access)
- Prometheus/Grafana (for monitoring)

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check GPU nodes
   kubectl get nodes -l accelerator=nvidia-tesla-k80
   
   # Verify GPU drivers
   kubectl exec -it <pod-name> -- nvidia-smi
   ```

2. **Service Not Responding**
   ```bash
   # Check pod status
   kubectl get pods -n aura-production
   
   # Check logs
   kubectl logs deployment/aura-intelligence -n aura-production
   
   # Force restart
   ./deploy-production.sh rollback
   ```

3. **High Resource Usage**
   ```bash
   # Check resource usage
   kubectl top pods -n aura-production
   kubectl top nodes
   
   # Scale down if needed
   kubectl scale deployment aura-intelligence --replicas=1 -n aura-production
   ```

### Performance Tuning

1. **GPU Optimization**
   - Ensure GPU nodes are properly labeled
   - Use appropriate resource requests/limits
   - Monitor GPU utilization

2. **Memory Optimization**
   - Configure appropriate JVM heap settings
   - Use memory-mapped files for large datasets
   - Implement proper caching strategies

3. **Network Optimization**
   - Use service mesh for microservices communication
   - Configure proper load balancing
   - Implement connection pooling

## Security Considerations

### Production Security
- Use non-root containers
- Implement network policies
- Rotate secrets regularly
- Enable RBAC
- Use encrypted storage

### Monitoring Security
- Secure webhook endpoints
- Use TLS for all communications
- Implement proper authentication
- Monitor for security events

## Performance Targets

### Response Times
- Health checks: < 50ms
- GPU processing: < 100ms
- System processing: < 200ms
- End-to-end pipeline: < 1000ms

### Throughput
- Health endpoint: > 1000 req/s
- Processing endpoint: > 100 req/s
- Concurrent users: > 200

### Resource Usage
- Memory: < 2GB per instance
- CPU: < 2 cores per instance
- GPU: < 50% utilization baseline

### Availability
- System uptime: > 99.9%
- Error rate: < 0.1%
- Recovery time: < 30 seconds

## Monitoring and Alerts

### Key Metrics
- Pod health status
- Resource utilization (CPU, Memory, GPU)
- Response times and throughput
- Error rates and exceptions
- Redis connectivity and performance

### Alert Thresholds
- High CPU usage: > 80%
- High memory usage: > 1GB
- Slow response time: > 100ms
- Pod failures: any unhealthy pods
- Redis connectivity issues

### Grafana Dashboards
- System overview dashboard
- Performance metrics dashboard
- Resource utilization dashboard
- Error tracking dashboard

## Contributing

### Adding New Scripts
1. Follow the existing naming convention
2. Include comprehensive error handling
3. Add configuration via environment variables
4. Include usage documentation
5. Add to this README

### Testing Scripts
```bash
# Test deployment script
./deploy-production.sh health

# Test monitoring script
./monitor-system.sh check

# Test load testing script
./load-test.sh health
```

## Support

For issues and support:
1. Check the troubleshooting section above
2. Review system logs and monitoring dashboards
3. Run health checks and performance tests
4. Create detailed issue reports with logs and metrics

---

**⚡ AURA Intelligence Production Scripts - Built for Scale, Performance, and Reliability**