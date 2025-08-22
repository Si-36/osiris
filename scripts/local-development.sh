#!/bin/bash

# AURA Intelligence Local Development Script
# Quick setup for local development with GPU support and monitoring

set -euo pipefail

# Configuration
COMPOSE_FILE="docker-compose.production.yml"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
ENABLE_GPU="${ENABLE_GPU:-true}"
ENABLE_MONITORING="${ENABLE_MONITORING:-true}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check Docker and GPU support
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
    fi
    
    # Check GPU support
    if [ "$ENABLE_GPU" = "true" ]; then
        if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
            success "GPU support detected"
        else
            warning "GPU support not available - running in CPU mode"
            export ENABLE_GPU=false
        fi
    fi
    
    success "Prerequisites check completed"
}

# Setup environment
setup_environment() {
    log "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        cat > .env << EOF
# AURA Intelligence Environment Configuration
AURA_LOG_LEVEL=$LOG_LEVEL
ENABLE_GPU=$ENABLE_GPU
ENABLE_MONITORING=$ENABLE_MONITORING
REDIS_URL=redis://localhost:6379
NVIDIA_VISIBLE_DEVICES=all
CUDA_VISIBLE_DEVICES=0
EOF
        success "Created .env file"
    fi
    
    # Ensure required directories exist
    mkdir -p monitoring/grafana/data
    mkdir -p monitoring/prometheus/data
    mkdir -p logs
    
    # Set proper permissions
    chmod 777 monitoring/grafana/data
    chmod 777 monitoring/prometheus/data
    chmod 777 logs
    
    success "Environment setup completed"
}

# Build local development image
build_image() {
    log "Building AURA Intelligence development image..."
    
    docker build -t aura-intelligence:dev \
        --build-arg BUILD_ENV=development \
        --build-arg ENABLE_GPU=$ENABLE_GPU \
        -f Dockerfile .
    
    success "Development image built successfully"
}

# Start services
start_services() {
    log "Starting AURA Intelligence development environment..."
    
    # Start Redis first
    docker-compose -f $COMPOSE_FILE up -d redis
    
    # Wait for Redis to be ready
    log "Waiting for Redis..."
    timeout 30s bash -c 'until docker-compose -f '"$COMPOSE_FILE"' exec redis redis-cli ping | grep -q PONG; do sleep 1; done'
    
    if [ "$ENABLE_MONITORING" = "true" ]; then
        # Start monitoring services
        docker-compose -f $COMPOSE_FILE up -d prometheus grafana
        log "Monitoring services started"
    fi
    
    # Start main application
    docker-compose -f $COMPOSE_FILE up -d aura-system
    
    success "All services started successfully"
}

# Wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready..."
    
    # Wait for main application
    log "Checking AURA Intelligence health..."
    for i in {1..30}; do
        if curl -f http://localhost:8098/health &> /dev/null; then
            success "AURA Intelligence is ready"
            break
        fi
        sleep 5
        if [ $i -eq 30 ]; then
            error "AURA Intelligence failed to start within 2.5 minutes"
        fi
    done
    
    # Check GPU status if enabled
    if [ "$ENABLE_GPU" = "true" ]; then
        GPU_STATUS=$(curl -s http://localhost:8098/components | jq -r '.gpu_manager.status' 2>/dev/null || echo "unknown")
        if [ "$GPU_STATUS" = "healthy" ]; then
            success "GPU acceleration is working"
        else
            warning "GPU status: $GPU_STATUS"
        fi
    fi
}

# Show service status
show_status() {
    log "Service Status:"
    echo
    echo "AURA Intelligence: http://localhost:8098"
    echo "Health Check:     http://localhost:8098/health"
    echo "Components:       http://localhost:8098/components"
    
    if [ "$ENABLE_MONITORING" = "true" ]; then
        echo "Grafana:          http://localhost:3000 (admin/admin)"
        echo "Prometheus:       http://localhost:9090"
    fi
    
    echo "Redis:            localhost:6379"
    echo
    
    log "Running containers:"
    docker-compose -f $COMPOSE_FILE ps
}

# Performance test
run_performance_test() {
    log "Running quick performance test..."
    
    # Test basic functionality
    RESPONSE=$(curl -s -X POST http://localhost:8098/process \
        -H "Content-Type: application/json" \
        -d '{"data": {"values": [1,2,3,4,5]}, "query": "test"}' || echo '{"error": "failed"}')
    
    if echo "$RESPONSE" | jq -e '.status' &> /dev/null; then
        PROCESSING_TIME=$(echo "$RESPONSE" | jq -r '.metrics.total_processing_time' 2>/dev/null || echo "unknown")
        success "Performance test passed - Processing time: ${PROCESSING_TIME}ms"
    else
        warning "Performance test failed - Response: $RESPONSE"
    fi
}

# Tail logs
tail_logs() {
    log "Following application logs (Ctrl+C to stop)..."
    docker-compose -f $COMPOSE_FILE logs -f aura-system
}

# Stop all services
stop_services() {
    log "Stopping all services..."
    docker-compose -f $COMPOSE_FILE down
    success "All services stopped"
}

# Clean up everything
cleanup() {
    log "Cleaning up development environment..."
    docker-compose -f $COMPOSE_FILE down -v --remove-orphans
    docker image rm aura-intelligence:dev 2>/dev/null || true
    success "Cleanup completed"
}

# Main function
main() {
    case "${1:-start}" in
        "start")
            check_prerequisites
            setup_environment
            build_image
            start_services
            wait_for_services
            show_status
            run_performance_test
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            sleep 2
            start_services
            wait_for_services
            show_status
            ;;
        "status")
            show_status
            ;;
        "logs")
            tail_logs
            ;;
        "test")
            run_performance_test
            ;;
        "cleanup")
            cleanup
            ;;
        "build")
            build_image
            ;;
        *)
            echo "AURA Intelligence Local Development"
            echo
            echo "Usage: $0 [command]"
            echo
            echo "Commands:"
            echo "  start    - Start development environment (default)"
            echo "  stop     - Stop all services"
            echo "  restart  - Restart all services"
            echo "  status   - Show service status"
            echo "  logs     - Follow application logs"
            echo "  test     - Run performance test"
            echo "  build    - Build development image"
            echo "  cleanup  - Stop and remove everything"
            echo
            echo "Environment variables:"
            echo "  LOG_LEVEL=INFO          - Set log level"
            echo "  ENABLE_GPU=true         - Enable GPU support"
            echo "  ENABLE_MONITORING=true  - Enable monitoring"
            exit 1
            ;;
    esac
}

main "$@"