#!/bin/bash

# AURA Intelligence Production Deployment Script
# Automated deployment with GPU support, monitoring, and health checks

set -euo pipefail

# Configuration
NAMESPACE="aura-intelligence"
DOCKER_REGISTRY="your-registry.com"
IMAGE_TAG="${1:-latest}"
TIMEOUT=300
LOG_FILE="/tmp/aura-deploy-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if kubectl is installed and configured
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        error "docker is not installed or not in PATH"
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        warning "helm is not installed - some features may not be available"
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    # Check GPU nodes availability
    GPU_NODES=$(kubectl get nodes -l accelerator=nvidia-tesla-k80 --no-headers 2>/dev/null | wc -l || echo "0")
    if [ "$GPU_NODES" -eq 0 ]; then
        warning "No GPU nodes found in cluster - GPU acceleration will be disabled"
    else
        success "Found $GPU_NODES GPU nodes in cluster"
    fi
    
    success "Prerequisites check completed"
}

# Build and push Docker image
build_and_push() {
    log "Building and pushing Docker image..."
    
    # Build production image
    log "Building AURA Intelligence production image..."
    docker build -t "${DOCKER_REGISTRY}/aura-intelligence:${IMAGE_TAG}" \
        -f Dockerfile \
        --build-arg BUILD_ENV=production \
        --build-arg ENABLE_GPU=true \
        .
    
    # Push to registry
    log "Pushing image to registry..."
    docker push "${DOCKER_REGISTRY}/aura-intelligence:${IMAGE_TAG}"
    
    success "Docker image built and pushed successfully"
}

# Create namespace and apply base resources
setup_namespace() {
    log "Setting up namespace and base resources..."
    
    # Create namespace
    kubectl apply -f k8s/namespace.yaml
    
    # Apply configmaps and secrets
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secrets.yaml
    
    # Apply storage classes and persistent volumes
    kubectl apply -f k8s/storage.yaml
    
    success "Namespace and base resources created"
}

# Deploy Redis cluster
deploy_redis() {
    log "Deploying Redis cluster..."
    
    kubectl apply -f k8s/redis-deployment.yaml
    
    # Wait for Redis to be ready
    log "Waiting for Redis to be ready..."
    kubectl wait --for=condition=available --timeout=${TIMEOUT}s deployment/redis -n "$NAMESPACE"
    
    success "Redis cluster deployed successfully"
}

# Deploy monitoring stack
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    # Apply monitoring resources
    kubectl apply -f k8s/monitoring.yaml
    
    # Deploy Grafana dashboards
    kubectl create configmap grafana-dashboards \
        --from-file=monitoring/grafana/dashboards/ \
        -n "$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Prometheus alerts
    kubectl create configmap prometheus-alerts \
        --from-file=monitoring/prometheus/alerts/ \
        -n "$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    success "Monitoring stack deployed successfully"
}

# Deploy AURA Intelligence application
deploy_aura() {
    log "Deploying AURA Intelligence application..."
    
    # Update image tag in deployment
    sed -i "s|image: .*aura-intelligence:.*|image: ${DOCKER_REGISTRY}/aura-intelligence:${IMAGE_TAG}|g" k8s/aura-deployment.yaml
    
    # Apply main deployment
    kubectl apply -f k8s/aura-deployment.yaml
    
    # Apply ingress
    kubectl apply -f k8s/ingress.yaml
    
    # Wait for deployment to be ready
    log "Waiting for AURA deployment to be ready..."
    kubectl wait --for=condition=available --timeout=${TIMEOUT}s deployment/aura-intelligence -n "$NAMESPACE"
    
    success "AURA Intelligence application deployed successfully"
}

# Health checks
perform_health_checks() {
    log "Performing health checks..."
    
    # Get service endpoint
    INGRESS_IP=$(kubectl get ingress aura-ingress -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [ -z "$INGRESS_IP" ]; then
        INGRESS_IP=$(kubectl get service aura-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "localhost")
    fi
    
    # Wait for service to be responsive
    log "Waiting for service to be responsive..."
    for i in {1..30}; do
        if curl -f "http://${INGRESS_IP}/health" &> /dev/null; then
            success "Service is responsive"
            break
        fi
        sleep 10
        if [ $i -eq 30 ]; then
            error "Service failed to become responsive after 5 minutes"
        fi
    done
    
    # Perform comprehensive health check
    log "Performing comprehensive health check..."
    HEALTH_RESPONSE=$(curl -s "http://${INGRESS_IP}/health" || echo '{"status": "error"}')
    HEALTH_STATUS=$(echo "$HEALTH_RESPONSE" | jq -r '.status' 2>/dev/null || echo "unknown")
    
    if [ "$HEALTH_STATUS" = "healthy" ]; then
        success "All health checks passed"
    else
        warning "Health check returned status: $HEALTH_STATUS"
        log "Health response: $HEALTH_RESPONSE"
    fi
    
    # Check GPU availability
    GPU_STATUS=$(curl -s "http://${INGRESS_IP}/components" | jq -r '.gpu_manager.status' 2>/dev/null || echo "unknown")
    if [ "$GPU_STATUS" = "healthy" ]; then
        success "GPU acceleration is available"
    else
        warning "GPU acceleration status: $GPU_STATUS"
    fi
}

# Deployment rollback function
rollback_deployment() {
    log "Rolling back deployment..."
    kubectl rollout undo deployment/aura-intelligence -n "$NAMESPACE"
    kubectl wait --for=condition=available --timeout=${TIMEOUT}s deployment/aura-intelligence -n "$NAMESPACE"
    warning "Deployment rolled back to previous version"
}

# Main deployment function
main() {
    log "Starting AURA Intelligence production deployment..."
    log "Image tag: ${IMAGE_TAG}"
    log "Namespace: ${NAMESPACE}"
    log "Log file: ${LOG_FILE}"
    
    # Trap to handle errors
    trap 'error "Deployment failed. Check log file: $LOG_FILE"' ERR
    
    check_prerequisites
    
    # Optional: Build and push (skip if image already exists)
    if [ "${BUILD_IMAGE:-true}" = "true" ]; then
        build_and_push
    else
        log "Skipping image build (BUILD_IMAGE=false)"
    fi
    
    setup_namespace
    deploy_redis
    deploy_monitoring
    deploy_aura
    
    # Wait a bit for everything to settle
    sleep 30
    
    perform_health_checks
    
    success "AURA Intelligence deployment completed successfully!"
    success "Service endpoint: http://${INGRESS_IP:-localhost}"
    success "Grafana dashboard: http://${INGRESS_IP:-localhost}/grafana"
    success "Deployment log: $LOG_FILE"
    
    # Show resource usage
    log "Current resource usage:"
    kubectl top pods -n "$NAMESPACE" 2>/dev/null || log "Resource metrics not available"
}

# Script options
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback_deployment
        ;;
    "health")
        perform_health_checks
        ;;
    "logs")
        kubectl logs -f deployment/aura-intelligence -n "$NAMESPACE"
        ;;
    *)
        echo "Usage: $0 [deploy|rollback|health|logs] [image-tag]"
        echo "  deploy   - Full deployment (default)"
        echo "  rollback - Rollback to previous version"
        echo "  health   - Run health checks only"
        echo "  logs     - Follow application logs"
        exit 1
        ;;
esac