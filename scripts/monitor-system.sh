#!/bin/bash

# AURA Intelligence Production Monitoring Script
# Real-time monitoring with automatic alerts and self-healing

set -euo pipefail

# Configuration
NAMESPACE="aura-production"
ALERT_WEBHOOK="${ALERT_WEBHOOK:-}"
HEALING_ENABLED="${AUTO_HEAL:-true}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

alert() {
    echo -e "${RED}[ALERT]${NC} $1"
    if [ -n "$ALERT_WEBHOOK" ]; then
        curl -X POST "$ALERT_WEBHOOK" \
            -H "Content-Type: application/json" \
            -d "{\"text\":\"🚨 AURA Alert: $1\"}" \
            &> /dev/null || true
    fi
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check system health
check_system_health() {
    log "Checking system health..."
    
    # Check pod status
    UNHEALTHY_PODS=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)
    
    if [ "$UNHEALTHY_PODS" -gt 0 ]; then
        alert "Found $UNHEALTHY_PODS unhealthy pods"
        kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running
        
        if [ "$HEALING_ENABLED" = "true" ]; then
            log "Attempting to restart unhealthy pods..."
            kubectl delete pods --field-selector=status.phase!=Running -n "$NAMESPACE" --force --grace-period=0
        fi
    else
        success "All pods are healthy"
    fi
    
    # Check service endpoints
    ENDPOINT=$(kubectl get service aura-intelligence -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")
    
    if [ -n "$ENDPOINT" ]; then
        if curl -f "http://${ENDPOINT}:8080/health" &> /dev/null; then
            success "Service endpoint is responsive"
        else
            alert "Service endpoint is not responding"
            if [ "$HEALING_ENABLED" = "true" ]; then
                log "Restarting service..."
                kubectl rollout restart deployment/aura-intelligence -n "$NAMESPACE"
            fi
        fi
    else
        alert "Service endpoint not available"
    fi
}

# Check resource usage
check_resource_usage() {
    log "Checking resource usage..."
    
    # Check memory usage
    MEMORY_USAGE=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '{sum+=$3} END {print sum}' || echo "0")
    MEMORY_THRESHOLD=1000  # MB
    
    if [ "${MEMORY_USAGE%Mi}" -gt "$MEMORY_THRESHOLD" 2>/dev/null ]; then
        warning "High memory usage: ${MEMORY_USAGE}"
    else
        success "Memory usage within limits: ${MEMORY_USAGE}"
    fi
    
    # Check CPU usage
    CPU_USAGE=$(kubectl top nodes --no-headers 2>/dev/null | awk '{sum+=$3} END {print sum/NR}' || echo "0")
    CPU_THRESHOLD=80  # %
    
    if [ "${CPU_USAGE%\%}" -gt "$CPU_THRESHOLD" 2>/dev/null ]; then
        warning "High CPU usage: ${CPU_USAGE}"
    else
        success "CPU usage within limits: ${CPU_USAGE}"
    fi
}

# Check GPU status
check_gpu_status() {
    log "Checking GPU status..."
    
    GPU_PODS=$(kubectl get pods -n "$NAMESPACE" -l app=aura-intelligence --no-headers 2>/dev/null | wc -l)
    
    if [ "$GPU_PODS" -gt 0 ]; then
        # Check if GPU is being utilized
        kubectl exec -n "$NAMESPACE" deployment/aura-intelligence -- nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | while read -r util; do
            if [ "$util" -lt 5 ]; then
                warning "Low GPU utilization: ${util}%"
            else
                success "GPU utilization: ${util}%"
            fi
        done || warning "Could not check GPU utilization"
    else
        warning "No GPU-enabled pods found"
    fi
}

# Check Redis connectivity
check_redis() {
    log "Checking Redis connectivity..."
    
    REDIS_POD=$(kubectl get pods -n "$NAMESPACE" -l app=redis --no-headers -o custom-columns=":metadata.name" | head -1)
    
    if [ -n "$REDIS_POD" ]; then
        if kubectl exec -n "$NAMESPACE" "$REDIS_POD" -- redis-cli ping 2>/dev/null | grep -q PONG; then
            success "Redis is responsive"
        else
            alert "Redis is not responding"
            if [ "$HEALING_ENABLED" = "true" ]; then
                log "Restarting Redis..."
                kubectl rollout restart deployment/redis -n "$NAMESPACE"
            fi
        fi
    else
        alert "Redis pod not found"
    fi
}

# Performance benchmarks
run_performance_test() {
    log "Running performance benchmarks..."
    
    ENDPOINT=$(kubectl get service aura-intelligence -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "localhost")
    
    if [ -n "$ENDPOINT" ]; then
        # Test response time
        RESPONSE_TIME=$(curl -w "%{time_total}" -s -o /dev/null "http://${ENDPOINT}:8080/health" || echo "999")
        RESPONSE_MS=$(echo "$RESPONSE_TIME * 1000" | bc 2>/dev/null || echo "999")
        
        if [ "${RESPONSE_MS%.*}" -lt 100 ]; then
            success "Response time: ${RESPONSE_MS%.*}ms"
        else
            warning "Slow response time: ${RESPONSE_MS%.*}ms"
        fi
        
        # Test GPU processing if available
        GPU_TIME=$(curl -s "http://${ENDPOINT}:8080/test/gpu" | jq -r '.processing_time_ms' 2>/dev/null || echo "N/A")
        if [ "$GPU_TIME" != "N/A" ] && [ "$GPU_TIME" != "null" ]; then
            if [ "${GPU_TIME%.*}" -lt 50 ]; then
                success "GPU processing time: ${GPU_TIME}ms"
            else
                warning "Slow GPU processing: ${GPU_TIME}ms"
            fi
        fi
    fi
}

# Generate monitoring report
generate_report() {
    log "Generating monitoring report..."
    
    REPORT_FILE="/tmp/aura-monitoring-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$REPORT_FILE" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "system_status": {
    "pods": $(kubectl get pods -n "$NAMESPACE" -o json | jq '.items | length'),
    "healthy_pods": $(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Running --no-headers | wc -l),
    "services": $(kubectl get services -n "$NAMESPACE" -o json | jq '.items | length'),
    "namespace": "$NAMESPACE"
  },
  "resource_usage": {
    "memory_usage": "$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '{sum+=$3} END {print sum}' || echo "Unknown")",
    "cpu_usage": "$(kubectl top nodes --no-headers 2>/dev/null | awk '{sum+=$3} END {print sum/NR}' || echo "Unknown")"
  },
  "performance_metrics": {
    "endpoint_response_time_ms": $(curl -w "%{time_total}" -s -o /dev/null "http://$(kubectl get service aura-intelligence -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "localhost"):8080/health" 2>/dev/null | awk '{print $1*1000}' || echo "null")
  }
}
EOF
    
    success "Monitoring report saved to: $REPORT_FILE"
    cat "$REPORT_FILE"
}

# Main monitoring loop
main() {
    log "Starting AURA Intelligence monitoring..."
    log "Namespace: $NAMESPACE"
    log "Auto-healing: $HEALING_ENABLED"
    log "Check interval: ${CHECK_INTERVAL}s"
    
    while true; do
        echo "=================================="
        log "Running monitoring checks..."
        
        check_system_health
        check_resource_usage
        check_gpu_status
        check_redis
        run_performance_test
        
        if [ "${GENERATE_REPORT:-false}" = "true" ]; then
            generate_report
        fi
        
        log "Monitoring cycle complete. Sleeping for ${CHECK_INTERVAL}s..."
        sleep "$CHECK_INTERVAL"
    done
}

# Handle script arguments
case "${1:-monitor}" in
    "monitor")
        main
        ;;
    "check")
        check_system_health
        check_resource_usage
        check_gpu_status
        check_redis
        ;;
    "benchmark")
        run_performance_test
        ;;
    "report")
        GENERATE_REPORT=true
        generate_report
        ;;
    "heal")
        HEALING_ENABLED=true
        check_system_health
        ;;
    *)
        echo "Usage: $0 [monitor|check|benchmark|report|heal]"
        echo "  monitor   - Continuous monitoring (default)"
        echo "  check     - Single health check"
        echo "  benchmark - Performance benchmark"
        echo "  report    - Generate monitoring report"
        echo "  heal      - Force healing actions"
        exit 1
        ;;
esac