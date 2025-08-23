#!/bin/bash

# AURA Intelligence Load Testing Script
# Comprehensive load testing for production validation

set -euo pipefail

# Configuration
ENDPOINT="${ENDPOINT:-http://localhost:8080}"
CONCURRENT_USERS="${CONCURRENT_USERS:-50}"
DURATION="${DURATION:-60}"
RAMP_UP="${RAMP_UP:-10}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v curl &> /dev/null; then
        error "curl is required but not installed"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        warning "jq not found - JSON parsing will be limited"
    fi
    
    # Test endpoint connectivity
    if ! curl -sf "$ENDPOINT/health" > /dev/null; then
        error "Cannot connect to endpoint: $ENDPOINT"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Single request test
single_request_test() {
    local endpoint="$1"
    local payload="$2"
    
    local start_time=$(date +%s.%N)
    local response=$(curl -s -w "HTTPSTATUS:%{http_code};TIME:%{time_total}" \
        -X POST "$ENDPOINT$endpoint" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null || echo "HTTPSTATUS:000;TIME:999")
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    
    local http_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    local response_time=$(echo "$response" | grep -o "TIME:[0-9.]*" | cut -d: -f2)
    local response_body=$(echo "$response" | sed 's/HTTPSTATUS:[0-9]*;TIME:[0-9.]*//')
    
    echo "$http_code,$response_time,$duration"
}

# Health check load test
health_check_test() {
    log "Running health check load test..."
    
    local results_file="/tmp/health_test_results.csv"
    echo "timestamp,http_code,response_time,duration" > "$results_file"
    
    local success_count=0
    local error_count=0
    local total_time=0
    
    log "Starting $CONCURRENT_USERS concurrent health checks for ${DURATION}s..."
    
    # Background processes
    local pids=()
    
    for ((i=1; i<=CONCURRENT_USERS; i++)); do
        (
            local end_time=$(($(date +%s) + DURATION))
            while [ $(date +%s) -lt $end_time ]; do
                local timestamp=$(date +%s.%N)
                local result=$(single_request_test "/health" "{}")
                echo "$timestamp,$result" >> "$results_file"
                sleep 0.1
            done
        ) &
        pids+=($!)
    done
    
    # Wait for all background processes
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    
    # Analyze results
    while IFS=, read -r timestamp http_code response_time duration; do
        if [ "$http_code" = "200" ]; then
            ((success_count++))
        else
            ((error_count++))
        fi
        total_time=$(echo "$total_time + $response_time" | bc)
    done < <(tail -n +2 "$results_file")
    
    local total_requests=$((success_count + error_count))
    local success_rate=$(echo "scale=2; $success_count * 100 / $total_requests" | bc)
    local avg_response_time=$(echo "scale=3; $total_time / $total_requests" | bc)
    
    log "Health Check Load Test Results:"
    log "  Total Requests: $total_requests"
    log "  Successful: $success_count"
    log "  Failed: $error_count"
    log "  Success Rate: ${success_rate}%"
    log "  Average Response Time: ${avg_response_time}s"
    
    if [ "$success_rate" \> "95" ]; then
        success "Health check load test passed"
    else
        warning "Health check load test below threshold (95%)"
    fi
}

# GPU processing load test
gpu_processing_test() {
    log "Running GPU processing load test..."
    
    local test_data='{"data": {"values": [1,2,3,4,5,6,7,8,9,10]}, "query": "optimize performance"}'
    local results_file="/tmp/gpu_test_results.csv"
    echo "timestamp,http_code,response_time,duration" > "$results_file"
    
    local success_count=0
    local error_count=0
    local total_gpu_time=0
    
    log "Testing GPU processing with $CONCURRENT_USERS concurrent requests..."
    
    # Sequential test for GPU processing (to avoid resource contention)
    local gpu_concurrent=$((CONCURRENT_USERS / 5))  # Reduce concurrency for GPU
    
    for ((i=1; i<=gpu_concurrent; i++)); do
        local result=$(single_request_test "/test/gpu" "$test_data")
        local timestamp=$(date +%s.%N)
        echo "$timestamp,$result" >> "$results_file"
        
        local http_code=$(echo "$result" | cut -d, -f1)
        if [ "$http_code" = "200" ]; then
            ((success_count++))
        else
            ((error_count++))
        fi
        
        # Small delay to prevent overwhelming GPU
        sleep 0.2
    done
    
    # Analyze results
    while IFS=, read -r timestamp http_code response_time duration; do
        total_gpu_time=$(echo "$total_gpu_time + $response_time" | bc)
    done < <(tail -n +2 "$results_file")
    
    local total_requests=$((success_count + error_count))
    local avg_gpu_time=$(echo "scale=3; $total_gpu_time / $total_requests" | bc)
    local avg_gpu_ms=$(echo "$avg_gpu_time * 1000" | bc)
    
    log "GPU Processing Load Test Results:"
    log "  Total Requests: $total_requests"
    log "  Successful: $success_count"
    log "  Failed: $error_count"
    log "  Average GPU Processing Time: ${avg_gpu_ms}ms"
    
    if [ "${avg_gpu_ms%.*}" -lt 100 ]; then
        success "GPU processing load test passed"
    else
        warning "GPU processing slower than expected (>100ms)"
    fi
}

# System processing load test
system_processing_test() {
    log "Running system processing load test..."
    
    local test_data='{"data": {"values": [1,2,3,4,5,6,7,8,9,10]}, "query": "process system data"}'
    local results_file="/tmp/system_test_results.csv"
    echo "timestamp,http_code,response_time,duration,processing_time" > "$results_file"
    
    local success_count=0
    local error_count=0
    local total_processing_time=0
    
    log "Testing system processing with load..."
    
    # Run concurrent system tests
    local pids=()
    
    for ((i=1; i<=CONCURRENT_USERS; i++)); do
        (
            local result=$(single_request_test "/process" "$test_data")
            local timestamp=$(date +%s.%N)
            
            # Try to extract processing time from response
            local processing_time="0"
            local response_body=$(curl -s -X POST "$ENDPOINT/process" \
                -H "Content-Type: application/json" \
                -d "$test_data" 2>/dev/null || echo "{}")
            
            if command -v jq &> /dev/null; then
                processing_time=$(echo "$response_body" | jq -r '.processing_time_ms // 0' 2>/dev/null || echo "0")
            fi
            
            echo "$timestamp,$result,$processing_time" >> "$results_file"
        ) &
        pids+=($!)
        
        # Stagger requests
        sleep 0.05
    done
    
    # Wait for completion
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    
    # Analyze results
    while IFS=, read -r timestamp http_code response_time duration processing_time; do
        if [ "$http_code" = "200" ]; then
            ((success_count++))
        else
            ((error_count++))
        fi
        total_processing_time=$(echo "$total_processing_time + $processing_time" | bc)
    done < <(tail -n +2 "$results_file")
    
    local total_requests=$((success_count + error_count))
    local success_rate=$(echo "scale=2; $success_count * 100 / $total_requests" | bc)
    local avg_processing_time=$(echo "scale=2; $total_processing_time / $total_requests" | bc)
    
    log "System Processing Load Test Results:"
    log "  Total Requests: $total_requests"
    log "  Successful: $success_count" 
    log "  Failed: $error_count"
    log "  Success Rate: ${success_rate}%"
    log "  Average Processing Time: ${avg_processing_time}ms"
    
    if [ "$success_rate" \> "90" ] && [ "${avg_processing_time%.*}" -lt 50 ]; then
        success "System processing load test passed"
    else
        warning "System processing performance below expectations"
    fi
}

# Stress test
stress_test() {
    log "Running stress test..."
    
    local stress_concurrent=$((CONCURRENT_USERS * 2))
    local stress_duration=$((DURATION / 2))
    
    log "Stress testing with $stress_concurrent concurrent users for ${stress_duration}s..."
    
    # Mix of different endpoints
    local endpoints=("/health" "/test/system" "/test/gpu" "/components")
    local pids=()
    
    for ((i=1; i<=stress_concurrent; i++)); do
        (
            local endpoint_index=$((i % ${#endpoints[@]}))
            local endpoint="${endpoints[$endpoint_index]}"
            local payload="{\"stress_test\": true, \"user_id\": $i}"
            
            local end_time=$(($(date +%s) + stress_duration))
            while [ $(date +%s) -lt $end_time ]; do
                single_request_test "$endpoint" "$payload" > /dev/null
                sleep 0.01  # Very short delay for stress
            done
        ) &
        pids+=($!)
    done
    
    # Monitor system during stress
    (
        local start_time=$(date +%s)
        while [ $(($(date +%s) - start_time)) -lt $stress_duration ]; do
            log "Stress test in progress... $(date)"
            sleep 5
        done
    ) &
    local monitor_pid=$!
    
    # Wait for completion
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    
    kill "$monitor_pid" 2>/dev/null || true
    
    success "Stress test completed"
}

# Generate load test report
generate_report() {
    log "Generating load test report..."
    
    local report_file="/tmp/aura-loadtest-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "test_configuration": {
    "endpoint": "$ENDPOINT",
    "concurrent_users": $CONCURRENT_USERS,
    "duration_seconds": $DURATION,
    "ramp_up_seconds": $RAMP_UP
  },
  "system_info": {
    "os": "$(uname -s)",
    "kernel": "$(uname -r)",
    "architecture": "$(uname -m)"
  },
  "endpoint_status": {
    "health_endpoint": "$(curl -s "$ENDPOINT/health" | jq -r '.status' 2>/dev/null || echo "unknown")",
    "response_time_ms": $(curl -w "%{time_total}" -s -o /dev/null "$ENDPOINT/health" 2>/dev/null | awk '{print $1*1000}' || echo "null")
  }
}
EOF
    
    success "Load test report generated: $report_file"
    cat "$report_file"
}

# Main execution
main() {
    log "Starting AURA Intelligence Load Testing"
    log "========================================="
    log "Endpoint: $ENDPOINT"
    log "Concurrent Users: $CONCURRENT_USERS"
    log "Duration: ${DURATION}s"
    
    check_prerequisites
    
    # Run test suite
    health_check_test
    sleep 2
    
    gpu_processing_test
    sleep 2
    
    system_processing_test
    sleep 2
    
    stress_test
    
    generate_report
    
    success "All load tests completed!"
}

# Script options
case "${1:-all}" in
    "all")
        main
        ;;
    "health")
        check_prerequisites
        health_check_test
        ;;
    "gpu")
        check_prerequisites
        gpu_processing_test
        ;;
    "system")
        check_prerequisites
        system_processing_test
        ;;
    "stress")
        check_prerequisites
        stress_test
        ;;
    "report")
        generate_report
        ;;
    *)
        echo "Usage: $0 [all|health|gpu|system|stress|report]"
        echo "  all    - Run complete load test suite (default)"
        echo "  health - Health endpoint load test only"
        echo "  gpu    - GPU processing load test only"
        echo "  system - System processing load test only" 
        echo "  stress - Stress test only"
        echo "  report - Generate report only"
        echo ""
        echo "Environment variables:"
        echo "  ENDPOINT=$ENDPOINT"
        echo "  CONCURRENT_USERS=$CONCURRENT_USERS"
        echo "  DURATION=$DURATION"
        exit 1
        ;;
esac