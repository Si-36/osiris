#!/bin/bash

# AURA Intelligence GPU Performance Benchmark Script
# Comprehensive benchmarking for GPU acceleration validation

set -euo pipefail

# Configuration
BENCHMARK_ITERATIONS=100
WARMUP_ITERATIONS=10
OUTPUT_FILE="/tmp/aura-gpu-benchmark-$(date +%Y%m%d-%H%M%S).json"

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
    exit 1
}

# Check system and GPU availability
check_system() {
    log "Checking system configuration..."
    
    # Check if AURA is running
    if ! curl -f http://localhost:8098/health &> /dev/null; then
        error "AURA Intelligence is not running. Start it first with: ./scripts/local-development.sh start"
    fi
    
    # Check GPU availability
    GPU_STATUS=$(curl -s http://localhost:8098/components | jq -r '.gpu_manager.status' 2>/dev/null || echo "unknown")
    if [ "$GPU_STATUS" != "healthy" ]; then
        warning "GPU manager status: $GPU_STATUS"
    else
        success "GPU manager is healthy"
    fi
    
    # System info
    echo "System Information:"
    echo "  CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
    echo "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
    
    if command -v nvidia-smi &> /dev/null; then
        echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
        echo "  CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"
    fi
    
    echo
}

# Benchmark BERT processing
benchmark_bert() {
    log "Benchmarking BERT processing..."
    
    local times=()
    local total_time=0
    
    # Warmup
    log "Warming up BERT model ($WARMUP_ITERATIONS iterations)..."
    for i in $(seq 1 $WARMUP_ITERATIONS); do
        curl -s -X POST http://localhost:8098/process \
            -H "Content-Type: application/json" \
            -d '{"data": {"text": "This is a warmup test for BERT processing"}, "query": "warmup"}' \
            > /dev/null
    done
    
    # Actual benchmark
    log "Running BERT benchmark ($BENCHMARK_ITERATIONS iterations)..."
    for i in $(seq 1 $BENCHMARK_ITERATIONS); do
        local start_time=$(date +%s.%N)
        
        local response=$(curl -s -X POST http://localhost:8098/process \
            -H "Content-Type: application/json" \
            -d '{"data": {"text": "This is a benchmark test for measuring BERT processing performance with GPU acceleration"}, "query": "benchmark"}')
        
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc -l)
        local duration_ms=$(echo "$duration * 1000" | bc -l)
        
        times+=($duration_ms)
        total_time=$(echo "$total_time + $duration_ms" | bc -l)
        
        # Show progress every 10 iterations
        if [ $((i % 10)) -eq 0 ]; then
            echo -n "."
        fi
    done
    echo
    
    # Calculate statistics
    local avg_time=$(echo "scale=3; $total_time / $BENCHMARK_ITERATIONS" | bc -l)
    local min_time=$(printf '%s\n' "${times[@]}" | sort -n | head -1)
    local max_time=$(printf '%s\n' "${times[@]}" | sort -n | tail -1)
    
    # Calculate median
    local sorted_times=($(printf '%s\n' "${times[@]}" | sort -n))
    local median_index=$((BENCHMARK_ITERATIONS / 2))
    local median_time=${sorted_times[$median_index]}
    
    # Calculate 95th percentile
    local p95_index=$((BENCHMARK_ITERATIONS * 95 / 100))
    local p95_time=${sorted_times[$p95_index]}
    
    echo "BERT Processing Results:"
    echo "  Average:      ${avg_time}ms"
    echo "  Median:       ${median_time}ms"
    echo "  Min:          ${min_time}ms"
    echo "  Max:          ${max_time}ms"
    echo "  95th %ile:    ${p95_time}ms"
    echo "  Target:       <50ms"
    
    # Check if target is met
    if (( $(echo "$avg_time < 50" | bc -l) )); then
        success "BERT performance target achieved!"
    else
        warning "BERT performance target not met (${avg_time}ms > 50ms)"
    fi
    
    # Save to JSON
    cat >> "$OUTPUT_FILE" << EOF
{
  "bert_benchmark": {
    "iterations": $BENCHMARK_ITERATIONS,
    "average_ms": $avg_time,
    "median_ms": $median_time,
    "min_ms": $min_time,
    "max_ms": $max_time,
    "p95_ms": $p95_time,
    "target_ms": 50,
    "target_met": $([ $(echo "$avg_time < 50" | bc -l) -eq 1 ] && echo "true" || echo "false")
  },
EOF
}

# Benchmark total pipeline
benchmark_pipeline() {
    log "Benchmarking total processing pipeline..."
    
    local times=()
    local total_time=0
    
    # Warmup
    for i in $(seq 1 $WARMUP_ITERATIONS); do
        curl -s -X POST http://localhost:8098/process \
            -H "Content-Type: application/json" \
            -d '{"data": {"values": [1,2,3,4,5]}, "query": "pipeline_warmup"}' \
            > /dev/null
    done
    
    # Actual benchmark
    log "Running pipeline benchmark ($BENCHMARK_ITERATIONS iterations)..."
    for i in $(seq 1 $BENCHMARK_ITERATIONS); do
        local response=$(curl -s -X POST http://localhost:8098/process \
            -H "Content-Type: application/json" \
            -d '{"data": {"values": [1,2,3,4,5,6,7,8,9,10]}, "query": "pipeline_benchmark"}')
        
        # Extract processing time from response
        local processing_time=$(echo "$response" | jq -r '.metrics.total_processing_time // 0' 2>/dev/null || echo "0")
        
        if [ "$processing_time" != "0" ] && [ "$processing_time" != "null" ]; then
            times+=($processing_time)
            total_time=$(echo "$total_time + $processing_time" | bc -l)
        fi
        
        if [ $((i % 10)) -eq 0 ]; then
            echo -n "."
        fi
    done
    echo
    
    if [ ${#times[@]} -eq 0 ]; then
        warning "No processing times available from pipeline benchmark"
        return
    fi
    
    # Calculate statistics
    local avg_time=$(echo "scale=3; $total_time / ${#times[@]}" | bc -l)
    local min_time=$(printf '%s\n' "${times[@]}" | sort -n | head -1)
    local max_time=$(printf '%s\n' "${times[@]}" | sort -n | tail -1)
    
    echo "Pipeline Processing Results:"
    echo "  Average:      ${avg_time}ms"
    echo "  Min:          ${min_time}ms"
    echo "  Max:          ${max_time}ms"
    echo "  Target:       <1ms"
    
    # Check if target is met
    if (( $(echo "$avg_time < 1" | bc -l) )); then
        success "Pipeline performance target achieved!"
    else
        warning "Pipeline performance target not met (${avg_time}ms > 1ms)"
    fi
    
    # Append to JSON
    sed -i '$s/,$//' "$OUTPUT_FILE"
    cat >> "$OUTPUT_FILE" << EOF
,
  "pipeline_benchmark": {
    "iterations": ${#times[@]},
    "average_ms": $avg_time,
    "min_ms": $min_time,
    "max_ms": $max_time,
    "target_ms": 1,
    "target_met": $([ $(echo "$avg_time < 1" | bc -l) -eq 1 ] && echo "true" || echo "false")
  }
}
EOF
}

# Test concurrent processing
test_concurrent() {
    log "Testing concurrent processing capability..."
    
    local concurrent_requests=10
    local pids=()
    
    # Start concurrent requests
    for i in $(seq 1 $concurrent_requests); do
        (
            curl -s -X POST http://localhost:8098/process \
                -H "Content-Type: application/json" \
                -d '{"data": {"text": "Concurrent processing test"}, "query": "concurrent_'$i'"}' \
                > /tmp/concurrent_$i.json
        ) &
        pids+=($!)
    done
    
    # Wait for all to complete
    local start_time=$(date +%s)
    for pid in "${pids[@]}"; do
        wait $pid
    done
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    # Check results
    local successful=0
    for i in $(seq 1 $concurrent_requests); do
        if [ -f "/tmp/concurrent_$i.json" ] && jq -e '.status' "/tmp/concurrent_$i.json" &> /dev/null; then
            ((successful++))
        fi
        rm -f "/tmp/concurrent_$i.json"
    done
    
    echo "Concurrent Processing Results:"
    echo "  Requests:     $concurrent_requests"
    echo "  Successful:   $successful"
    echo "  Duration:     ${total_duration}s"
    echo "  Success Rate: $((successful * 100 / concurrent_requests))%"
    
    if [ $successful -eq $concurrent_requests ]; then
        success "All concurrent requests succeeded"
    else
        warning "$((concurrent_requests - successful)) requests failed"
    fi
}

# Memory usage analysis
analyze_memory() {
    log "Analyzing memory usage..."
    
    # Get component status
    local components_response=$(curl -s http://localhost:8098/components)
    
    # Extract memory info if available
    local memory_usage=$(echo "$components_response" | jq -r '.memory_manager.memory_usage_mb // "unknown"' 2>/dev/null)
    local gpu_memory=$(echo "$components_response" | jq -r '.gpu_manager.memory_usage_gb // "unknown"' 2>/dev/null)
    
    echo "Memory Usage:"
    echo "  System Memory: ${memory_usage}MB"
    echo "  GPU Memory:    ${gpu_memory}GB"
    
    # Check for memory leaks by monitoring over time
    log "Monitoring memory stability (30 seconds)..."
    local initial_memory=$memory_usage
    sleep 30
    
    components_response=$(curl -s http://localhost:8098/components)
    local final_memory=$(echo "$components_response" | jq -r '.memory_manager.memory_usage_mb // "unknown"' 2>/dev/null)
    
    if [ "$initial_memory" != "unknown" ] && [ "$final_memory" != "unknown" ]; then
        local memory_diff=$(echo "$final_memory - $initial_memory" | bc -l 2>/dev/null || echo "0")
        echo "  Memory Change: ${memory_diff}MB over 30s"
        
        if (( $(echo "$memory_diff < 10" | bc -l) )); then
            success "Memory usage is stable"
        else
            warning "Potential memory leak detected (+${memory_diff}MB)"
        fi
    fi
}

# Generate performance report
generate_report() {
    log "Generating performance report..."
    
    echo
    echo "=================================="
    echo "AURA Intelligence Performance Report"
    echo "Generated: $(date)"
    echo "=================================="
    echo
    
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Detailed results saved to: $OUTPUT_FILE"
        echo
        
        # Show summary
        local bert_avg=$(jq -r '.bert_benchmark.average_ms' "$OUTPUT_FILE" 2>/dev/null || echo "N/A")
        local bert_target=$(jq -r '.bert_benchmark.target_met' "$OUTPUT_FILE" 2>/dev/null || echo "false")
        local pipeline_avg=$(jq -r '.pipeline_benchmark.average_ms' "$OUTPUT_FILE" 2>/dev/null || echo "N/A")
        local pipeline_target=$(jq -r '.pipeline_benchmark.target_met' "$OUTPUT_FILE" 2>/dev/null || echo "false")
        
        echo "Performance Summary:"
        echo "  BERT Processing:    ${bert_avg}ms (target: <50ms) [$([ "$bert_target" = "true" ] && echo "✓" || echo "✗")]"
        echo "  Pipeline Total:     ${pipeline_avg}ms (target: <1ms) [$([ "$pipeline_target" = "true" ] && echo "✓" || echo "✗")]"
        echo
        
        if [ "$bert_target" = "true" ] && [ "$pipeline_target" = "true" ]; then
            success "All performance targets achieved!"
        else
            warning "Some performance targets not met - check system configuration"
        fi
    else
        warning "No benchmark data available"
    fi
}

# Main benchmark execution
main() {
    case "${1:-full}" in
        "full")
            check_system
            echo > "$OUTPUT_FILE"  # Initialize JSON file
            benchmark_bert
            benchmark_pipeline
            test_concurrent
            analyze_memory
            generate_report
            ;;
        "bert")
            check_system
            echo > "$OUTPUT_FILE"
            benchmark_bert
            ;;
        "pipeline")
            check_system
            echo > "$OUTPUT_FILE"
            benchmark_pipeline
            ;;
        "concurrent")
            test_concurrent
            ;;
        "memory")
            analyze_memory
            ;;
        *)
            echo "AURA Intelligence GPU Benchmark"
            echo
            echo "Usage: $0 [test]"
            echo
            echo "Tests:"
            echo "  full       - Run all benchmarks (default)"
            echo "  bert       - BERT processing benchmark only"
            echo "  pipeline   - Pipeline benchmark only"
            echo "  concurrent - Concurrent processing test"
            echo "  memory     - Memory analysis"
            echo
            echo "Results are saved to: $OUTPUT_FILE"
            exit 1
            ;;
    esac
}

main "$@"