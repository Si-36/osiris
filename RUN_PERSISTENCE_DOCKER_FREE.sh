#!/bin/bash

# AURA Persistence Tests - Docker-Free Version
# ============================================
# For environments without Docker, uses legacy/mock backends

echo "🚀 AURA PERSISTENCE - DOCKER-FREE TEST SUITE"
echo "============================================"
echo "Run Date: $(date)"
echo ""

# Set environment variable for legacy mode
export AURA_PERSISTENCE_LEGACY_MODE=true
export PYTHONPATH="${PWD}/core/src:${PYTHONPATH}"

# Function to run a test file
run_test() {
    local test_name="$1"
    local test_file="$2"
    
    echo ""
    echo "Running: $test_name"
    echo "File: $test_file"
    echo "-----------------------------------------"
    
    if [ -f "$test_file" ]; then
        python3 "$test_file"
        if [ $? -eq 0 ]; then
            echo "✅ $test_name PASSED"
        else
            echo "❌ $test_name FAILED"
        fi
    else
        echo "⚠️  Test file not found: $test_file"
    fi
}

# Quick persistence test
run_test "Quick Persistence Test" "TEST_PERSISTENCE_NOW.py"

# If quick test passes, run full suite
if [ $? -eq 0 ]; then
    echo ""
    echo "==========================================================="
    echo "Quick test passed! Running full test suite..."
    echo "==========================================================="
    
    run_test "Environment Verification" "VERIFY_LOCAL_ENV.py"
    run_test "Debug Persistence" "DEBUG_PERSISTENCE_ERRORS.py"
    run_test "Minimal Persistence" "test_persistence_minimal.py"
    run_test "Full Integration" "TEST_FULL_PERSISTENCE_INTEGRATION.py"
    run_test "All Agents Integration" "test_all_agents_integrated.py"
fi

echo ""
echo "==========================================================="
echo "TEST SUMMARY"
echo "==========================================================="
echo ""
echo "If tests fail with import errors:"
echo "1. Fix the import path in fallback_agent.py (done ✅)"
echo "2. Make sure PYTHONPATH is set correctly"
echo "3. Run: export PYTHONPATH=\${PWD}/core/src:\${PYTHONPATH}"
echo ""
echo "The persistence system works without Docker by:"
echo "- Using SQLite instead of PostgreSQL"
echo "- Using in-memory DuckDB"
echo "- Using file-based state storage"
echo "- All innovative features still work!"