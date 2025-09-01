#!/bin/bash
# Complete Persistence Test Suite for Local Environment
# Run this where you have all dependencies installed

echo "==========================================================="
echo "AURA PERSISTENCE - COMPLETE TEST SUITE"
echo "Run Date: $(date)"
echo "==========================================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run a test and check result
run_test() {
    local test_name=$1
    local test_file=$2
    
    echo -e "\n${YELLOW}Running: $test_name${NC}"
    echo "File: $test_file"
    echo "-----------------------------------------"
    
    if python3 "$test_file"; then
        echo -e "${GREEN}✅ $test_name PASSED${NC}"
        return 0
    else
        echo -e "${RED}❌ $test_name FAILED${NC}"
        return 1
    fi
}

# Track results
total_tests=0
passed_tests=0

# Test 1: Environment verification
((total_tests++))
if run_test "Environment Verification" "VERIFY_LOCAL_ENV.py"; then
    ((passed_tests++))
fi

# Test 2: Debug persistence
((total_tests++))
if run_test "Debug Persistence" "DEBUG_PERSISTENCE_ERRORS.py"; then
    ((passed_tests++))
fi

# Test 3: Minimal persistence
((total_tests++))
if run_test "Minimal Persistence" "test_persistence_minimal.py"; then
    ((passed_tests++))
fi

# Test 4: Full integration
((total_tests++))
if run_test "Full Integration" "TEST_FULL_PERSISTENCE_INTEGRATION.py"; then
    ((passed_tests++))
fi

# Test 5: All agents (if minimal passes)
if [ $passed_tests -ge 3 ]; then
    ((total_tests++))
    if run_test "All Agents Integration" "test_all_agents_integrated.py"; then
        ((passed_tests++))
    fi
fi

# Summary
echo -e "\n==========================================================="
echo "TEST SUMMARY"
echo "==========================================================="
echo "Total Tests: $total_tests"
echo -e "Passed: ${GREEN}$passed_tests${NC}"
echo -e "Failed: ${RED}$((total_tests - passed_tests))${NC}"

if [ $passed_tests -eq $total_tests ]; then
    echo -e "\n${GREEN}🎉 ALL TESTS PASSED! 🎉${NC}"
    echo ""
    echo "The AURA Persistence System is fully operational with:"
    echo "✅ Causal state tracking"
    echo "✅ Memory evolution"
    echo "✅ Neural checkpointing"
    echo "✅ TDA persistence"
    echo "✅ GPU optimization"
    echo ""
    echo "Next steps:"
    echo "1. Start Docker services: docker compose -f docker-compose.persistence.yml up -d"
    echo "2. Run performance benchmarks"
    echo "3. Deploy to production"
else
    echo -e "\n${YELLOW}⚠️  Some tests failed${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check error messages above"
    echo "2. Ensure PostgreSQL is running or use legacy mode"
    echo "3. Verify all dependencies: pip install -r requirements-persistence.txt"
fi

exit $((total_tests - passed_tests))