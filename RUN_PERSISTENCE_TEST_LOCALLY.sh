#!/bin/bash

# Complete test cycle for AURA Persistence System
# Run this in your local environment

echo "🚀 AURA Persistence Test - Complete Cycle"
echo "=========================================="
echo ""

# Step 1: Ensure we're in the right directory
echo "📍 Step 1: Setting up environment..."
cd ~/projects/osiris-2 || exit 1

# Step 2: Activate virtual environment
echo "🐍 Step 2: Activating virtual environment..."
source aura_venv/bin/activate || {
    echo "❌ Failed to activate virtual environment"
    exit 1
}

# Step 3: Install any missing dependencies
echo "📦 Step 3: Checking dependencies..."
pip install -q asyncpg duckdb zstandard aiokafka langgraph || {
    echo "⚠️  Some dependencies might already be installed"
}

# Step 4: Export PYTHONPATH
echo "🔧 Step 4: Setting PYTHONPATH..."
export PYTHONPATH="${PWD}/core/src:${PYTHONPATH}"

# Step 5: Run the comprehensive test with detailed output
echo "🧪 Step 5: Running persistence tests..."
echo ""
echo "========== TEST OUTPUT BEGINS =========="
python3 TEST_PERSISTENCE_NOW.py 2>&1 | tee persistence_test_output.log

# Capture exit code
TEST_EXIT_CODE=${PIPESTATUS[0]}

echo "========== TEST OUTPUT ENDS =========="
echo ""

# Step 6: Analyze results
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ PERSISTENCE TESTS PASSED!"
    echo ""
    echo "📊 Test Summary:"
    grep -E "✅|PASSED|Success" persistence_test_output.log | tail -20
else
    echo "❌ PERSISTENCE TESTS FAILED (Exit code: $TEST_EXIT_CODE)"
    echo ""
    echo "🔍 Last Error:"
    grep -E "Error:|Traceback|IndentationError|SyntaxError|ImportError|TypeError|AttributeError" persistence_test_output.log | tail -20
    echo ""
    echo "📋 Full log saved to: persistence_test_output.log"
fi

echo ""
echo "🎯 Next Steps:"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "1. All persistence tests passed!"
    echo "2. You can now proceed with agent integration"
    echo "3. Run: python3 test_persistence_integration_complete.py"
else
    echo "1. Review the error above"
    echo "2. The full log is in: persistence_test_output.log"
    echo "3. Share the error output for debugging"
fi

echo ""
echo "📝 Debug Information:"
echo "- Python version: $(python3 --version)"
echo "- Working directory: $(pwd)"
echo "- Virtual env: $VIRTUAL_ENV"
echo "- PYTHONPATH: $PYTHONPATH"

# Step 7: If tests failed, create a debug report
if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "📋 Creating debug report..."
    cat > persistence_debug_report.txt << EOF
AURA Persistence Debug Report
Generated: $(date)
=============================

Exit Code: $TEST_EXIT_CODE

Last 50 lines of output:
------------------------
$(tail -50 persistence_test_output.log)

Python Environment:
------------------
$(pip list | grep -E "asyncpg|duckdb|zstandard|aiokafka|langgraph|torch|numpy")

System Info:
-----------
$(uname -a)
$(python3 --version)

PYTHONPATH: $PYTHONPATH
PWD: $(pwd)
EOF
    echo "Debug report saved to: persistence_debug_report.txt"
fi

exit $TEST_EXIT_CODE