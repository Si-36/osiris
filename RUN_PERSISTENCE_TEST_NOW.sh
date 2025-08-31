#!/bin/bash

# Quick test script for AURA Persistence System
# This runs the test immediately with proper environment setup

echo "🚀 AURA Persistence Test - Fixed Version"
echo "========================================"
echo ""

# Ensure we're in the right directory
cd ~/projects/osiris-2 || exit 1

# Activate virtual environment
source aura_venv/bin/activate || {
    echo "❌ Failed to activate virtual environment"
    exit 1
}

# Export PYTHONPATH
export PYTHONPATH="${PWD}/core/src:${PYTHONPATH}"

# Run the test
echo "Running persistence test..."
python3 TEST_PERSISTENCE_NOW.py

# Save exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Persistence tests passed!"
    echo ""
    echo "Next steps:"
    echo "1. Review the test output above"
    echo "2. Run integration tests: python3 test_persistence_integration_complete.py"
    echo "3. Test with agents: python3 test_all_agents_integrated.py"
else
    echo ""
    echo "❌ Test failed with exit code: $EXIT_CODE"
    echo ""
    echo "If you see any errors, please share them for debugging."
fi

exit $EXIT_CODE