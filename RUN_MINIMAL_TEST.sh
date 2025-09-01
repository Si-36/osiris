#!/bin/bash

echo "🚀 AURA Persistence - Minimal Test"
echo "=================================="
echo ""

# Go to project directory
cd ~/projects/osiris-2 || exit 1

# Activate virtual environment
source aura_venv/bin/activate || {
    echo "❌ Failed to activate virtual environment"
    exit 1
}

# Set PYTHONPATH
export PYTHONPATH="${PWD}/core/src:${PYTHONPATH}"

# Run the minimal test
echo "Running minimal persistence test..."
echo ""

python3 TEST_PERSISTENCE_MINIMAL.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Minimal persistence test passed!"
    echo ""
    echo "The persistence system is working at a basic level."
    echo "To run full tests, we need to fix the remaining import errors in:"
    echo "- core/src/aura_intelligence/resilience/timeout.py"
    echo "- Any other files with indentation issues"
else
    echo ""
    echo "❌ Minimal test failed"
fi

exit $EXIT_CODE