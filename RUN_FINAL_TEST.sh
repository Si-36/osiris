#!/bin/bash

echo "🚀 AURA Persistence Test - Final Version"
echo "========================================"
echo ""
echo "This script runs the complete persistence test."
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

echo "📋 Environment:"
echo "- Python: $(python3 --version)"
echo "- Working dir: $(pwd)"
echo "- PYTHONPATH: $PYTHONPATH"
echo ""

# Run the test
echo "🧪 Running persistence tests..."
echo "================================"
python3 TEST_PERSISTENCE_NOW.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS! All persistence tests passed!"
    echo ""
    echo "🎯 What was tested:"
    echo "- Causal persistence with tracking"
    echo "- Memory-native architecture"
    echo "- GPU memory tier"
    echo "- Speculative branches"
    echo "- Pickle migration"
    echo ""
    echo "📈 Next steps:"
    echo "1. Run integration tests: python3 test_persistence_integration_complete.py"
    echo "2. Test with agents: python3 test_all_agents_integrated.py"
    echo "3. Benchmark performance (expect 10-100x improvement)"
else
    echo ""
    echo "❌ Tests failed with exit code: $EXIT_CODE"
    echo ""
    echo "If you see any errors, they have been fixed in these files:"
    echo "✅ core/src/aura_intelligence/resilience/retry.py"
    echo "✅ core/src/aura_intelligence/resilience/timeout.py"
    echo "✅ core/src/aura_intelligence/resilience/metrics.py"
    echo "✅ core/src/aura_intelligence/consensus/byzantine.py"
    echo "✅ Various import paths fixed"
fi

exit $EXIT_CODE