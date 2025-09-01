#!/bin/bash
# Script to run tests in your local environment with virtual env

echo "🧪 Running AURA tests in local environment"
echo "=========================================="

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Not in a virtual environment!"
    echo "Please activate your virtual environment first:"
    echo "  source ~/projects/osiris-2/aura_venv/bin/activate"
    exit 1
fi

echo "✅ Using Python: $(which python3)"
echo "✅ Virtual env: $VIRTUAL_ENV"
echo ""

# Check dependencies
echo "📦 Checking dependencies..."
python3 -c "import msgpack; print('✅ msgpack:', msgpack.version)"
python3 -c "import asyncpg; print('✅ asyncpg:', asyncpg.__version__)"
python3 -c "import aiokafka; print('✅ aiokafka:', aiokafka.__version__)"

echo ""
echo "🧪 Running import tests..."
echo ""

# Run the main test
python3 TEST_AURA_STEP_BY_STEP.py

echo ""
echo "💡 If you see import errors, try:"
echo "  pip install msgpack asyncpg aiokafka langgraph torch"