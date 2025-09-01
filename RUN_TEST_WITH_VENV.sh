#!/bin/bash
# Run AURA test with the correct virtual environment

echo "🚀 AURA Import Test with Virtual Environment"
echo "==========================================="
echo ""

# Check if we're already in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Already in virtual environment: $VIRTUAL_ENV"
    echo "   Python: $(which python3)"
else
    echo "⚠️  Not in virtual environment!"
    echo "   You need to activate your environment first:"
    echo ""
    echo "   source /home/sina/projects/osiris-2/aura_venv/bin/activate"
    echo ""
    echo "   Then run this script again."
    exit 1
fi

# Check dependencies
echo ""
echo "Checking dependencies..."
python3 -c "import aiokafka; print('✅ aiokafka:', aiokafka.__version__)" 2>/dev/null || echo "❌ aiokafka not found"
python3 -c "import langgraph; print('✅ langgraph available')" 2>/dev/null || echo "❌ langgraph not found"
python3 -c "import asyncpg; print('✅ asyncpg:', asyncpg.__version__)" 2>/dev/null || echo "❌ asyncpg not found"
python3 -c "import temporalio; print('✅ temporalio available')" 2>/dev/null || echo "❌ temporalio not found"

echo ""
echo "Running import test..."
echo ""

# Run the test
python3 TEST_AURA_STEP_BY_STEP.py