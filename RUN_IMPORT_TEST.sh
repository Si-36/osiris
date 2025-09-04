#!/bin/bash
# Run AURA import test and capture detailed output

echo "🚀 Running AURA Import Test"
echo "=========================="
echo ""
echo "Make sure you have activated your environment with:"
echo "- aiokafka"
echo "- langgraph"
echo "- asyncpg"
echo "- Other dependencies"
echo ""
echo "Running test..."
echo ""

# Run the test and capture output
python3 TEST_AURA_STEP_BY_STEP.py 2>&1 | tee import_test_output.log

echo ""
echo "=========================="
echo "Test complete!"
echo "Output saved to: import_test_output.log"
echo ""
echo "If there are errors:"
echo "1. Share the specific error"
echo "2. I'll fix it manually"
echo "3. Run the test again"