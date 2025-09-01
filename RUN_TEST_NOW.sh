#!/bin/bash

echo "🚀 Running AURA Persistence Test"
echo "================================"
echo ""

cd ~/projects/osiris-2 || exit 1
source aura_venv/bin/activate || exit 1
export PYTHONPATH="${PWD}/core/src:${PYTHONPATH}"

echo "Running persistence test..."
python3 TEST_PERSISTENCE_NOW.py

exit $?