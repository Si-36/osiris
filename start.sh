#!/bin/bash

echo "🚀 Starting AURA Intelligence Production System 2025"

# Start Redis
redis-server --daemonize yes --port 6379

# Wait for Redis
sleep 2

# Start AURA system
echo "🧬 Starting enhanced AURA systems..."
python3 production_aura_2025.py &

# Wait for system to start
sleep 5

# Run comprehensive tests
echo "🧪 Running comprehensive tests..."
python3 -m pytest test_enhanced_systems.py -v

# Keep container running
wait