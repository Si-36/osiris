#!/bin/bash
# Complete setup and test script for AURA Persistence
# Run this in your local environment where dependencies are installed

echo "🚀 AURA PERSISTENCE - COMPLETE SETUP & TEST"
echo "=========================================="

# Step 1: Ensure we're in the right directory
echo "📍 Step 1: Checking directory..."
if [ ! -d "core/src/aura_intelligence" ]; then
    echo "❌ Error: Not in the osiris-2 project directory"
    echo "Please run this from your project root"
    exit 1
fi
echo "✅ In correct directory"

# Step 2: Start Docker services
echo -e "\n📍 Step 2: Starting Docker services..."
if command -v docker &> /dev/null; then
    echo "Starting PostgreSQL and Redis..."
    docker compose -f docker-compose.persistence.yml up -d postgres redis
    echo "Waiting for services to start..."
    sleep 5
else
    echo "⚠️  Docker not found - will use legacy mode"
fi

# Step 3: Run the persistence tests
echo -e "\n📍 Step 3: Running persistence tests..."
python3 test_persistence_complete_debug.py

# Step 4: If tests pass, run integration
if [ $? -eq 0 ]; then
    echo -e "\n📍 Step 4: Running full integration test..."
    python3 test_all_agents_integrated.py
else
    echo -e "\n❌ Persistence tests failed - fix issues before integration"
fi

# Step 5: Show Docker logs if needed
echo -e "\n📍 Step 5: Docker service status..."
if command -v docker &> /dev/null; then
    docker compose -f docker-compose.persistence.yml ps
    echo -e "\nTo view logs: docker compose -f docker-compose.persistence.yml logs"
fi

echo -e "\n✅ Setup complete!"