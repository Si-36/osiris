#!/bin/bash

echo "🚀 AURA Intelligence 2025 - Complete Deployment & Testing"
echo "=========================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build and start all services
echo "🐳 Building and starting all services..."
docker-compose down --remove-orphans
docker-compose build --no-cache
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🏥 Checking service health..."
docker-compose ps

# Wait for AURA system to be fully ready
echo "⏳ Waiting for AURA system to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:8098/health > /dev/null; then
        echo "✅ AURA system is ready!"
        break
    fi
    echo "   Attempt $i/30..."
    sleep 2
done

# Run end-to-end tests
echo "🧪 Running comprehensive end-to-end tests..."
python3 test_end_to_end.py

# Show access URLs
echo ""
echo "🌐 SYSTEM ACCESS URLS:"
echo "================================"
echo "🔗 AURA API:        http://localhost:8098"
echo "📊 Prometheus:      http://localhost:9090"
echo "📈 Grafana:         http://localhost:3000 (admin/admin)"
echo "🗄️  Redis:           localhost:6379"
echo ""

# Show quick test commands
echo "🧪 QUICK TEST COMMANDS:"
echo "================================"
echo "# System health"
echo "curl http://localhost:8098/health"
echo ""
echo "# Enhanced processing"
echo "curl -X POST http://localhost:8098/enhanced/process \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"council_task\": {\"gpu_allocation\": {\"gpu_count\": 4}}}'"
echo ""
echo "# Memory operations"
echo "curl -X POST http://localhost:8098/memory/hybrid \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"key\": \"test\", \"data\": {\"value\": 123}, \"component_id\": \"mem_001\", \"operation\": \"store\"}'"
echo ""
echo "# Prometheus metrics"
echo "curl http://localhost:8098/metrics"
echo ""
echo "# System benchmark"
echo "curl http://localhost:8098/benchmark"
echo ""

# Show logs command
echo "📋 VIEW LOGS:"
echo "================================"
echo "docker-compose logs -f aura-system"
echo ""

# Keep services running
echo "🎉 Deployment complete! Services are running."
echo "Press Ctrl+C to stop all services."

# Wait for user interrupt
trap 'echo "🛑 Stopping services..."; docker-compose down; exit 0' INT
while true; do
    sleep 1
done