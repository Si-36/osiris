#!/bin/bash

echo "🚀 Starting REAL AURA Infrastructure 2025"
echo "=========================================="

# Create monitoring directory if it doesn't exist
mkdir -p monitoring

# Stop any existing containers
echo "🔄 Stopping existing containers..."
docker compose -f docker-compose.real.yml down

# Start all services
echo "🏗️  Starting all services..."
docker compose -f docker-compose.real.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🏥 Checking service health..."

# Check Neo4j
echo "📊 Neo4j status:"
curl -s http://localhost:7474/db/system/tx/commit \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic bmVvNGo6YXVyYXBhc3N3b3Jk" \
  -d '{"statements":[{"statement":"RETURN 1 as test"}]}' | jq '.results[0].data[0].row[0]' 2>/dev/null || echo "Neo4j not ready yet"

# Check Redis
echo "📦 Redis status:"
redis-cli -h localhost -p 6379 ping 2>/dev/null || echo "Redis not ready yet"

# Check Mem0
echo "🧠 Mem0 status:"
curl -s http://localhost:8080/health 2>/dev/null || echo "Mem0 not ready yet"

# Check Prometheus
echo "📈 Prometheus status:"
curl -s http://localhost:9090/-/healthy 2>/dev/null || echo "Prometheus not ready yet"

# Check Grafana
echo "📊 Grafana status:"
curl -s http://localhost:3000/api/health 2>/dev/null || echo "Grafana not ready yet"

echo ""
echo "✅ Infrastructure started! Services available at:"
echo "   🌐 Neo4j Browser: http://localhost:7474 (neo4j/aurapassword)"
echo "   📦 Redis: localhost:6379"
echo "   🧠 Mem0 API: http://localhost:8080"
echo "   📈 Prometheus: http://localhost:9090"
echo "   📊 Grafana: http://localhost:3000 (admin/auraadmin)"
echo "   🔍 Jaeger: http://localhost:16686"
echo ""
echo "🔧 To stop: docker compose -f docker-compose.real.yml down"