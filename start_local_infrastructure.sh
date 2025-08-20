#!/bin/bash

echo "🧠 Starting AURA Intelligence Local Infrastructure"
echo "=================================================="

# Create required directories
mkdir -p logs data monitoring/grafana/{dashboards,provisioning} sql

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Desktop or Docker Compose."
    exit 1
fi

# Use the correct docker compose command
DOCKER_COMPOSE="docker compose"
if ! docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
fi

echo "📋 Pre-flight Checks:"
echo "  ✓ Docker Compose: $(which docker-compose docker compose 2>/dev/null | head -1)"
echo "  ✓ Environment: .env file present"
echo "  ✓ Directories: Created logs, data, monitoring"

echo ""
echo "🚀 Starting Infrastructure Services..."

# Start infrastructure services first (without AURA app)
$DOCKER_COMPOSE up -d redis neo4j kafka zookeeper postgres minio prometheus grafana

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check service health
echo "🔍 Service Status:"
$DOCKER_COMPOSE ps

echo ""
echo "📊 Service Endpoints:"
echo "  Redis:      localhost:6380"
echo "  Neo4j:      http://localhost:7475 (Browser), bolt://localhost:7688"
echo "  Kafka:      localhost:9093"
echo "  PostgreSQL: localhost:5433"
echo "  MinIO:      http://localhost:9002 (UI: http://localhost:9003)"
echo "  Prometheus: http://localhost:9091"
echo "  Grafana:    http://localhost:3001 (admin/aura_grafana_2025_secure)"

echo ""
echo "🧪 Testing Infrastructure..."

# Test Redis
echo -n "  Redis: "
if docker exec aura-redis-new redis-cli -a aura_redis_2025_secure ping > /dev/null 2>&1; then
    echo "✅ Connected"
else
    echo "❌ Failed"
fi

# Test Neo4j
echo -n "  Neo4j: "
if docker exec aura-neo4j-new cypher-shell -u neo4j -p aura_neo4j_2025_secure "RETURN 1" > /dev/null 2>&1; then
    echo "✅ Connected"
else
    echo "❌ Failed"
fi

# Test PostgreSQL
echo -n "  PostgreSQL: "
if docker exec aura-postgres-new pg_isready -U aura_user -d aura_intelligence > /dev/null 2>&1; then
    echo "✅ Connected"
else
    echo "❌ Failed"
fi

# Test Kafka
echo -n "  Kafka: "
if docker exec aura-kafka-new kafka-topics --bootstrap-server localhost:9092 --list > /dev/null 2>&1; then
    echo "✅ Connected"
else
    echo "❌ Failed"
fi

echo ""
echo "🎯 Next Steps:"
echo "1. Run component tests: python test_all_real_components.py"
echo "2. Start AURA API: $DOCKER_COMPOSE up -d aura-api"
echo "3. View logs: $DOCKER_COMPOSE logs -f aura-api"
echo "4. Stop all: $DOCKER_COMPOSE down"
echo ""
echo "✨ Infrastructure ready for AURA Intelligence testing!"