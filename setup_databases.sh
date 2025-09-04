#!/bin/bash
# Setup script for Redis and Neo4j using Docker

echo "🚀 Setting up Redis and Neo4j for AURA Memory System"
echo "====================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Start Redis
echo ""
echo "1️⃣ Starting Redis..."
docker run -d \
    --name aura-redis \
    -p 6379:6379 \
    redis:7-alpine \
    2>/dev/null || echo "   Redis container already exists"

# Wait for Redis to be ready
echo "   Waiting for Redis..."
sleep 2
docker exec aura-redis redis-cli ping > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ Redis is running on localhost:6379"
else
    echo "   ⚠️  Redis may not be ready yet. Try again in a few seconds."
fi

# Start Neo4j
echo ""
echo "2️⃣ Starting Neo4j..."
docker run -d \
    --name aura-neo4j \
    -p 7474:7474 \
    -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_PLUGINS='["apoc"]' \
    neo4j:5-community \
    2>/dev/null || echo "   Neo4j container already exists"

echo "   Waiting for Neo4j (this may take 30-60 seconds)..."
sleep 10

# Check Neo4j status
for i in {1..30}; do
    if curl -s http://localhost:7474 > /dev/null 2>&1; then
        echo "   ✅ Neo4j is running on localhost:7687 (browser: http://localhost:7474)"
        echo "      Username: neo4j"
        echo "      Password: password"
        break
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "====================================================="
echo "📊 Container Status:"
docker ps --filter "name=aura-redis" --filter "name=aura-neo4j" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🧪 To test the connections:"
echo "   Redis:  docker exec aura-redis redis-cli ping"
echo "   Neo4j:  curl http://localhost:7474"

echo ""
echo "🛑 To stop the containers:"
echo "   docker stop aura-redis aura-neo4j"

echo ""
echo "🗑️  To remove the containers:"
echo "   docker rm -f aura-redis aura-neo4j"

echo ""
echo "✅ Setup complete! You can now run the memory tests."