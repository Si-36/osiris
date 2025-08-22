#!/bin/bash

# AURA Intelligence System Startup Script
# This script starts all infrastructure components and the AURA system

set -e

echo "🚀 Starting AURA Intelligence System"
echo "=================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service
wait_for_service() {
    local service_name=$1
    local host=$2
    local port=$3
    local max_attempts=30
    local attempt=1
    
    echo -n "⏳ Waiting for $service_name..."
    while ! nc -z $host $port 2>/dev/null; do
        if [ $attempt -eq $max_attempts ]; then
            echo -e "${RED}Failed${NC}"
            echo "❌ $service_name did not start in time"
            return 1
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo -e " ${GREEN}Ready${NC}"
    return 0
}

# Check requirements
echo "📋 Checking requirements..."

if ! command_exists docker; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command_exists docker-compose; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo -e "${GREEN}✅ All requirements met${NC}"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models cache/tda logs data
mkdir -p infrastructure/prometheus infrastructure/grafana

# Start infrastructure
echo "🐳 Starting infrastructure services..."
cd infrastructure
docker-compose up -d

cd ..

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
wait_for_service "Neo4j" localhost 7687
wait_for_service "Redis" localhost 6379
wait_for_service "PostgreSQL" localhost 5432
wait_for_service "Kafka" localhost 19092
wait_for_service "Prometheus" localhost 9090
wait_for_service "Grafana" localhost 3000

echo -e "${GREEN}✅ All infrastructure services are running${NC}"

# Check Python dependencies
echo "📦 Checking Python dependencies..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Initialize databases
echo "🗄️ Initializing databases..."

# Neo4j initialization
python3 -c "
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
auth = ('neo4j', os.getenv('NEO4J_PASSWORD', 'aura_password'))

try:
    driver = GraphDatabase.driver(uri, auth=auth)
    with driver.session() as session:
        # Create indexes
        session.run('CREATE INDEX agent_id IF NOT EXISTS FOR (n:Agent) ON (n.id)')
        session.run('CREATE INDEX component_id IF NOT EXISTS FOR (n:Component) ON (n.id)')
        session.run('CREATE INDEX memory_key IF NOT EXISTS FOR (n:Memory) ON (n.key)')
    driver.close()
    print('✅ Neo4j initialized')
except Exception as e:
    print(f'⚠️  Neo4j initialization failed: {e}')
"

# Redis initialization
python3 -c "
import redis
import os
from dotenv import load_dotenv

load_dotenv()

r = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    password=os.getenv('REDIS_PASSWORD', None),
    decode_responses=True
)

try:
    r.ping()
    r.set('aura:initialized', 'true')
    print('✅ Redis initialized')
except Exception as e:
    print(f'⚠️  Redis initialization failed: {e}')
"

# PostgreSQL initialization
python3 -c "
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

try:
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', 5432),
        database=os.getenv('POSTGRES_DB', 'aura_db'),
        user=os.getenv('POSTGRES_USER', 'aura_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'aura_password')
    )
    
    cur = conn.cursor()
    
    # Create tables
    cur.execute('''
        CREATE TABLE IF NOT EXISTS aura_events (
            id SERIAL PRIMARY KEY,
            event_type VARCHAR(100),
            event_data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS aura_metrics (
            id SERIAL PRIMARY KEY,
            metric_name VARCHAR(100),
            metric_value FLOAT,
            tags JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    cur.close()
    conn.close()
    print('✅ PostgreSQL initialized')
except Exception as e:
    print(f'⚠️  PostgreSQL initialization failed: {e}')
"

echo -e "${GREEN}✅ Databases initialized${NC}"

# Start AURA services
echo "🧠 Starting AURA services..."

# Kill any existing AURA processes
pkill -f "aura_working_demo" || true
pkill -f "uvicorn" || true

# Start the main demo
echo "Starting AURA Working Demo..."
python3 demos/aura_working_demo_2025.py &
DEMO_PID=$!

sleep 5

# Check if demo is running
if ps -p $DEMO_PID > /dev/null; then
    echo -e "${GREEN}✅ AURA Working Demo started (PID: $DEMO_PID)${NC}"
else
    echo -e "${RED}❌ Failed to start AURA Working Demo${NC}"
fi

# Display status
echo ""
echo "📊 AURA Intelligence System Status"
echo "=================================="
echo -e "🌐 Web Interface:     ${GREEN}http://localhost:8080${NC}"
echo -e "📊 Grafana Dashboard: ${GREEN}http://localhost:3000${NC} (admin/aura_admin)"
echo -e "🔍 Neo4j Browser:     ${GREEN}http://localhost:7474${NC} (neo4j/aura_password)"
echo -e "📈 Redis Insight:     ${GREEN}http://localhost:8001${NC}"
echo -e "🔭 Jaeger UI:        ${GREEN}http://localhost:16686${NC}"
echo -e "📊 Prometheus:        ${GREEN}http://localhost:9090${NC}"
echo ""
echo "To stop all services, run: ./stop_aura_system.sh"
echo ""

# Create stop script
cat > stop_aura_system.sh << 'EOF'
#!/bin/bash

echo "🛑 Stopping AURA Intelligence System..."

# Stop Python processes
pkill -f "aura_working_demo" || true
pkill -f "uvicorn" || true

# Stop Docker services
cd infrastructure
docker-compose down

echo "✅ All services stopped"
EOF

chmod +x stop_aura_system.sh

echo -e "${GREEN}✨ AURA Intelligence System is ready!${NC}"