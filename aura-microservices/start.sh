#!/bin/bash

# AURA Microservices Quick Start Script

echo "🚀 Starting AURA Intelligence Microservices..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p monitoring/prometheus
mkdir -p monitoring/grafana/dashboards
mkdir -p nginx

# Create basic Prometheus config
cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'aura-lnn'
    static_configs:
      - targets: ['aura-lnn:8001']
  
  - job_name: 'aura-tda'
    static_configs:
      - targets: ['aura-tda:8002']
  
  - job_name: 'aura-consensus'
    static_configs:
      - targets: ['aura-consensus:8003']
  
  - job_name: 'aura-neuromorphic'
    static_configs:
      - targets: ['aura-neuromorphic:8004']
  
  - job_name: 'aura-memory'
    static_configs:
      - targets: ['aura-memory:8005']
EOF

# Create basic nginx config
cat > nginx/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream lnn {
        server aura-lnn:8001;
    }
    
    upstream tda {
        server aura-tda:8002;
    }
    
    upstream consensus {
        server aura-consensus:8003;
    }
    
    upstream neuromorphic {
        server aura-neuromorphic:8004;
    }
    
    upstream memory {
        server aura-memory:8005;
    }
    
    server {
        listen 80;
        
        location /lnn/ {
            proxy_pass http://lnn/;
        }
        
        location /tda/ {
            proxy_pass http://tda/;
        }
        
        location /consensus/ {
            proxy_pass http://consensus/;
        }
        
        location /neuromorphic/ {
            proxy_pass http://neuromorphic/;
        }
        
        location /memory/ {
            proxy_pass http://memory/;
        }
        
        location /health {
            return 200 "healthy\n";
        }
    }
}
EOF

# Build services
echo "🔨 Building microservices..."
docker-compose build

# Start infrastructure first
echo "🏗️ Starting infrastructure services..."
docker-compose up -d neo4j kafka redis postgres

# Wait for infrastructure
echo "⏳ Waiting for infrastructure to be ready..."
sleep 30

# Start AURA services
echo "🚀 Starting AURA microservices..."
docker-compose up -d

# Show status
echo "✅ All services started!"
echo ""
echo "📊 Service URLs:"
echo "  - LNN Service: http://localhost:8001"
echo "  - TDA Engine: http://localhost:8002"
echo "  - Consensus Service: http://localhost:8003"
echo "  - Neuromorphic Service: http://localhost:8004"
echo "  - Memory Service: http://localhost:8005"
echo "  - Neo4j Browser: http://localhost:7474"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - API Gateway: http://localhost:80"
echo ""
echo "Use 'docker-compose logs -f' to view logs"
echo "Use 'docker-compose down' to stop all services"