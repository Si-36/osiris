#!/bin/bash

echo "🚀 Starting AURA Real System..."

# Check if Redis is running
if ! command -v redis-cli &> /dev/null || ! redis-cli ping &> /dev/null; then
    echo "⚠️  Redis not running. Starting Redis in Docker..."
    docker run -d --name aura-redis -p 6379:6379 redis:7-alpine
    sleep 2
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Start collector in background
echo "🔄 Starting Metric Collector..."
python core/collector.py &
COLLECTOR_PID=$!

# Start API server in background
echo "🌐 Starting API Server..."
python api/main.py &
API_PID=$!

# Wait a bit for services to start
sleep 3

# Run the dashboard
echo "📊 Starting Dashboard..."
python demo/terminal_dashboard.py

# Cleanup on exit
echo "🧹 Cleaning up..."
kill $COLLECTOR_PID $API_PID 2>/dev/null
docker stop aura-redis 2>/dev/null
docker rm aura-redis 2>/dev/null