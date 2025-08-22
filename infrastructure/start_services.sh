#!/bin/bash
# Start AURA services

echo "Starting AURA services..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Start services
echo "Starting infrastructure services..."
cd infrastructure
docker-compose up -d

echo "Waiting for services to be ready..."
sleep 10

echo "Services started. Check status with: docker-compose ps"
