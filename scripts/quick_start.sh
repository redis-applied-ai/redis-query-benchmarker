#!/bin/bash

# Redis Query Benchmarker - Quick Start Script
echo "Redis Query Benchmarker - Quick Start"
echo "====================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is required but not installed."
    echo "Please install Docker and try again."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is required but not installed."
    echo "Please install Docker Compose and try again."
    exit 1
fi

echo "✓ Docker and Docker Compose found"

# Start Redis container
echo "Starting Redis container..."
docker-compose up -d

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
sleep 10

# Check if Redis is responding
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✓ Redis is ready"
else
    echo "Error: Redis is not responding"
    echo "Check the Docker logs: docker-compose logs redis"
    exit 1
fi

# Check if Python environment is set up
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

echo "✓ Python 3 found"

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

# Generate sample data
echo "Generating sample data..."
python -m redis_benchmarker.data_generator \
    --documents 10000 \
    --vector-dim 1536 \
    --index-name quickstart_index \
    --create-index \
    --force

# Run a quick benchmark
echo "Running quick benchmark test..."
python -m redis_benchmarker \
    --total-requests 100 \
    --workers 8 \
    --query-type vector_search \
    --index-name quickstart_index \
    --vector-field embedding

echo ""
echo "🎉 Quick start completed successfully!"
echo ""
echo "Next steps:"
echo "1. Try different configurations: python -m redis_benchmarker --help"
echo "2. Run custom queries: python examples/custom_executor_example.py"
echo "3. Performance comparison: python examples/performance_comparison.py"
echo "4. View Redis data: http://localhost:8001 (RedisInsight)"
echo ""
echo "To stop Redis: docker-compose down"