#!/bin/bash

# MedBioRAG Deployment Script

IMAGE_NAME="medbiorag-api"
PORT=8000

echo "--- Building Docker Image: $IMAGE_NAME ---"
docker build -t $IMAGE_NAME .

echo "--- Stopping existing containers ---"
docker stop $IMAGE_NAME 2>/dev/null || true
docker rm $IMAGE_NAME 2>/dev/null || true

echo "--- Starting MedBioRAG API on port $PORT ---"
# Note: storage folder is mounted as a volume for persistence
docker run -d \
  --name $IMAGE_NAME \
  -p $PORT:8000 \
  -v $(pwd)/storage:/app/storage \
  -e GITHUB_TOKEN=$GITHUB_TOKEN \
  $IMAGE_NAME

echo "--- Deployment Complete ---"
echo "API is running at http://localhost:$PORT"
echo "Health check: http://localhost:$PORT/health"
echo "API Docs: http://localhost:$PORT/docs"
