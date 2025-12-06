#!/bin/bash
# Build and Push Docker Image to Registry
# For RunPod deployment, you can use Docker Hub or any container registry
#
# Usage:
#   ./scripts/build_and_push.sh <registry>/<image-name>:<tag>
#
# Example:
#   ./scripts/build_and_push.sh myusername/medgemma-api:latest

set -e

# Default image name
IMAGE_NAME="${1:-medgemma-api:latest}"

echo "=========================================="
echo "Building MedGemma Docker Image"
echo "Image: $IMAGE_NAME"
echo "=========================================="

# Build the Docker image
echo "Building image..."
docker build -t "$IMAGE_NAME" .

# Show image size
echo ""
echo "Image size:"
docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# Push to registry if specified
if [[ "$IMAGE_NAME" == *"/"* ]]; then
    echo ""
    echo "Pushing to registry..."
    docker push "$IMAGE_NAME"
    echo "Image pushed successfully!"
else
    echo ""
    echo "To push to a registry, use:"
    echo "  docker tag $IMAGE_NAME <registry>/<image-name>:<tag>"
    echo "  docker push <registry>/<image-name>:<tag>"
fi

echo ""
echo "=========================================="
echo "Build complete!"
echo ""
echo "To test locally:"
echo "  docker run --gpus all -p 8000:8000 --env-file .env $IMAGE_NAME"
echo ""
echo "For RunPod deployment:"
echo "1. Push image to Docker Hub or another registry"
echo "2. Create a new Pod on RunPod"
echo "3. Select A6000 GPU (48GB VRAM)"
echo "4. Set image name to: $IMAGE_NAME"
echo "5. Set environment variables (HF_TOKEN, API_KEY)"
echo "6. Expose port 8000"
echo "=========================================="
