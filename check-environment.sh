#!/bin/bash

# Check BuildKit and Docker environment for reproducibility issues

echo "=== Docker Build Environment Check ==="
echo ""

echo "1. Docker Version:"
docker version --format '{{.Server.Version}}' 2>/dev/null || echo "Unable to get Docker version"
echo ""

echo "2. Docker Buildx Version:"
docker buildx version
echo ""

echo "3. Available Builders:"
docker buildx ls
echo ""

echo "4. BuildKit_20 Builder Details:"
if docker buildx inspect buildkit_20 &>/dev/null; then
    docker buildx inspect buildkit_20
    echo ""
    echo "Builder Status: $(docker buildx inspect buildkit_20 2>/dev/null | grep 'Status:' | head -1 | awk '{print $2}')"
else
    echo "❌ buildkit_20 builder not found"
    echo "Run: docker buildx create --use --driver-opt image=moby/buildkit:v0.20.2 --name buildkit_20"
fi
echo ""

echo "5. System Information:"
echo "OS: $(uname -s)"
echo "Architecture: $(uname -m)"
echo "Kernel: $(uname -r)"
echo ""

echo "6. Testing BuildKit Reproducibility Feature:"
echo "Creating minimal test..."

# Create minimal test Dockerfile
cat > /tmp/test-repro.Dockerfile <<'EOF'
FROM alpine:3.19@sha256:c5b1261d6d3e43071626931fc004f70149baeba2c8ec672bd4f27761f8e1ad6b
ARG SOURCE_DATE_EPOCH=0
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}
RUN echo "test" > /test.txt && date > /date.txt
EOF

# Build twice
echo "Building test image 1..."
docker buildx build --builder buildkit_20 --no-cache \
    --build-arg SOURCE_DATE_EPOCH="0" \
    --output type=oci,dest=/tmp/test1.tar,rewrite-timestamp=true \
    -f /tmp/test-repro.Dockerfile \
    /tmp > /dev/null 2>&1

if [ ! -f /tmp/test1.tar ]; then
    echo "❌ First test build failed"
    exit 1
fi

echo "Building test image 2..."
sleep 1
docker buildx build --builder buildkit_20 --no-cache \
    --build-arg SOURCE_DATE_EPOCH="0" \
    --output type=oci,dest=/tmp/test2.tar,rewrite-timestamp=true \
    -f /tmp/test-repro.Dockerfile \
    /tmp > /dev/null 2>&1

if [ ! -f /tmp/test2.tar ]; then
    echo "❌ Second test build failed"
    exit 1
fi

# Compare
TEST_DIGEST1=$(skopeo inspect oci-archive:/tmp/test1.tar 2>/dev/null | jq -r .Digest)
TEST_DIGEST2=$(skopeo inspect oci-archive:/tmp/test2.tar 2>/dev/null | jq -r .Digest)

echo ""
echo "Test Build 1 Digest: $TEST_DIGEST1"
echo "Test Build 2 Digest: $TEST_DIGEST2"
echo ""

if [ "$TEST_DIGEST1" = "$TEST_DIGEST2" ]; then
    echo "✅ BuildKit reproducibility is WORKING correctly"
    echo "The issue is likely with the application Dockerfile or build context"
else
    echo "❌ BuildKit reproducibility is NOT WORKING"
    echo "This indicates a problem with:"
    echo "  - BuildKit version or configuration"
    echo "  - Docker Desktop version (Mac/Windows)"
    echo "  - System-level filesystem or Docker settings"
    echo ""
    echo "Recommendations:"
    echo "  1. Try updating Docker Desktop to latest version"
    echo "  2. Try recreating the buildkit_20 builder:"
    echo "     docker buildx rm buildkit_20"
    echo "     docker buildx create --use --driver-opt image=moby/buildkit:v0.20.2 --name buildkit_20"
    echo "  3. Test on a Linux machine if currently on Mac/Windows"
fi

echo ""
echo "7. Checking for Docker Desktop issues:"
if [ -f /Applications/Docker.app/Contents/Info.plist ]; then
    DOCKER_DESKTOP_VERSION=$(defaults read /Applications/Docker.app/Contents/Info.plist CFBundleShortVersionString 2>/dev/null || echo "unknown")
    echo "Docker Desktop Version: $DOCKER_DESKTOP_VERSION"
    echo "Note: Some Docker Desktop versions on Mac have reproducibility issues"
else
    echo "Not running Docker Desktop or not on Mac"
fi

echo ""
echo "=== Check Complete ==="

# Cleanup
rm -f /tmp/test1.tar /tmp/test2.tar /tmp/test-repro.Dockerfile
