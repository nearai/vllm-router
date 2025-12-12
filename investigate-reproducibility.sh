#!/bin/bash

# Script to investigate Docker image reproducibility issues

echo "=== Docker Image Reproducibility Investigation ==="
echo ""

# 1. Check architecture
echo "1. System Architecture:"
uname -m
echo ""

# 2. Check Docker buildx builder info
echo "2. Docker Buildx Builder Info:"
docker buildx inspect buildkit_20 2>/dev/null || echo "Builder 'buildkit_20' not found"
echo ""

# 3. Check BuildKit version
echo "3. BuildKit Version:"
docker buildx inspect buildkit_20 --bootstrap 2>/dev/null | grep "Driver Options:"
echo ""

# 4. Build image and inspect layers
echo "4. Building image..."
./build-image.sh

if [ -f oci.tar ]; then
    echo ""
    echo "5. OCI Archive Details:"
    echo "Manifest:"
    skopeo inspect oci-archive:./oci.tar | jq '{Digest, Architecture, Os, Layers, Created}'
    
    echo ""
    echo "6. Layer Digests:"
    skopeo inspect oci-archive:./oci.tar | jq -r '.Layers[]'
    
    echo ""
    echo "7. Full Config (for detailed inspection):"
    skopeo inspect --config oci-archive:./oci.tar > /tmp/oci-config.json
    echo "Config saved to /tmp/oci-config.json"
    echo ""
    
    # Extract and inspect the tar
    echo "8. Extracting OCI tar for inspection..."
    rm -rf /tmp/oci-extract
    mkdir -p /tmp/oci-extract
    tar -xf oci.tar -C /tmp/oci-extract
    
    echo "Contents of OCI tar:"
    ls -lah /tmp/oci-extract/
    echo ""
    
    echo "9. Checking for timestamps in blobs..."
    find /tmp/oci-extract/blobs -type f | head -3 | while read blob; do
        echo "Blob: $blob"
        ls -l "$blob"
    done
else
    echo "No oci.tar found. Build may have failed."
fi

echo ""
echo "=== Investigation Complete ==="
echo ""
echo "To compare with another machine:"
echo "1. Run this script on both machines"
echo "2. Compare the Layer Digests section"
echo "3. Compare the Architecture"
echo "4. Share /tmp/oci-config.json from both machines"
