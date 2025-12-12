#!/bin/bash

# Test reproducibility by building twice and comparing

set -e

echo "=== Testing Docker Image Reproducibility ==="
echo ""

# Clean up any previous artifacts
rm -f oci1.tar oci2.tar digest1.txt digest2.txt

echo "Building first image..."
./build-image.sh
if [ ! -f oci.tar ]; then
    echo "Error: First build failed to create oci.tar"
    exit 1
fi
mv oci.tar oci1.tar
skopeo inspect oci-archive:./oci1.tar | jq -r .Digest > digest1.txt
echo "First digest: $(cat digest1.txt)"
echo ""

echo "Sleeping 2 seconds..."
sleep 2
echo ""

echo "Building second image..."
./build-image.sh
if [ ! -f oci.tar ]; then
    echo "Error: Second build failed to create oci.tar"
    exit 1
fi
mv oci.tar oci2.tar
skopeo inspect oci-archive:./oci2.tar | jq -r .Digest > digest2.txt
echo "Second digest: $(cat digest2.txt)"
echo ""

# Compare
if diff digest1.txt digest2.txt > /dev/null; then
    echo "✅ SUCCESS: Digests are identical!"
    echo "Digest: $(cat digest1.txt)"
else
    echo "❌ FAILURE: Digests differ!"
    echo ""
    echo "Digest 1: $(cat digest1.txt)"
    echo "Digest 2: $(cat digest2.txt)"
    echo ""
    
    # Detailed comparison
    echo "=== Detailed Comparison ==="
    echo ""
    
    echo "Manifest 1:"
    skopeo inspect oci-archive:./oci1.tar | jq '{Digest, Architecture, Layers, Created}' > manifest1.json
    cat manifest1.json
    echo ""
    
    echo "Manifest 2:"
    skopeo inspect oci-archive:./oci2.tar | jq '{Digest, Architecture, Layers, Created}' > manifest2.json
    cat manifest2.json
    echo ""
    
    echo "Layer comparison:"
    echo "Build 1 layers:"
    skopeo inspect oci-archive:./oci1.tar | jq -r '.Layers[]'
    echo ""
    echo "Build 2 layers:"
    skopeo inspect oci-archive:./oci2.tar | jq -r '.Layers[]'
    echo ""
    
    # Check configs
    echo "Checking image configs..."
    skopeo inspect --config oci-archive:./oci1.tar > config1.json
    skopeo inspect --config oci-archive:./oci2.tar > config2.json
    
    echo "Config diff:"
    diff -u config1.json config2.json || true
    echo ""
    
    # Extract and compare timestamps in blobs
    echo "Extracting archives for detailed inspection..."
    rm -rf /tmp/oci1 /tmp/oci2
    mkdir -p /tmp/oci1 /tmp/oci2
    tar -xf oci1.tar -C /tmp/oci1
    tar -xf oci2.tar -C /tmp/oci2
    
    echo "Comparing blob directories..."
    ls -lah /tmp/oci1/blobs/sha256/ > /tmp/blobs1.txt
    ls -lah /tmp/oci2/blobs/sha256/ > /tmp/blobs2.txt
    diff -u /tmp/blobs1.txt /tmp/blobs2.txt || true
    
    exit 1
fi

# Clean up
rm -f oci1.tar oci2.tar digest1.txt digest2.txt

echo ""
echo "Test completed successfully!"
