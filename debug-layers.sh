#!/bin/bash

# Debug script to identify which layer changes between builds

set -e

echo "=== Layer-by-Layer Reproducibility Test ==="
echo ""

# Build 1
echo "Building first image..."
./build-image.sh > /tmp/build1.log 2>&1
mv oci.tar oci1.tar

# Build 2  
echo "Building second image..."
sleep 1
./build-image.sh > /tmp/build2.log 2>&1
mv oci.tar oci2.tar

echo ""
echo "Extracting layer information..."

# Extract to temporary directories
rm -rf /tmp/oci1 /tmp/oci2
mkdir -p /tmp/oci1 /tmp/oci2
tar -xf oci1.tar -C /tmp/oci1
tar -xf oci2.tar -C /tmp/oci2

echo ""
echo "=== Manifest Comparison ==="
skopeo inspect oci-archive:./oci1.tar > /tmp/manifest1.json
skopeo inspect oci-archive:./oci2.tar > /tmp/manifest2.json

DIGEST1=$(jq -r .Digest /tmp/manifest1.json)
DIGEST2=$(jq -r .Digest /tmp/manifest2.json)

echo "Manifest Digest 1: $DIGEST1"
echo "Manifest Digest 2: $DIGEST2"

if [ "$DIGEST1" = "$DIGEST2" ]; then
    echo "✅ Manifests are identical - builds are reproducible!"
    rm -f oci1.tar oci2.tar
    exit 0
fi

echo ""
echo "❌ Manifests differ - investigating..."
echo ""

# Compare layers
echo "=== Layer Digests ==="
echo "Build 1 layers:"
jq -r '.Layers[]' /tmp/manifest1.json | nl
echo ""
echo "Build 2 layers:"
jq -r '.Layers[]' /tmp/manifest2.json | nl
echo ""

# Find differing layers
jq -r '.Layers[]' /tmp/manifest1.json > /tmp/layers1.txt
jq -r '.Layers[]' /tmp/manifest2.json > /tmp/layers2.txt

echo "=== Layer Differences ==="
diff -y /tmp/layers1.txt /tmp/layers2.txt || true
echo ""

# Count differing layers
DIFF_COUNT=$(diff /tmp/layers1.txt /tmp/layers2.txt | grep -c "^<\|^>" || true)
echo "Number of differing layer lines: $DIFF_COUNT"
echo ""

# Check created timestamps
echo "=== Created Timestamps ==="
echo "Build 1: $(jq -r .Created /tmp/manifest1.json)"
echo "Build 2: $(jq -r .Created /tmp/manifest2.json)"
echo ""

# Compare configs
echo "=== Config Comparison ==="
skopeo inspect --config oci-archive:./oci1.tar > /tmp/config1.json
skopeo inspect --config oci-archive:./oci2.tar > /tmp/config2.json

if diff /tmp/config1.json /tmp/config2.json > /dev/null; then
    echo "Configs are identical"
else
    echo "Configs differ:"
    diff -u /tmp/config1.json /tmp/config2.json | head -50
fi
echo ""

# Check blob counts
echo "=== Blob Comparison ==="
BLOB_COUNT1=$(ls /tmp/oci1/blobs/sha256/ | wc -l)
BLOB_COUNT2=$(ls /tmp/oci2/blobs/sha256/ | wc -l)
echo "Build 1 blob count: $BLOB_COUNT1"
echo "Build 2 blob count: $BLOB_COUNT2"
echo ""

# List blobs that differ
echo "Blobs only in build 1:"
comm -23 <(ls /tmp/oci1/blobs/sha256/ | sort) <(ls /tmp/oci2/blobs/sha256/ | sort) | head -10
echo ""
echo "Blobs only in build 2:"
comm -13 <(ls /tmp/oci1/blobs/sha256/ | sort) <(ls /tmp/oci2/blobs/sha256/ | sort) | head -10
echo ""

# Check index.json
echo "=== Index Comparison ==="
if diff /tmp/oci1/index.json /tmp/oci2/index.json > /dev/null; then
    echo "index.json files are identical"
else
    echo "index.json files differ:"
    diff -u /tmp/oci1/index.json /tmp/oci2/index.json
fi
echo ""

echo "=== Investigation Complete ==="
echo ""
echo "Artifacts saved:"
echo "  - /tmp/manifest1.json, /tmp/manifest2.json"
echo "  - /tmp/config1.json, /tmp/config2.json"
echo "  - /tmp/oci1/, /tmp/oci2/ (extracted OCI archives)"
echo "  - /tmp/build1.log, /tmp/build2.log"
echo ""
echo "To inspect a specific layer blob:"
echo "  tar -xzf /tmp/oci1/blobs/sha256/<blob-hash> -C /tmp/layer1"
echo ""

# Cleanup local files but keep /tmp for analysis
rm -f oci1.tar oci2.tar
