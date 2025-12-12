#!/bin/bash

# Quick test - build twice and show what's different

set -e

echo "=== Quick Reproducibility Test ==="
echo ""

# Build 1
echo "Build 1..."
./build-image.sh > /dev/null 2>&1
if [ ! -f oci.tar ]; then
    echo "Build 1 failed"
    exit 1
fi
mv oci.tar oci1.tar
DIGEST1=$(skopeo inspect oci-archive:./oci1.tar | jq -r .Digest)
echo "Digest 1: $DIGEST1"

# Small delay
sleep 1

# Build 2
echo "Build 2..."
./build-image.sh > /dev/null 2>&1
if [ ! -f oci.tar ]; then
    echo "Build 2 failed"
    exit 1
fi
mv oci.tar oci2.tar
DIGEST2=$(skopeo inspect oci-archive:./oci2.tar | jq -r .Digest)
echo "Digest 2: $DIGEST2"

echo ""
if [ "$DIGEST1" = "$DIGEST2" ]; then
    echo "✅ REPRODUCIBLE"
else
    echo "❌ NOT REPRODUCIBLE"
    echo ""
    echo "Comparing layers..."
    echo ""
    echo "Layers 1:"
    skopeo inspect oci-archive:./oci1.tar | jq -r '.Layers[]' > layers1.txt
    cat layers1.txt
    echo ""
    echo "Layers 2:"
    skopeo inspect oci-archive:./oci2.tar | jq -r '.Layers[]' > layers2.txt
    cat layers2.txt
    echo ""
    echo "Diff:"
    diff -u layers1.txt layers2.txt || true
fi

# Cleanup
rm -f oci1.tar oci2.tar layers1.txt layers2.txt
