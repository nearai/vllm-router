#!/bin/bash

# Parse command line arguments
PUSH=false
REPO=""
LOCAL_DEV=false

while [[ $# -gt 0 ]]; do
	case $1 in
	--push)
		PUSH=true
		REPO="$2"
		if [ -z "$REPO" ]; then
			echo "Error: --push requires a repository argument"
			echo "Usage: $0 [--push <repo>[:<tag>]] [--local]"
			exit 1
		fi
		shift 2
		;;
	--local)
		LOCAL_DEV=true
		shift
		;;
	*)
		echo "Usage: $0 [--push <repo>[:<tag>]] [--local]"
		exit 1
		;;
	esac
done

require_command() {
	local cmd="$1"
	if ! command -v "$cmd" >/dev/null 2>&1; then
		echo "Error: required command '$cmd' not found in PATH" >&2
		exit 1
	fi
}

for required in docker skopeo jq git; do
	require_command "$required"
done

# Check if buildkit_20 already exists before creating it
if ! docker buildx inspect buildkit_20 &>/dev/null; then
	docker buildx create --use --driver-opt image=moby/buildkit:v0.20.2 --name buildkit_20
fi

if [ "$LOCAL_DEV" = true ]; then
	# Local dev build - simple docker build
	TEMP_TAG="vllm-router:dev"
	echo "Building local development image..."
	
	# Create .GIT_REV file for Dockerfile (even in local dev)
	git rev-parse HEAD >.GIT_REV
	
	docker build -t "$TEMP_TAG" .
	
	if [ "$?" -ne 0 ]; then
		echo "Build failed"
		rm .GIT_REV
		exit 1
	fi
	
	echo "Local development build completed: $TEMP_TAG"
else
	# Reproducible build with OCI archive
	git rev-parse HEAD >.GIT_REV
	TEMP_TAG="vllm-router:$(date +%s)"
	docker buildx build --builder buildkit_20 --no-cache \
		--build-arg SOURCE_DATE_EPOCH="0" \
		--output type=oci,dest=./oci.tar,rewrite-timestamp=true \
		--output type=docker,name="$TEMP_TAG",rewrite-timestamp=true .

	if [ "$?" -ne 0 ]; then
		echo "Build failed"
		rm .GIT_REV
		exit 1
	fi

	echo "Build completed, manifest digest:"
	echo ""
	skopeo inspect oci-archive:./oci.tar | jq .Digest
	echo ""
fi

if [ "$PUSH" = true ]; then
	if [ "$LOCAL_DEV" = true ]; then
		echo "Pushing local development image to $REPO..."
		docker tag "$TEMP_TAG" "$REPO"
		docker push "$REPO"
		echo "Local development image pushed successfully to $REPO"
	else
		echo "Pushing image to $REPO..."
		skopeo copy --insecure-policy oci-archive:./oci.tar docker://"$REPO"
		echo "Image pushed successfully to $REPO"
	fi
else
	if [ "$LOCAL_DEV" = false ]; then
		echo "To push the image to a registry, run:"
		echo ""
		echo " $0 --push <repo>[:<tag>]"
		echo ""
		echo "Or use skopeo directly:"
		echo ""
		echo " skopeo copy --insecure-policy oci-archive:./oci.tar docker://<repo>[:<tag>]"
		echo ""
	fi
fi
echo ""

# Clean up the temporary image from Docker daemon (unless keeping for local dev)
if [ "$LOCAL_DEV" = false ]; then
	docker rmi "$TEMP_TAG" 2>/dev/null || true
else
	echo "Local development image kept: $TEMP_TAG"
	echo "You can run it with: docker run -it $TEMP_TAG"
fi

# Clean up .GIT_REV file (created for both local and production builds)
rm .GIT_REV
