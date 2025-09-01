#!/bin/bash
# Build script for Mojo kernels

echo "🔨 Building Mojo Kernels for AURA"
echo "================================="

# Set paths
MOJO_DIR="/workspace/core/src/aura_intelligence/mojo"
BUILD_DIR="$MOJO_DIR/build"
LIB_DIR="$MOJO_DIR/lib"

# Build selective scan kernel
echo "1. Building selective scan kernel..."
mojo build $MOJO_DIR/selective_scan_kernel.mojo \
    -o $BUILD_DIR/selective_scan_kernel.so \
    --no-optimization-warnings \
    -D SIMD_WIDTH=16

# Build TDA distance kernel
echo "2. Building TDA distance kernel..."
mojo build $MOJO_DIR/tda_distance_kernel.mojo \
    -o $BUILD_DIR/tda_distance_kernel.so \
    --no-optimization-warnings \
    -D SIMD_WIDTH=16

# Build expert routing kernel
echo "3. Building expert routing kernel..."
mojo build $MOJO_DIR/expert_routing_kernel.mojo \
    -o $BUILD_DIR/expert_routing_kernel.so \
    --no-optimization-warnings \
    -D SIMD_WIDTH=16

# Create Python bindings
echo "4. Creating Python bindings..."
mojo package $MOJO_DIR -o $LIB_DIR/aura_mojo_kernels

echo "✅ Mojo kernel compilation complete!"
echo "Kernels available at: $BUILD_DIR"