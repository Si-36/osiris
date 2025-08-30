#!/bin/bash
# Setup script for Mojo/MAX installation

echo "⚡ Installing Mojo/MAX for AURA Intelligence"
echo "==========================================="

# Install Modular CLI
echo "1. Installing Modular CLI..."
curl -s https://get.modular.com | sh -

# Add to PATH
export MODULAR_HOME="$HOME/.modular"
export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"

# Install Mojo
echo "2. Installing Mojo..."
modular install mojo

# Install MAX
echo "3. Installing MAX..."
modular install max

# Verify installation
echo "4. Verifying installation..."
mojo --version
max --version

# Create build directory
echo "5. Creating build directories..."
mkdir -p /workspace/core/src/aura_intelligence/mojo/build
mkdir -p /workspace/core/src/aura_intelligence/mojo/lib

echo "✅ Mojo/MAX installation complete!"