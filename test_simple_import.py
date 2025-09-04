#!/usr/bin/env python3
"""Simple test to check if we can import the test agents."""

import sys
import os

# Add the core/src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core', 'src'))

print("Testing imports...")

try:
    print("1. Importing test_code_agent...")
    from aura_intelligence.agents.test_code_agent import create_code_agent
    print("   ✓ Successfully imported test_code_agent")
except Exception as e:
    print(f"   ✗ Failed to import test_code_agent: {e}")
    sys.exit(1)

try:
    print("2. Importing test_data_agent...")
    from aura_intelligence.agents.test_data_agent import create_data_agent
    print("   ✓ Successfully imported test_data_agent")
except Exception as e:
    print(f"   ✗ Failed to import test_data_agent: {e}")
    sys.exit(1)

try:
    print("3. Importing test_creative_agent...")
    from aura_intelligence.agents.test_creative_agent import create_creative_agent
    print("   ✓ Successfully imported test_creative_agent")
except Exception as e:
    print(f"   ✗ Failed to import test_creative_agent: {e}")
    sys.exit(1)

try:
    print("4. Importing test_architect_agent...")
    from aura_intelligence.agents.test_architect_agent import create_architect_agent
    print("   ✓ Successfully imported test_architect_agent")
except Exception as e:
    print(f"   ✗ Failed to import test_architect_agent: {e}")
    sys.exit(1)

try:
    print("5. Importing test_coordinator_agent...")
    from aura_intelligence.agents.test_coordinator_agent import create_coordinator_agent
    print("   ✓ Successfully imported test_coordinator_agent")
except Exception as e:
    print(f"   ✗ Failed to import test_coordinator_agent: {e}")
    sys.exit(1)

print("\n✓ All test agents imported successfully!")

# Try to create one agent
print("\nTrying to create a code agent...")
try:
    agent = create_code_agent("TestCodeAgent")
    print(f"✓ Successfully created {agent.name}")
except Exception as e:
    print(f"✗ Failed to create agent: {e}")

print("\nImport test complete!")