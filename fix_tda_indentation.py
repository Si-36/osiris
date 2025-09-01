#!/usr/bin/env python3
"""Fix indentation in agent_topology.py"""

import re

# Read the file
with open('core/src/aura_intelligence/tda/agent_topology.py', 'r') as f:
    content = f.read()

# Fix the analyze_workflow method - replace 12 spaces with 8 spaces for the body
lines = content.split('\n')
fixed_lines = []
in_analyze_workflow = False
fix_indent = False

for i, line in enumerate(lines):
    if 'async def analyze_workflow' in line:
        in_analyze_workflow = True
        fixed_lines.append(line)
    elif in_analyze_workflow and line.strip().startswith('return features'):
        # Fix this line and end
        fixed_lines.append('        return features')
        in_analyze_workflow = False
    elif in_analyze_workflow and line.startswith('            '):
        # Replace 12 spaces with 8 spaces
        fixed_lines.append(line[4:])  # Remove 4 spaces
    else:
        fixed_lines.append(line)

# Write back
with open('core/src/aura_intelligence/tda/agent_topology.py', 'w') as f:
    f.write('\n'.join(fixed_lines))

print("Fixed indentation in agent_topology.py")