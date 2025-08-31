#!/usr/bin/env python3
"""Fix streams.py to handle missing EventProcessor"""

# Read the file
with open('/workspace/core/src/aura_intelligence/events/streams.py', 'r') as f:
    content = f.read()

# Find where classes inherit from EventProcessor
import re

# Add a check after the imports
check_code = '''
# If EventProcessor is not available, create a dummy base class
if EventProcessor is None:
    class EventProcessor:
        """Dummy EventProcessor when aiokafka not available"""
        pass
'''

# Find the line after the imports (after line 40 based on what we saw)
lines = content.split('\n')

# Insert after the logger/tracer/meter definitions (around line 39)
insert_pos = None
for i, line in enumerate(lines):
    if 'meter = metrics.get_meter(__name__)' in line:
        insert_pos = i + 1
        break

if insert_pos:
    lines.insert(insert_pos, '')
    lines.insert(insert_pos + 1, check_code)
    
    # Write back
    with open('/workspace/core/src/aura_intelligence/events/streams.py', 'w') as f:
        f.write('\n'.join(lines))
    
    print("✅ Added EventProcessor dummy class to streams.py")
else:
    print("❌ Could not find insertion point")