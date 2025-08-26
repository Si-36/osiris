#!/usr/bin/env python3
"""Aggressively fix all indentation in unified_interfaces.py."""

def fix_file():
    file_path = '/workspace/core/src/aura_intelligence/core/unified_interfaces.py'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix any method that starts with 8 spaces (should be 4 for class methods)
        if (line.startswith('        async def ') or 
            line.startswith('        def ') or
            line.startswith('        @')):
            # Remove 4 spaces
            fixed_lines.append(line[4:])
        else:
            fixed_lines.append(line)
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print("✅ Aggressively fixed all indentation in unified_interfaces.py")

if __name__ == "__main__":
    fix_file()