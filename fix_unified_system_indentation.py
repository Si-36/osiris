#!/usr/bin/env python3
"""Fix indentation issues in unified_system.py."""

def fix_file():
    file_path = '/workspace/core/src/aura_intelligence/core/unified_system.py'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix over-indented methods (8 spaces -> 4 spaces)
        if line.startswith('        async def ') or line.startswith('        def '):
            # This is likely a method that should be at class level (4 spaces)
            fixed_lines.append(line[4:])  # Remove 4 spaces
        else:
            fixed_lines.append(line)
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print("✅ Fixed indentation in unified_system.py")

if __name__ == "__main__":
    fix_file()