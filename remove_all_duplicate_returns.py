#!/usr/bin/env python3
"""Remove all duplicate return type lines in error_topology.py."""

def fix_file():
    file_path = '/workspace/core/src/aura_intelligence/core/error_topology.py'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this line and next line are duplicate return types
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            
            # Pattern: ) -> Type:
            if (') -> ' in line and line.strip().endswith(':') and
                ') -> ' in next_line and next_line.strip().endswith(':')):
                # Check if they're similar (duplicate)
                if line.strip().split('->')[1] == next_line.strip().split('->')[1]:
                    # Skip the duplicate
                    fixed_lines.append(line)
                    i += 2  # Skip both, we already added one
                    continue
        
        fixed_lines.append(line)
        i += 1
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    # Second pass: fix any remaining issues
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove standalone ) -> Type: lines
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Skip lines that are just ) -> Type: without a function definition
        if line.strip().startswith(') -> ') and line.strip().endswith(':'):
            # Check if previous line has a function definition
            if i > 0 and 'def ' not in lines[i-1]:
                continue  # Skip this orphaned return type
        
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Removed all duplicate return types")

if __name__ == "__main__":
    fix_file()