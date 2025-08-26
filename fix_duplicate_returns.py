#!/usr/bin/env python3
"""Fix duplicate return type lines in exceptions.py."""

def fix_duplicates():
    file_path = '/workspace/core/src/aura_intelligence/core/exceptions.py'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this line and next line are duplicate return types
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            
            # If both lines are identical and contain ') -> '
            if (') -> ' in line and 
                line.strip() == next_line.strip() and
                line.strip().endswith(':')):
                # Skip the duplicate
                fixed_lines.append(line)
                i += 2  # Skip both lines, we already added one
                continue
        
        fixed_lines.append(line)
        i += 1
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"✅ Fixed duplicate return type lines in exceptions.py")

if __name__ == "__main__":
    fix_duplicates()