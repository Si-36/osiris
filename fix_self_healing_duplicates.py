#!/usr/bin/env python3
"""Fix duplicate lines and syntax issues in self_healing.py."""

def fix_file():
    file_path = '/workspace/core/src/aura_intelligence/core/self_healing.py'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for duplicate lines
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            
            # If lines are identical and contain function signature parts
            if (line.strip() == next_line.strip() and 
                (') -> ' in line or 'def ' in line or 'async def' in line)):
                # Skip the duplicate
                fixed_lines.append(line)
                i += 2
                continue
        
        fixed_lines.append(line)
        i += 1
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print("✅ Fixed duplicates in self_healing.py")

if __name__ == "__main__":
    fix_file()