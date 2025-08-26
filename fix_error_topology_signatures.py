#!/usr/bin/env python3
"""Fix all split function signatures in error_topology.py."""

import re

def fix_file():
    file_path = '/workspace/core/src/aura_intelligence/core/error_topology.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to find split function signatures with duplicates
    # This matches: func(self, param:\n    Type) -> Return:\n    Type) -> Return:
    pattern = r'(\w+\(self[^)]*,\s*\w+:\s*\n\s+)(\w+[^)]*\))\s*->\s*([^:]+):\s*\n\s+\2\s*->\s*\3:'
    
    # Replace with properly formatted signature
    content = re.sub(pattern, r'\1\2 -> \3:', content)
    
    # Another pattern: func(self,\n    param) -> Type:\n    param) -> Type:
    pattern2 = r'def (\w+)\(self,\s*(\w+):\s*\n\s+([^)]+)\)\s*->\s*([^:]+):\s*\n\s+\3\)\s*->\s*\4:'
    content = re.sub(pattern2, r'def \1(self, \2: \3) -> \4:', content)
    
    # Fix any remaining split signatures
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for function definitions
        if 'def ' in line and '(self' in line and not line.strip().endswith(':'):
            # Collect full signature
            sig_lines = [line]
            j = i + 1
            
            while j < len(lines) and not lines[j].strip().endswith(':'):
                sig_lines.append(lines[j])
                j += 1
                if j - i > 4:
                    break
            
            if j < len(lines) and '):' in lines[j]:
                sig_lines.append(lines[j])
                
                # Join the signature
                full_sig = ' '.join(l.strip() for l in sig_lines)
                # Remove duplicates
                parts = full_sig.split(') -> ')
                if len(parts) >= 2:
                    # Check for duplicate return types
                    ret_parts = parts[1].split(':')
                    if len(ret_parts) >= 2:
                        # Clean up
                        clean_sig = parts[0] + ') -> ' + ret_parts[0].strip() + ':'
                        # Get proper indentation
                        indent = len(line) - len(line.lstrip())
                        fixed_lines.append(' ' * indent + clean_sig)
                        i = j + 1
                        continue
        
        fixed_lines.append(line)
        i += 1
    
    content = '\n'.join(fixed_lines)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed split signatures in error_topology.py")

if __name__ == "__main__":
    fix_file()