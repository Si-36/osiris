#!/usr/bin/env python3
"""Final comprehensive fix for all issues in error_topology.py."""

import re

def fix_file():
    file_path = '/workspace/core/src/aura_intelligence/core/error_topology.py'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Fix split function signatures
        if 'def ' in line and '(self' in line and ':' not in line:
            # Collect the full signature
            sig_lines = [line.rstrip()]
            j = i + 1
            
            # Look for continuation
            while j < len(lines):
                next_line = lines[j].rstrip()
                sig_lines.append(next_line)
                
                if '):' in next_line or '-> ' in next_line:
                    if ':' in next_line:
                        # Found end of signature
                        break
                j += 1
                if j - i > 5:  # Safety
                    break
            
            # Check if we have a complete signature
            full_sig = ' '.join(l.strip() for l in sig_lines)
            
            # Check for the next line - might be duplicate or docstring
            if j + 1 < len(lines):
                next_after_sig = lines[j + 1].strip()
                
                # Skip duplicate lines
                if '):' in next_after_sig and '->' in next_after_sig:
                    j += 1  # Skip the duplicate
                
                # Check for incorrectly indented docstring
                if j + 1 < len(lines) and '"""' in lines[j + 1]:
                    doc_line = lines[j + 1]
                    # Fix docstring indentation
                    indent = len(line) - len(line.lstrip())
                    lines[j + 1] = ' ' * (indent + 4) + doc_line.strip() + '\n'
            
            # Clean and write the signature
            if '->' in full_sig and ':' in full_sig:
                indent = len(line) - len(line.lstrip())
                fixed_lines.append(' ' * indent + full_sig + '\n')
                i = j + 1
                continue
        
        fixed_lines.append(line)
        i += 1
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print("✅ Applied final fixes to error_topology.py")

if __name__ == "__main__":
    fix_file()