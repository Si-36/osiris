#!/usr/bin/env python3
"""Fix split and duplicate function signatures."""

import re
from pathlib import Path

def fix_split_signatures(file_path):
    """Fix function signatures that are split incorrectly."""
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for function definitions that might be split
        if 'def ' in line and not line.strip().endswith(':'):
            # Collect the full signature
            signature_lines = [line]
            j = i + 1
            
            # Look ahead for the rest of the signature
            while j < len(lines) and not lines[j].strip().endswith(':'):
                signature_lines.append(lines[j])
                j += 1
                if j - i > 5:  # Safety limit
                    break
            
            if j < len(lines) and lines[j].strip().endswith(':'):
                signature_lines.append(lines[j])
                
                # Join and clean the signature
                full_sig = ' '.join(line.strip() for line in signature_lines)
                # Remove duplicate parts
                full_sig = re.sub(r'(\w+\))\s*->\s*(\w+:)\s*\1\s*->\s*\2', r'\1 -> \2', full_sig)
                
                # Add proper indentation
                indent = len(line) - len(line.lstrip())
                fixed_lines.append(' ' * indent + full_sig + '\n')
                
                i = j + 1
                continue
        
        fixed_lines.append(line)
        i += 1
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    return True

def main():
    core_dir = Path('/workspace/core/src/aura_intelligence/core')
    
    print("🔧 Fixing split function signatures...")
    
    for py_file in sorted(core_dir.glob('*.py')):
        if fix_split_signatures(py_file):
            print(f"✅ Checked {py_file.name}")

if __name__ == "__main__":
    main()