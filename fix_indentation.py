#!/usr/bin/env python3

import re

def fix_indentation_issues(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the specific pattern: method with wrong indentation followed by duplicate pass
    patterns = [
        # Fix method definitions with wrong indentation
        (r'(\s+@abstractmethod\s*\n\s+async def [^:]+:\s*\n)\s{8,12}("""[^"]+"""\s*\n)\s{8,12}(pass\s*\n)\s{4,8}(pass)', r'\1        \2        \3'),
        (r'(\s+@abstractmethod\s*\n\s+def [^:]+:\s*\n)\s{8,12}("""[^"]+"""\s*\n)\s{8,12}(pass\s*\n)\s{4,8}(pass)', r'\1        \2        \3'),
        # Fix async def with wrong indentation
        (r'(\s+async def [^:]+:\s*\n)\s{8,12}("""[^"]+"""\s*\n)\s{8,12}(pass)', r'\1        \2        \3'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Split into lines to fix line by line
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Skip empty lines
        if not line.strip():
            fixed_lines.append(line)
            i += 1
            continue
            
        # Check if this is a method definition
        if re.match(r'\s+(@abstractmethod\s*$|async def |def )', line):
            # This is a method line, keep as is
            fixed_lines.append(line)
            i += 1
            
            # Check next few lines for docstring and pass issues
            while i < len(lines):
                next_line = lines[i]
                if not next_line.strip():
                    fixed_lines.append(next_line)
                    i += 1
                    continue
                    
                # Fix docstring indentation
                if '"""' in next_line and next_line.count(' ') > 8:
                    # Fix docstring to 8 spaces
                    next_line = '        ' + next_line.strip()
                
                # Fix pass statement indentation  
                elif next_line.strip() == 'pass' and next_line.count(' ') > 8:
                    # Fix pass to 8 spaces
                    next_line = '        pass'
                
                # If we find another method or class, break
                elif re.match(r'\s*(@abstractmethod|async def |def |class )', next_line):
                    break
                    
                fixed_lines.append(next_line)
                i += 1
                
                # If this was a pass statement, we're done with this method
                if next_line.strip() == 'pass':
                    break
        else:
            fixed_lines.append(line)
            i += 1
    
    # Join back and remove duplicate pass statements
    content = '\n'.join(fixed_lines)
    
    # Remove duplicate pass statements (pass followed by pass)
    content = re.sub(r'(\s+pass\s*\n)\s+pass', r'\1', content, flags=re.MULTILINE)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed indentation in {file_path}")

if __name__ == "__main__":
    fix_indentation_issues('/home/sina/projects/osiris-2/core/src/aura_intelligence/core/interfaces.py')
    fix_indentation_issues('/home/sina/projects/osiris-2/core/src/aura_intelligence/core/types.py')