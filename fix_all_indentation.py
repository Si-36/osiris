#!/usr/bin/env python3

import os
import re
import glob

def fix_indentation_in_file(file_path):
    """Fix common indentation patterns in a Python file."""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    original_content = content
    
    # Fix the specific pattern of wrong indentation after method definitions
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this line has wrong indentation after a method/function definition  
        if i > 0:
            prev_line = lines[i-1]
            # If previous line is a method definition and current line has wrong indentation
            if (re.match(r'\s+def ', prev_line) or re.match(r'\s+async def ', prev_line)) and line.strip():
                # Fix indentation to 8 spaces for method body
                if re.match(r'\s{12,}', line):  # 12+ spaces
                    line = '        ' + line.strip()
        
        # Fix pass statements with wrong indentation
        if line.strip() == 'pass' and re.match(r'\s{12,}', line):
            line = '        pass'
        
        # Fix docstrings with wrong indentation
        if '"""' in line and re.match(r'\s{12,}', line):
            line = '        ' + line.strip()
        
        # Skip duplicate pass statements
        if (line.strip() == 'pass' and 
            i + 1 < len(lines) and 
            lines[i + 1].strip() == 'pass'):
            # Skip this pass, keep the next one
            pass
        else:
            fixed_lines.append(line)
        
        i += 1
    
    content = '\n'.join(fixed_lines)
    
    # Additional pattern fixes
    patterns = [
        # Fix method definitions with wrong body indentation
        (r'(\s+def [^:]+:\s*\n)\s{12,}("""[^"]*"""\s*\n)\s{12,}(pass)', r'\1        \2        \3'),
        (r'(\s+async def [^:]+:\s*\n)\s{12,}("""[^"]*"""\s*\n)\s{12,}(pass)', r'\1        \2        \3'),
        
        # Fix simple pass statements 
        (r'(\s+def [^:]+:\s*\n)\s{12,}(pass)', r'\1        \2'),
        (r'(\s+async def [^:]+:\s*\n)\s{12,}(pass)', r'\1        \2'),
        
        # Remove duplicate pass statements
        (r'(\s+pass\s*\n)\s+pass', r'\1'),
        
        # Fix return statements with wrong indentation
        (r'(\s+def [^:]+:\s*\n)\s{12,}(return [^\\n]+)', r'\1        \2'),
        (r'(\s+async def [^:]+:\s*\n)\s{12,}(return [^\\n]+)', r'\1        \2'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Only write if content changed
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return False
    
    return False

def find_python_files(directory):
    """Find all Python files in directory and subdirectories."""
    pattern = os.path.join(directory, '**', '*.py')
    return glob.glob(pattern, recursive=True)

def main():
    # Fix files in the core/src directory
    core_dir = '/home/sina/projects/osiris-2/core/src'
    
    if not os.path.exists(core_dir):
        print(f"Directory {core_dir} not found")
        return
    
    python_files = find_python_files(core_dir)
    print(f"Found {len(python_files)} Python files")
    
    fixed_count = 0
    
    for file_path in python_files:
        if fix_indentation_in_file(file_path):
            print(f"Fixed: {file_path}")
            fixed_count += 1
    
    print(f"Fixed indentation in {fixed_count} files")

if __name__ == "__main__":
    main()