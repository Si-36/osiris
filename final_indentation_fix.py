#!/usr/bin/env python3

import os
import ast
import traceback
from pathlib import Path

def fix_specific_indentation_issues(file_path):
    """Fix specific indentation issues that occur after method/function definitions."""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    original_content = content
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line is a function definition
        if (line.strip().startswith('def ') or line.strip().startswith('async def ')) and line.rstrip().endswith(':'):
            fixed_lines.append(line)
            i += 1
            
            # Check the next lines for incorrect indentation
            while i < len(lines):
                next_line = lines[i]
                
                if not next_line.strip():  # Empty line
                    fixed_lines.append(next_line)
                    i += 1
                    continue
                
                # If this line starts at column 0 and isn't a function/class def, we're done
                if not next_line.startswith(' ') and not next_line.startswith('\t'):
                    if (next_line.strip().startswith('def ') or 
                        next_line.strip().startswith('class ') or 
                        next_line.strip().startswith('async def ')):
                        # This is a new function/class, break and handle it normally
                        break
                    elif next_line.strip():
                        # This is code that should be indented
                        fixed_lines.append('    ' + next_line.strip())
                        i += 1
                        continue
                
                # Calculate expected indentation based on function definition
                func_indent = len(line) - len(line.lstrip())
                expected_body_indent = func_indent + 4
                
                current_indent = len(next_line) - len(next_line.lstrip())
                
                # If indentation is wrong, fix it
                if next_line.strip() and current_indent != expected_body_indent:
                    # Fix the indentation to expected level
                    fixed_lines.append(' ' * expected_body_indent + next_line.strip())
                else:
                    fixed_lines.append(next_line)
                
                i += 1
                
                # If we encounter another function/class definition at the same level, stop
                if (next_line.strip().startswith('def ') or 
                    next_line.strip().startswith('class ') or
                    next_line.strip().startswith('async def ')) and current_indent == func_indent:
                    break
        else:
            fixed_lines.append(line)
            i += 1
    
    content = '\n'.join(fixed_lines)
    
    # Apply some final cleanup patterns
    patterns = [
        # Fix common except block patterns
        (r'(\s+except [^:]+:\s*\n)\s{0,3}([a-zA-Z_])', r'\1            \2'),
        # Fix pass statements after except
        (r'(\s+except [^:]+:\s*\n)\s{0,3}(pass)', r'\1            \2'),
        # Fix return statements after except
        (r'(\s+except [^:]+:\s*\n)\s{0,3}(return)', r'\1            \2'),
        # Fix raise statements 
        (r'(\s+def [^:]+:\s*\n)\s{0,3}(raise)', r'\1        \2'),
        (r'(\s+async def [^:]+:\s*\n)\s{0,3}(raise)', r'\1        \2'),
    ]
    
    for pattern, replacement in patterns:
        import re
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

def find_files_with_syntax_errors():
    """Find Python files with syntax errors in the core directory."""
    
    core_dir = Path('/home/sina/projects/osiris-2/core/src')
    problematic_files = []
    
    for py_file in core_dir.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            # Try to parse with ast
            ast.parse(content)
        except (SyntaxError, IndentationError) as e:
            problematic_files.append((str(py_file), str(e)))
            
    return problematic_files

def main():
    print("Finding files with syntax/indentation errors...")
    problematic_files = find_files_with_syntax_errors()
    
    print(f"Found {len(problematic_files)} files with syntax/indentation errors")
    
    fixed_count = 0
    for file_path, error in problematic_files:
        print(f"Fixing: {file_path} - {error}")
        if fix_specific_indentation_issues(file_path):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")
    
    # Check again
    print("\nChecking remaining issues...")
    remaining_issues = find_files_with_syntax_errors()
    
    if remaining_issues:
        print(f"Still {len(remaining_issues)} files with issues:")
        for file_path, error in remaining_issues[:10]:  # Show first 10
            print(f"  {file_path}: {error}")
    else:
        print("✅ All syntax/indentation errors fixed!")

if __name__ == "__main__":
    main()