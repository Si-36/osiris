#!/usr/bin/env python3
"""
Fix systematic indentation and syntax errors in collective supervisor.py
"""

def fix_collective_supervisor():
    """Fix all indentation and syntax issues"""
    
    file_path = 'core/src/aura_intelligence/collective/supervisor.py'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_class = False
    in_method = False
    
    for i, line in enumerate(lines):
        original = line
        line = line.rstrip() + '\n'
        
        # Fix class definition tracking
        if line.strip().startswith('class ') and line.strip().endswith(':'):
            in_class = True
            in_method = False
            fixed_lines.append(line)
            continue
        
        # Fix method definitions (should be indented 4 spaces in class)
        if in_class and line.strip().startswith('def '):
            in_method = True
            # Ensure method is indented 4 spaces
            fixed_line = '    ' + line.lstrip()
            fixed_lines.append(fixed_line)
            continue
            
        # Fix orphaned control structures
        if line.strip().startswith(('if ', 'elif ', 'else:', 'try:', 'except ', 'for ', 'while ')):
            # Check if this is properly indented within a method/class
            current_indent = len(line) - len(line.lstrip())
            
            if in_method and current_indent < 8:
                # Should be at least 8 spaces for method content
                fixed_line = '        ' + line.lstrip()
                fixed_lines.append(fixed_line)
                continue
            elif in_class and not in_method and current_indent < 4:
                # Should be at least 4 spaces for class content
                fixed_line = '    ' + line.lstrip()
                fixed_lines.append(fixed_line)
                continue
        
        # Fix return statements and other method content
        if line.strip().startswith(('return ', 'logger.', 'print(')):
            current_indent = len(line) - len(line.lstrip())
            
            if in_method and current_indent < 8:
                fixed_line = '        ' + line.lstrip()
                fixed_lines.append(fixed_line)
                continue
        
        # Fix variable assignments in methods
        if in_method and '=' in line and not line.strip().startswith('#'):
            current_indent = len(line) - len(line.lstrip())
            if current_indent < 8 and current_indent > 0:
                fixed_line = '        ' + line.lstrip()
                fixed_lines.append(fixed_line)
                continue
        
        # Keep original line if no fixes needed
        fixed_lines.append(line)
    
    # Write back the fixed content
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed indentation issues in {file_path}")
    print(f"Processed {len(lines)} lines")

if __name__ == '__main__':
    fix_collective_supervisor()