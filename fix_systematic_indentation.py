#!/usr/bin/env python3
"""
Fix systematic indentation errors in AURA Intelligence codebase
Based on the specific error patterns found
"""

import os
import re
from pathlib import Path

def fix_indentation_errors(file_path):
    """Fix common indentation errors in a Python file"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        print(f"Skipping binary file: {file_path}")
        return False
    
    fixed_lines = []
    changes_made = 0
    
    for i, line in enumerate(lines):
        current_line = line
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        
        # Pattern 1: Function definition followed by docstring without indentation
        if (re.match(r'^\s*(def|async def)\s+\w+\([^)]*\):\s*$', current_line) and
            next_line.strip().startswith('"""') and 
            len(next_line) - len(next_line.lstrip()) <= len(current_line) - len(current_line.lstrip())):
            
            # Fix the next line indentation
            base_indent = len(current_line) - len(current_line.lstrip())
            fixed_next_line = ' ' * (base_indent + 4) + next_line.lstrip()
            lines[i + 1] = fixed_next_line
            changes_made += 1
            print(f"  Fixed function docstring indentation at line {i+2}")
        
        # Pattern 2: Function definition followed by pass without indentation  
        elif (re.match(r'^\s*(def|async def)\s+\w+\([^)]*\):\s*$', current_line) and
              next_line.strip() == "pass" and
              len(next_line) - len(next_line.lstrip()) <= len(current_line) - len(current_line.lstrip())):
            
            base_indent = len(current_line) - len(current_line.lstrip())
            lines[i + 1] = ' ' * (base_indent + 4) + "pass\n"
            changes_made += 1
            print(f"  Fixed function pass indentation at line {i+2}")
        
        # Pattern 3: If statement without indented body
        elif (re.match(r'^\s*if\s+.*:\s*$', current_line) and
              next_line.strip() and
              len(next_line) - len(next_line.lstrip()) <= len(current_line) - len(current_line.lstrip())):
            
            base_indent = len(current_line) - len(current_line.lstrip())
            lines[i + 1] = ' ' * (base_indent + 4) + next_line.lstrip()
            changes_made += 1
            print(f"  Fixed if statement indentation at line {i+2}")
        
        # Pattern 4: Class definition without indented body
        elif (re.match(r'^\s*class\s+\w+.*:\s*$', current_line) and
              next_line.strip() == "pass" and
              len(next_line) - len(next_line.lstrip()) <= len(current_line) - len(current_line.lstrip())):
            
            base_indent = len(current_line) - len(current_line.lstrip())
            lines[i + 1] = ' ' * (base_indent + 4) + "pass\n"
            changes_made += 1
            print(f"  Fixed class pass indentation at line {i+2}")
        
        # Pattern 5: Try statement without indented body
        elif (re.match(r'^\s*try:\s*$', current_line) and
              next_line.strip() and
              len(next_line) - len(next_line.lstrip()) <= len(current_line) - len(current_line.lstrip())):
            
            base_indent = len(current_line) - len(current_line.lstrip())
            lines[i + 1] = ' ' * (base_indent + 4) + next_line.lstrip()
            changes_made += 1
            print(f"  Fixed try statement indentation at line {i+2}")
        
        # Pattern 6: With statement without indented body
        elif (re.match(r'^\s*with\s+.*:\s*$', current_line) and
              next_line.strip() and
              len(next_line) - len(next_line.lstrip()) <= len(current_line) - len(current_line.lstrip())):
            
            base_indent = len(current_line) - len(current_line.lstrip())
            lines[i + 1] = ' ' * (base_indent + 4) + next_line.lstrip()
            changes_made += 1
            print(f"  Fixed with statement indentation at line {i+2}")
        
        fixed_lines.append(current_line)
    
    if changes_made > 0:
        # Write the fixed content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"✅ Fixed {changes_made} indentation errors in {file_path}")
        return True
    
    return False

def main():
    """Fix indentation errors in all Python files in core/src/aura_intelligence"""
    
    print("🔧 Starting systematic indentation fix for AURA Intelligence core...")
    
    core_path = Path("core/src/aura_intelligence")
    if not core_path.exists():
        print(f"❌ Core path not found: {core_path}")
        return
    
    python_files = list(core_path.rglob("*.py"))
    print(f"📁 Found {len(python_files)} Python files to check")
    
    fixed_count = 0
    
    for file_path in python_files:
        try:
            # Test if file has syntax errors first
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            try:
                compile(content, str(file_path), 'exec')
                continue  # File is already valid
            except SyntaxError as e:
                if "expected an indented block" in str(e):
                    print(f"🔧 Fixing indentation in {file_path}")
                    if fix_indentation_errors(file_path):
                        fixed_count += 1
                else:
                    print(f"⚠️  Non-indentation syntax error in {file_path}: {e}")
        
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    print(f"\n✅ Fixed indentation errors in {fixed_count} files")
    
    # Test a few key files to see if they now compile
    print("\n🧪 Testing key files after fixes:")
    test_files = [
        "core/src/aura_intelligence/infrastructure/guardrails.py",
        "core/src/aura_intelligence/tda/real_tda.py", 
        "core/src/aura_intelligence/lnn/real_mit_lnn.py",
        "core/src/aura_intelligence/memory/shape_memory_v2.py"
    ]
    
    for test_file in test_files:
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            compile(content, test_file, 'exec')
            print(f"✅ {test_file} - syntax OK")
        except SyntaxError as e:
            print(f"❌ {test_file} - still has syntax error: {e}")
        except FileNotFoundError:
            print(f"⚠️  {test_file} - not found")

if __name__ == "__main__":
    main()