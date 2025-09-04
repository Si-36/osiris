#!/usr/bin/env python3
"""
Check for syntax errors in all Python files
===========================================
"""

import ast
import os
import sys
from pathlib import Path

def check_file(filepath):
    """Check a single Python file for syntax errors"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse the file
        ast.parse(content, filename=str(filepath))
        return True, None
    except SyntaxError as e:
        return False, f"{e.msg} at line {e.lineno}"
    except Exception as e:
        return False, str(e)

def check_directory(directory):
    """Check all Python files in a directory for syntax errors"""
    errors = []
    checked = 0
    
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ directories
        if '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                checked += 1
                
                success, error = check_file(filepath)
                if not success:
                    errors.append((filepath, error))
    
    return checked, errors

def main():
    """Check all Python files in the project"""
    print("🔍 Checking for syntax errors in Python files...")
    print("=" * 60)
    
    # Directories to check
    directories = [
        'core/src/aura_intelligence',
        'test_*.py',
        '*.py'
    ]
    
    total_checked = 0
    all_errors = []
    
    # Check directories
    if os.path.exists('core/src/aura_intelligence'):
        checked, errors = check_directory('core/src/aura_intelligence')
        total_checked += checked
        all_errors.extend(errors)
    
    # Check test files in root
    for pattern in ['test_*.py', '*.py']:
        for filepath in Path('.').glob(pattern):
            if filepath.is_file():
                total_checked += 1
                success, error = check_file(filepath)
                if not success:
                    all_errors.append((str(filepath), error))
    
    # Report results
    print(f"\nChecked {total_checked} Python files")
    
    if all_errors:
        print(f"\n❌ Found {len(all_errors)} syntax errors:\n")
        for filepath, error in all_errors:
            print(f"  {filepath}")
            print(f"    Error: {error}\n")
        return 1
    else:
        print("\n✅ No syntax errors found!")
        return 0

if __name__ == "__main__":
    sys.exit(main())