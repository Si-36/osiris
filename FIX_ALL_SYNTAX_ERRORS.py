#!/usr/bin/env python3
"""
Fix ALL syntax errors in the AURA codebase
==========================================
This script will fix all common syntax errors:
1. 'await' outside async function
2. Incorrect indentation
3. Misplaced pass statements
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Tuple, Optional

class SyntaxFixer:
    def __init__(self):
        self.fixed_count = 0
        self.error_count = 0
        self.files_processed = 0
        
    def find_python_files(self, directory: str) -> List[Path]:
        """Find all Python files in directory"""
        return list(Path(directory).rglob("*.py"))
    
    def check_syntax(self, filepath: Path) -> Optional[Tuple[str, int, str]]:
        """Check if file has syntax errors"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content)
            return None
        except SyntaxError as e:
            return (str(filepath), e.lineno or 0, e.msg or "Unknown error")
        except Exception as e:
            return (str(filepath), 0, str(e))
    
    def fix_await_outside_async(self, content: str) -> Tuple[str, bool]:
        """Fix 'await' outside async function errors"""
        lines = content.split('\n')
        fixed = False
        
        # Pattern to find function definitions
        func_pattern = re.compile(r'^(\s*)def\s+(\w+)\s*\(')
        await_pattern = re.compile(r'\s+await\s+')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if this is a function definition
            func_match = func_pattern.match(line)
            if func_match:
                indent = func_match.group(1)
                func_name = func_match.group(2)
                
                # Look ahead to see if this function uses await
                j = i + 1
                func_end = None
                uses_await = False
                
                # Find the function body
                while j < len(lines):
                    next_line = lines[j]
                    
                    # Check if we're still in the function
                    if next_line.strip() and not next_line.startswith(indent + ' ') and not next_line.startswith(indent + '\t'):
                        # Check if it's a dedent (end of function)
                        if not next_line.startswith(indent):
                            func_end = j
                            break
                    
                    # Check if line contains await
                    if await_pattern.search(next_line):
                        uses_await = True
                    
                    j += 1
                
                # If function uses await but isn't async, make it async
                if uses_await and not line.strip().startswith('async '):
                    lines[i] = line.replace('def ', 'async def ', 1)
                    fixed = True
                    print(f"  Fixed: Made {func_name} async")
            
            i += 1
        
        return '\n'.join(lines), fixed
    
    def fix_indentation_errors(self, content: str) -> Tuple[str, bool]:
        """Fix common indentation errors"""
        lines = content.split('\n')
        fixed = False
        
        # Fix misplaced pass statements
        i = 0
        while i < len(lines) - 1:
            line = lines[i]
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            
            # Remove pass statements that are followed by docstrings
            if line.strip() == 'pass' and next_line.strip().startswith('"""'):
                lines.pop(i)
                fixed = True
                print(f"  Fixed: Removed misplaced pass statement")
                continue
            
            # Fix incorrect indentation after function definitions
            if re.match(r'^\s*(async\s+)?def\s+\w+.*:$', line):
                # Check if next line is incorrectly indented
                if next_line and not next_line.strip().startswith('"""') and not next_line.strip().startswith('#'):
                    current_indent = len(line) - len(line.lstrip())
                    next_indent = len(next_line) - len(next_line.lstrip())
                    
                    # If next line is not indented more than current, fix it
                    if next_line.strip() and next_indent <= current_indent:
                        lines[i + 1] = ' ' * (current_indent + 4) + next_line.lstrip()
                        fixed = True
                        print(f"  Fixed: Indentation after function definition")
            
            i += 1
        
        # Fix methods that are incorrectly indented inside classes
        in_class = False
        class_indent = 0
        
        for i in range(len(lines)):
            line = lines[i]
            
            # Detect class definition
            if re.match(r'^(\s*)class\s+\w+.*:$', line):
                in_class = True
                class_indent = len(line) - len(line.lstrip())
            
            # Fix method indentation inside classes
            if in_class and re.match(r'^(\s*)(async\s+)?def\s+\w+.*:$', line):
                current_indent = len(line) - len(line.lstrip())
                expected_indent = class_indent + 4
                
                if current_indent != expected_indent:
                    # Fix the method definition line
                    lines[i] = ' ' * expected_indent + line.lstrip()
                    fixed = True
                    print(f"  Fixed: Method indentation in class")
        
        return '\n'.join(lines), fixed
    
    def fix_file(self, filepath: Path) -> bool:
        """Fix all syntax errors in a file"""
        print(f"\nProcessing: {filepath}")
        
        try:
            # Read file
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            total_fixed = False
            
            # Apply fixes
            content, fixed1 = self.fix_await_outside_async(content)
            total_fixed |= fixed1
            
            content, fixed2 = self.fix_indentation_errors(content)
            total_fixed |= fixed2
            
            # Only write if we made changes
            if total_fixed:
                # Verify the fixed content is valid Python
                try:
                    ast.parse(content)
                    # Write back
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"✅ Fixed and saved: {filepath}")
                    self.fixed_count += 1
                    return True
                except SyntaxError as e:
                    print(f"❌ Fix attempt failed: {e}")
                    # Restore original content
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                    self.error_count += 1
                    return False
            else:
                print(f"  No fixes needed")
                return True
                
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            self.error_count += 1
            return False
    
    def fix_directory(self, directory: str):
        """Fix all Python files in directory"""
        print(f"🔧 Fixing syntax errors in: {directory}")
        print("=" * 60)
        
        # Find all Python files
        python_files = self.find_python_files(directory)
        print(f"Found {len(python_files)} Python files")
        
        # Check and fix files with errors
        for filepath in python_files:
            error = self.check_syntax(filepath)
            if error:
                print(f"\n❌ Syntax error in {error[0]}")
                print(f"   Line {error[1]}: {error[2]}")
                self.fix_file(filepath)
            self.files_processed += 1
        
        print("\n" + "=" * 60)
        print(f"✅ Processed {self.files_processed} files")
        print(f"✅ Fixed {self.fixed_count} files")
        print(f"❌ Failed to fix {self.error_count} files")

def main():
    """Fix syntax errors for persistence tests"""
    fixer = SyntaxFixer()
    
    # Fix the directories needed for persistence tests
    directories = [
        'core/src/aura_intelligence/persistence',
        'core/src/aura_intelligence/agents/resilience',
        'core/src/aura_intelligence/neural',
        'core/src/aura_intelligence/adapters',
        'core/src/aura_intelligence/tda',
        'core/src/aura_intelligence/events',
        'core/src/aura_intelligence/consensus',
        'core/src/aura_intelligence/observability',
        'core/src/aura_intelligence/resilience',
        'core/src/aura_intelligence/core',
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            fixer.fix_directory(directory)
    
    print("\n🎯 Now try running your persistence tests:")
    print("   python3 TEST_PERSISTENCE_NOW.py")
    print("   ./RUN_PERSISTENCE_DOCKER_FREE.sh")

if __name__ == "__main__":
    main()