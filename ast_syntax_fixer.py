#!/usr/bin/env python3
"""Advanced AST-based syntax fixer for Python files."""

import ast
import os
import sys
import re
from pathlib import Path
import autopep8
import black
from typing import List, Tuple
import textwrap

class ASTSyntaxFixer:
    def __init__(self):
        self.fixes_applied = 0
        self.files_fixed = 0
        self.files_failed = 0
        
    def fix_file(self, filepath: str) -> bool:
        """Fix a single Python file using multiple strategies."""
        print(f"🔧 Fixing {os.path.basename(filepath)}...")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Strategy 1: Fix common patterns with regex
            content = self.fix_common_patterns(original_content)
            
            # Strategy 2: Try to parse and reconstruct with AST
            try:
                tree = ast.parse(content)
                # If it parses, format with black
                content = black.format_str(content, mode=black.Mode())
                
            except SyntaxError as e:
                # Strategy 3: Fix specific syntax errors
                content = self.fix_syntax_error(content, e)
                
                # Strategy 4: Use autopep8 for aggressive fixing
                content = autopep8.fix_code(content, options={
                    'aggressive': 3,
                    'max_line_length': 100,
                    'experimental': True
                })
                
                # Try parsing again
                try:
                    ast.parse(content)
                except SyntaxError:
                    # Strategy 5: Line-by-line reconstruction
                    content = self.reconstruct_file(original_content)
            
            # Final validation
            try:
                ast.parse(content)
                # Success! Save the file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.files_fixed += 1
                print(f"  ✅ Fixed successfully!")
                return True
                
            except SyntaxError as e:
                print(f"  ❌ Still has errors: {e.msg} (line {e.lineno})")
                self.files_failed += 1
                return False
                
        except Exception as e:
            print(f"  ❌ Error processing file: {e}")
            self.files_failed += 1
            return False
    
    def fix_common_patterns(self, content: str) -> str:
        """Fix common syntax patterns."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Fix standalone pass statements
            if line.strip() == 'pass' and i > 0:
                prev_line = lines[i-1].strip()
                if not prev_line.endswith(':'):
                    continue  # Skip this pass
            
            # Fix merged async def lines
            if 'async def' in line and not line.strip().startswith('async def'):
                parts = line.split('async def')
                if len(parts) == 2:
                    fixed_lines.append(parts[0])
                    fixed_lines.append('    async def' + parts[1])
                    continue
            
            # Fix docstrings after function definitions
            if '"""' in line and i > 0:
                prev_line = lines[i-1]
                if prev_line.strip().endswith(':'):
                    indent = len(prev_line) - len(prev_line.lstrip())
                    fixed_lines.append(' ' * (indent + 4) + line.strip())
                    continue
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_syntax_error(self, content: str, error: SyntaxError) -> str:
        """Fix specific syntax error."""
        lines = content.split('\n')
        
        if error.lineno and error.lineno <= len(lines):
            line_idx = error.lineno - 1
            
            if 'unexpected indent' in error.msg:
                # Fix unexpected indentation
                problem_line = lines[line_idx]
                # Find expected indentation from context
                expected_indent = self.get_expected_indent(lines, line_idx)
                lines[line_idx] = ' ' * expected_indent + problem_line.lstrip()
                
            elif 'expected an indented block' in error.msg:
                # Add missing indentation
                if line_idx > 0:
                    prev_indent = len(lines[line_idx - 1]) - len(lines[line_idx - 1].lstrip())
                    # Insert a pass statement if needed
                    if line_idx < len(lines) and not lines[line_idx].strip():
                        lines[line_idx] = ' ' * (prev_indent + 4) + 'pass'
                    else:
                        lines.insert(line_idx, ' ' * (prev_indent + 4) + 'pass')
                        
            elif 'unindent does not match' in error.msg:
                # Fix unindent mismatch
                problem_line = lines[line_idx]
                # Find the matching indentation level
                indent_level = self.find_matching_indent(lines, line_idx)
                lines[line_idx] = ' ' * indent_level + problem_line.lstrip()
        
        return '\n'.join(lines)
    
    def get_expected_indent(self, lines: List[str], line_idx: int) -> int:
        """Get expected indentation for a line."""
        # Look backward for context
        for i in range(line_idx - 1, -1, -1):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                continue
                
            indent = len(lines[i]) - len(lines[i].lstrip())
            
            # If previous line ends with :, we need more indent
            if line.endswith(':'):
                return indent + 4
            else:
                return indent
        
        return 0
    
    def find_matching_indent(self, lines: List[str], line_idx: int) -> int:
        """Find matching indentation level."""
        problem_line = lines[line_idx].strip()
        
        # Common dedent keywords
        if problem_line.startswith(('except', 'elif', 'else', 'finally')):
            # Find the matching if/try
            for i in range(line_idx - 1, -1, -1):
                line = lines[i].strip()
                if line.startswith(('if ', 'try:', 'for ', 'while ')):
                    return len(lines[i]) - len(lines[i].lstrip())
        
        # Default to previous non-empty line
        for i in range(line_idx - 1, -1, -1):
            if lines[i].strip():
                return len(lines[i]) - len(lines[i].lstrip())
        
        return 0
    
    def reconstruct_file(self, content: str) -> str:
        """Reconstruct file with proper structure."""
        lines = content.split('\n')
        reconstructed = []
        indent_stack = [0]
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped or stripped.startswith('#'):
                reconstructed.append(line)
                continue
            
            # Determine proper indentation
            if stripped.startswith(('class ', 'def ', 'async def')):
                # New definition
                if indent_stack:
                    indent = indent_stack[-1]
                else:
                    indent = 0
                reconstructed.append(' ' * indent + stripped)
                if stripped.endswith(':'):
                    indent_stack.append(indent + 4)
                    
            elif stripped.startswith(('return', 'pass', 'continue', 'break')):
                # Statement that doesn't increase indent
                indent = indent_stack[-1] if indent_stack else 0
                reconstructed.append(' ' * indent + stripped)
                
            elif stripped.startswith(('except', 'elif', 'else', 'finally')):
                # Dedent to match if/try
                if len(indent_stack) > 1:
                    indent_stack.pop()
                indent = indent_stack[-1] if indent_stack else 0
                reconstructed.append(' ' * indent + stripped)
                if stripped.endswith(':'):
                    indent_stack.append(indent + 4)
                    
            else:
                # Regular line
                indent = indent_stack[-1] if indent_stack else 0
                reconstructed.append(' ' * indent + stripped)
                
                # Check if we need to dedent
                if stripped.endswith(':'):
                    indent_stack.append(indent + 4)
        
        return '\n'.join(reconstructed)
    
    def fix_critical_files(self, critical_files: List[Tuple[str, str]]) -> dict:
        """Fix a list of critical files."""
        results = {'fixed': [], 'failed': []}
        
        for filepath, error in critical_files:
            if self.fix_file(filepath):
                results['fixed'].append(filepath)
            else:
                results['failed'].append((filepath, error))
        
        return results

def load_critical_files():
    """Load critical files from the index report."""
    import json
    
    with open('/workspace/AURA_CODEBASE_INDEX.json', 'r') as f:
        report = json.load(f)
    
    # Get critical path files
    critical_files = []
    for file_info in report['critical_path']:
        filepath = '/workspace/' + file_info['file']
        if os.path.exists(filepath):
            critical_files.append((filepath, file_info['error']))
    
    # Add infrastructure files
    if 'infrastructure' in report['components']:
        for filepath in report['components']['infrastructure']['files']:
            if filepath.endswith('.py') and os.path.exists(filepath):
                critical_files.append((filepath, 'infrastructure component'))
    
    return critical_files

def main():
    # Install required packages if needed
    try:
        import autopep8
        import black
    except ImportError:
        print("📦 Installing required packages...")
        os.system("pip install autopep8 black --quiet")
        import autopep8
        import black
    
    print("🔧 Advanced AST-based Syntax Fixer")
    print("=" * 50)
    
    fixer = ASTSyntaxFixer()
    
    # Load critical files
    critical_files = load_critical_files()
    
    print(f"\n📋 Found {len(critical_files)} critical files to fix")
    
    # Fix critical files first
    print("\n🚀 Fixing critical path files...")
    results = fixer.fix_critical_files(critical_files[:10])  # Start with top 10
    
    print(f"\n📊 Results:")
    print(f"  ✅ Fixed: {len(results['fixed'])} files")
    print(f"  ❌ Failed: {len(results['failed'])} files")
    
    if results['fixed']:
        print("\n✅ Successfully fixed:")
        for f in results['fixed']:
            print(f"  - {os.path.relpath(f, '/workspace')}")
    
    if results['failed']:
        print("\n❌ Still have errors:")
        for f, error in results['failed'][:5]:
            print(f"  - {os.path.relpath(f, '/workspace')}: {error}")

if __name__ == "__main__":
    main()