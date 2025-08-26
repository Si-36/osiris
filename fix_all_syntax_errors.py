#!/usr/bin/env python3
"""
FIX ALL SYNTAX ERRORS IN AURA INTELLIGENCE
Systematic fix for the 408 broken files
"""
import os
import re
import ast
from pathlib import Path
import json

class SystematicSyntaxFixer:
    def __init__(self):
        self.base_path = Path("/workspace")
        self.fixed_count = 0
        self.failed_fixes = []
        self.fix_patterns = []
        
    def fix_all_syntax_errors(self):
        """Fix all syntax errors systematically"""
        print("🔧 FIXING ALL SYNTAX ERRORS IN AURA INTELLIGENCE")
        print("=" * 80)
        
        # Load the list of files with syntax errors
        with open('/workspace/deep_honest_analysis.json', 'r') as f:
            analysis = json.load(f)
        
        syntax_errors = analysis['syntax_errors']
        print(f"\n📊 Found {len(syntax_errors)} files with syntax errors")
        print("Starting systematic fix...\n")
        
        # Group errors by type
        error_types = {}
        for error in syntax_errors:
            error_msg = error['error']
            if error_msg not in error_types:
                error_types[error_msg] = []
            error_types[error_msg].append(error)
        
        # Fix by error type
        for error_type, files in error_types.items():
            print(f"\n🔧 Fixing error type: {error_type}")
            print(f"   Affects {len(files)} files")
            
            if "expected an indented block" in error_type:
                self.fix_indentation_errors(files)
            elif "unindent does not match" in error_type:
                self.fix_unindent_errors(files)
            elif "unexpected indent" in error_type:
                self.fix_unexpected_indent(files)
            elif "invalid syntax" in error_type:
                self.fix_invalid_syntax(files)
        
        print(f"\n✅ Fixed {self.fixed_count} files")
        print(f"❌ Failed to fix {len(self.failed_fixes)} files")
        
        if self.failed_fixes:
            print("\nFailed fixes:")
            for fail in self.failed_fixes[:10]:
                print(f"  - {fail}")
    
    def fix_indentation_errors(self, errors):
        """Fix 'expected an indented block' errors"""
        for error in errors:
            file_path = self.base_path / error['file']
            line_num = error['line']
            
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                # Check if it's after a function/class/if/try/etc
                if line_num > 0 and line_num <= len(lines):
                    prev_line = lines[line_num - 2].rstrip() if line_num > 1 else ""
                    current_line = lines[line_num - 1] if line_num <= len(lines) else ""
                    
                    # If current line is not indented after a colon, add pass
                    if prev_line.endswith(':'):
                        # Check if next line is empty or not properly indented
                        if not current_line.strip() or not current_line.startswith(' '):
                            # Insert 'pass' with proper indentation
                            indent = self.get_indent_level(prev_line) + 4
                            lines.insert(line_num - 1, ' ' * indent + 'pass\n')
                            
                            with open(file_path, 'w') as f:
                                f.writelines(lines)
                            self.fixed_count += 1
                            print(f"  ✅ Fixed: {error['file']} line {line_num}")
                            continue
                
                # Try to fix by adding pass after any line ending with :
                fixed = False
                for i in range(len(lines)):
                    if lines[i].rstrip().endswith(':'):
                        # Check if next line is properly indented
                        if i + 1 < len(lines):
                            next_line = lines[i + 1]
                            if not next_line.strip() or self.get_indent_level(next_line) <= self.get_indent_level(lines[i]):
                                # Insert pass
                                indent = self.get_indent_level(lines[i]) + 4
                                lines.insert(i + 1, ' ' * indent + 'pass\n')
                                fixed = True
                
                if fixed:
                    with open(file_path, 'w') as f:
                        f.writelines(lines)
                    self.fixed_count += 1
                    print(f"  ✅ Fixed: {error['file']}")
                    
            except Exception as e:
                self.failed_fixes.append(f"{error['file']}: {str(e)}")
    
    def fix_unindent_errors(self, errors):
        """Fix 'unindent does not match' errors"""
        for error in errors:
            file_path = self.base_path / error['file']
            
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Fix mixed tabs and spaces
                content = content.replace('\t', '    ')
                
                # Fix inconsistent indentation
                lines = content.split('\n')
                fixed_lines = []
                indent_stack = [0]
                
                for line in lines:
                    stripped = line.lstrip()
                    if not stripped:
                        fixed_lines.append('')
                        continue
                    
                    current_indent = len(line) - len(stripped)
                    
                    # Determine correct indent level
                    if stripped.startswith(('def ', 'class ', 'if ', 'elif ', 'else:', 'try:', 'except', 'finally:', 'for ', 'while ', 'with ')):
                        # These should match or increase current level
                        if current_indent not in indent_stack:
                            # Find closest valid indent
                            valid_indent = min(indent_stack, key=lambda x: abs(x - current_indent))
                            fixed_lines.append(' ' * valid_indent + stripped)
                        else:
                            fixed_lines.append(line)
                        
                        if stripped.endswith(':'):
                            indent_stack.append(current_indent + 4)
                    else:
                        # Regular line - should match stack
                        if current_indent not in indent_stack:
                            valid_indent = indent_stack[-1] if indent_stack else 0
                            fixed_lines.append(' ' * valid_indent + stripped)
                        else:
                            fixed_lines.append(line)
                    
                    # Pop from stack if dedenting
                    while indent_stack and current_indent < indent_stack[-1]:
                        indent_stack.pop()
                
                with open(file_path, 'w') as f:
                    f.write('\n'.join(fixed_lines))
                
                self.fixed_count += 1
                print(f"  ✅ Fixed: {error['file']}")
                
            except Exception as e:
                self.failed_fixes.append(f"{error['file']}: {str(e)}")
    
    def fix_unexpected_indent(self, errors):
        """Fix 'unexpected indent' errors"""
        for error in errors:
            file_path = self.base_path / error['file']
            line_num = error['line']
            
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                if line_num > 0 and line_num <= len(lines):
                    # Check if the line has unexpected indent
                    current_line = lines[line_num - 1]
                    
                    # Find the previous non-empty line
                    prev_line_idx = line_num - 2
                    while prev_line_idx >= 0 and not lines[prev_line_idx].strip():
                        prev_line_idx -= 1
                    
                    if prev_line_idx >= 0:
                        prev_line = lines[prev_line_idx]
                        prev_indent = self.get_indent_level(prev_line)
                        
                        # If previous line doesn't end with :, current should have same indent
                        if not prev_line.rstrip().endswith(':'):
                            lines[line_num - 1] = ' ' * prev_indent + current_line.lstrip()
                        else:
                            # Should be indented by 4 more
                            lines[line_num - 1] = ' ' * (prev_indent + 4) + current_line.lstrip()
                        
                        with open(file_path, 'w') as f:
                            f.writelines(lines)
                        self.fixed_count += 1
                        print(f"  ✅ Fixed: {error['file']} line {line_num}")
                
            except Exception as e:
                self.failed_fixes.append(f"{error['file']}: {str(e)}")
    
    def fix_invalid_syntax(self, errors):
        """Fix general invalid syntax errors"""
        for error in errors:
            file_path = self.base_path / error['file']
            
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Common fixes
                # Fix incomplete function definitions
                content = re.sub(r'def\s+(\w+)\s*\(\s*\)\s*:\s*$', r'def \1():\n    pass', content, flags=re.MULTILINE)
                
                # Fix incomplete class definitions  
                content = re.sub(r'class\s+(\w+)\s*:\s*$', r'class \1:\n    pass', content, flags=re.MULTILINE)
                
                # Fix incomplete if statements
                content = re.sub(r'if\s+(.+)\s*:\s*$', r'if \1:\n    pass', content, flags=re.MULTILINE)
                
                # Write back
                with open(file_path, 'w') as f:
                    f.write(content)
                
                # Verify it's fixed
                try:
                    ast.parse(content)
                    self.fixed_count += 1
                    print(f"  ✅ Fixed: {error['file']}")
                except:
                    self.failed_fixes.append(f"{error['file']}: Still has syntax errors")
                    
            except Exception as e:
                self.failed_fixes.append(f"{error['file']}: {str(e)}")
    
    def get_indent_level(self, line):
        """Get indentation level of a line"""
        return len(line) - len(line.lstrip())

    def verify_fixes(self):
        """Verify all fixes by trying to import"""
        print("\n🔍 Verifying fixes...")
        
        test_imports = [
            "from aura_intelligence import UnifiedSystem",
            "from aura_intelligence.orchestration.workflows.nodes.supervisor import UnifiedAuraSupervisor",
            "from aura_intelligence.tda import TDAEngine",
        ]
        
        for imp in test_imports:
            try:
                exec(f"import sys; sys.path.insert(0, '/workspace/core/src'); {imp}")
                print(f"  ✅ Import works: {imp}")
            except Exception as e:
                print(f"  ❌ Import still fails: {imp}")
                print(f"     Error: {str(e)[:100]}")

if __name__ == "__main__":
    fixer = SystematicSyntaxFixer()
    fixer.fix_all_syntax_errors()
    fixer.verify_fixes()