#!/usr/bin/env python3
"""
Fix all syntax errors in the project using best practices
"""

import re
from pathlib import Path
from typing import List, Tuple

class SyntaxErrorFixer:
    def __init__(self):
        self.fixed_count = 0
        self.error_files = [
            # From the analysis
            "/workspace/core/src/aura_intelligence/streaming/kafka_integration.py",
            "/workspace/core/src/aura_intelligence/agents/council/multi_agent/supervisor.py",
            "/workspace/core/src/aura_intelligence/collective/memory_manager.py",
            "/workspace/core/src/aura_intelligence/agents/base_classes/instrumentation.py",
            "/workspace/src/aura_intelligence_integrations/adapters/base_classes/agent.py",
            "/workspace/orchestration/distributed/crewai/tasks/planning/activities/main.py",
            "/workspace/utilities/fix_real_issues.py",
            "/workspace/core/src/aura_intelligence/orchestration/durable/test_workflow_observability.py",
            "/workspace/core/src/aura_intelligence/agents/nodes/observer.py",
            "/workspace/core/src/aura_intelligence/agents/nodes/supervisor.py",
            "/workspace/core/src/aura_intelligence/core/fixed_components.py"
        ]
    
    def fix_file(self, filepath: str) -> bool:
        """Fix syntax errors in a single file"""
        if not Path(filepath).exists():
            print(f"⚠️  File not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            fixed_lines = self.fix_indentation_errors(lines)
            fixed_lines = self.fix_docstring_indentation(fixed_lines)
            fixed_lines = self.fix_try_blocks(fixed_lines)
            fixed_lines = self.fix_unterminated_strings(fixed_lines)
            
            # Write back if changes were made
            if fixed_lines != lines:
                with open(filepath, 'w') as f:
                    f.writelines(fixed_lines)
                self.fixed_count += 1
                print(f"✅ Fixed: {Path(filepath).name}")
                return True
            else:
                print(f"⏭️  No changes needed: {Path(filepath).name}")
                return False
                
        except Exception as e:
            print(f"❌ Error fixing {filepath}: {e}")
            return False
    
    def fix_indentation_errors(self, lines: List[str]) -> List[str]:
        """Fix missing indentation after colons"""
        fixed = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            fixed.append(line)
            
            # Check if line ends with colon (function, class, if, try, etc.)
            if line.strip() and line.strip().endswith(':'):
                # Check next line
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    
                    # If next line is not indented and not empty
                    if next_line.strip() and not next_line.startswith(' ') and not next_line.startswith('\t'):
                        # Determine indentation level
                        current_indent = len(line) - len(line.lstrip())
                        new_indent = current_indent + 4
                        
                        # Skip the original unindented line
                        i += 1
                        
                        # Add properly indented line
                        fixed.append(' ' * new_indent + next_line.strip() + '\n')
                        continue
            
            i += 1
        
        return fixed
    
    def fix_docstring_indentation(self, lines: List[str]) -> List[str]:
        """Fix docstring indentation specifically"""
        fixed = []
        
        for i, line in enumerate(lines):
            # Check for function/method definition
            if re.match(r'\s*def\s+\w+\s*\(.*\)\s*.*:\s*$', line):
                fixed.append(line)
                
                # Check if next line is an unindented docstring
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if next_line.strip().startswith('"""') and not next_line.startswith(' '):
                        # Get function indentation
                        func_indent = len(line) - len(line.lstrip())
                        docstring_indent = func_indent + 4
                        
                        # Fix docstring line
                        fixed.append(' ' * docstring_indent + next_line.strip() + '\n')
                        continue
            
            # Skip if we already processed this line
            if i > 0 and lines[i-1].strip().endswith(':') and line.strip().startswith('"""'):
                continue
                
            fixed.append(line)
        
        return fixed
    
    def fix_try_blocks(self, lines: List[str]) -> List[str]:
        """Fix try/except block indentation"""
        fixed = []
        
        for i, line in enumerate(lines):
            fixed.append(line)
            
            # Check for try: statement
            if line.strip() == 'try:':
                # Ensure next line is indented
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if next_line.strip() and not next_line.startswith(' '):
                        # Get try indentation
                        try_indent = len(line) - len(line.lstrip())
                        content_indent = try_indent + 4
                        
                        # Add placeholder if needed
                        fixed.append(' ' * content_indent + 'pass  # TODO: Add implementation\n')
        
        return fixed
    
    def fix_unterminated_strings(self, lines: List[str]) -> List[str]:
        """Fix unterminated triple-quoted strings"""
        fixed = []
        in_triple_quote = False
        quote_char = None
        
        for line in lines:
            # Check for triple quotes
            if '"""' in line:
                count = line.count('"""')
                if count % 2 == 1:  # Odd number means state change
                    in_triple_quote = not in_triple_quote
                    quote_char = '"""'
            elif "'''" in line:
                count = line.count("'''")
                if count % 2 == 1:
                    in_triple_quote = not in_triple_quote
                    quote_char = "'''"
            
            fixed.append(line)
        
        # If still in triple quote at end, close it
        if in_triple_quote:
            fixed.append(quote_char + '\n')
        
        return fixed
    
    def fix_all(self):
        """Fix all known syntax error files"""
        print("🔧 FIXING SYNTAX ERRORS")
        print("=" * 60)
        
        for filepath in self.error_files:
            self.fix_file(filepath)
        
        print(f"\n✅ Fixed {self.fixed_count} files")

if __name__ == "__main__":
    fixer = SyntaxErrorFixer()
    fixer.fix_all()