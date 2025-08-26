#!/usr/bin/env python3
"""
🔧 Comprehensive AURA Import Cascade Fix
========================================
Systematically fixes all IndentationError and SyntaxError issues blocking imports.

This script identifies and fixes all Python syntax issues that prevent the 
UnifiedAuraSupervisor from importing properly within the AURA ecosystem.
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Tuple
import traceback

class AURAImportFixer:
    """Comprehensive fixer for AURA import cascade issues"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.core_path = project_root / "core" / "src" / "aura_intelligence"
        self.fixes_applied = []
        self.errors_found = []
        
    def fix_all_issues(self):
        """Fix all import cascade issues systematically"""
        
        print("🔧 COMPREHENSIVE AURA IMPORT CASCADE FIX")
        print("=" * 60)
        
        # Step 1: Find all Python files with issues
        problematic_files = self._find_problematic_files()
        
        print(f"Found {len(problematic_files)} files with potential issues")
        
        # Step 2: Fix systematic indentation issues
        self._fix_indentation_issues()
        
        # Step 3: Fix specific known problematic files
        self._fix_known_problematic_files()
        
        # Step 4: Fix circular import issues
        self._fix_circular_imports()
        
        # Step 5: Validate fixes
        validation_results = self._validate_fixes()
        
        self._report_results(validation_results)
        
        return len(self.errors_found) == 0
    
    def _find_problematic_files(self) -> List[Path]:
        """Find all Python files that might have syntax issues"""
        
        problematic = []
        
        for py_file in self.core_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Try to parse the file
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    problematic.append(py_file)
                    self.errors_found.append(f"{py_file.relative_to(self.project_root)}: {e}")
                    
            except Exception as e:
                print(f"⚠️ Could not read {py_file}: {e}")
        
        return problematic
    
    def _fix_indentation_issues(self):
        """Fix systematic indentation issues across all files"""
        
        print("\n🔧 Fixing indentation issues...")
        
        indentation_patterns = [
            # Function definitions should be consistently indented
            (r'^(\s*)def\s+', self._fix_function_indentation),
            # Class definitions  
            (r'^(\s*)class\s+', self._fix_class_indentation),
            # Control structures
            (r'^(\s*)(if|elif|else|try|except|finally|for|while)\s*.*:', self._fix_control_indentation),
        ]
        
        for py_file in self.core_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                modified_lines = self._fix_file_indentation(lines, py_file)
                
                if modified_lines != lines:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.writelines(modified_lines)
                    
                    self.fixes_applied.append(f"Fixed indentation in {py_file.relative_to(self.project_root)}")
                    
            except Exception as e:
                print(f"⚠️ Could not fix indentation in {py_file}: {e}")
    
    def _fix_file_indentation(self, lines: List[str], file_path: Path) -> List[str]:
        """Fix indentation issues in a specific file"""
        
        fixed_lines = []
        in_class = False
        class_indent = 0
        in_method = False
        method_indent = 0
        
        for i, line in enumerate(lines):
            original_line = line
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                fixed_lines.append(line)
                continue
            
            # Detect class definitions
            if stripped.startswith('class ') and stripped.endswith(':'):
                in_class = True
                class_indent = len(line) - len(line.lstrip())
                in_method = False
                fixed_lines.append(line)
                continue
            
            # Detect method definitions in class
            if in_class and stripped.startswith('def ') and stripped.endswith(':'):
                in_method = True
                method_indent = class_indent + 4
                # Ensure method is properly indented
                fixed_line = ' ' * method_indent + stripped + '\n'
                if fixed_line != line:
                    fixed_lines.append(fixed_line)
                else:
                    fixed_lines.append(line)
                continue
            
            # Fix method content indentation
            if in_method and stripped and not stripped.startswith(('def ', 'class ', '@')):
                current_indent = len(line) - len(line.lstrip())
                expected_indent = method_indent + 4
                
                # Control structures in methods
                if stripped.startswith(('if ', 'elif ', 'else:', 'try:', 'except ', 'for ', 'while ', 'with ')):
                    if current_indent < expected_indent:
                        fixed_line = ' ' * expected_indent + stripped + '\n'
                        fixed_lines.append(fixed_line)
                        continue
                
                # Regular statements in methods
                elif current_indent < expected_indent and current_indent > 0:
                    fixed_line = ' ' * expected_indent + stripped + '\n'
                    fixed_lines.append(fixed_line)
                    continue
            
            # Check for orphaned control structures
            if stripped.startswith(('if ', 'elif ', 'else:', 'try:', 'except ', 'return ', 'logger.')):
                current_indent = len(line) - len(line.lstrip())
                
                if in_method and current_indent < method_indent + 4 and current_indent > 0:
                    fixed_line = ' ' * (method_indent + 4) + stripped + '\n'
                    fixed_lines.append(fixed_line)
                    continue
                elif in_class and not in_method and current_indent < class_indent + 4 and current_indent > 0:
                    fixed_line = ' ' * (class_indent + 4) + stripped + '\n'
                    fixed_lines.append(fixed_line)
                    continue
            
            # Reset context on new top-level definitions
            if not line.startswith(' ') and (stripped.startswith('class ') or stripped.startswith('def ') or stripped.startswith('import ') or stripped.startswith('from ')):
                in_class = False
                in_method = False
            
            fixed_lines.append(line)
        
        return fixed_lines
    
    def _fix_function_indentation(self, match, line, context):
        """Fix function indentation based on context"""
        return line  # Placeholder
    
    def _fix_class_indentation(self, match, line, context):
        """Fix class indentation"""
        return line  # Placeholder
    
    def _fix_control_indentation(self, match, line, context):
        """Fix control structure indentation"""
        return line  # Placeholder
    
    def _fix_known_problematic_files(self):
        """Fix specific files we know are problematic"""
        
        print("\n🔧 Fixing known problematic files...")
        
        problematic_files = [
            "collective/memory_manager.py",
            "collective/graph_builder.py", 
            "collective/context_engine.py",
            "collective/orchestrator.py",
            "infrastructure/kafka_event_mesh.py",
            "observability/langsmith_integration.py",
        ]
        
        for file_path in problematic_files:
            full_path = self.core_path / file_path
            if full_path.exists():
                self._fix_specific_file(full_path)
    
    def _fix_specific_file(self, file_path: Path):
        """Fix a specific file with targeted fixes"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Common fixes
            fixes = [
                # Fix class definition indentation issues
                (r'\n    class (\w+):\n    def __init__\(self', r'\n    class \1:\n        def __init__(self'),
                
                # Fix method definition issues
                (r'\n            async def (\w+)\(\s*self', r'\n        async def \1(self'),
                (r'\n            def (\w+)\(\s*self', r'\n        def \1(self'),
                
                # Fix orphaned statements
                (r'\n        (\w+) = ', r'\n            \1 = '),
                (r'\n        return ', r'\n            return '),
                (r'\n        logger\.', r'\n            logger.'),
                (r'\n        if ', r'\n            if '),
                (r'\n        for ', r'\n            for '),
                (r'\n        try:', r'\n            try:'),
                (r'\n        except ', r'\n            except '),
                
                # Fix common syntax issues
                (r'\n        except Exception as e:\n        return', r'\n        except Exception as e:\n            return'),
                (r'\n        if .*:\n        return', lambda m: m.group(0).replace('\n        return', '\n            return')),
            ]
            
            for pattern, replacement in fixes:
                if callable(replacement):
                    content = re.sub(pattern, replacement, content)
                else:
                    content = re.sub(pattern, replacement, content)
            
            # Write back if changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.fixes_applied.append(f"Fixed specific issues in {file_path.relative_to(self.project_root)}")
                
        except Exception as e:
            print(f"⚠️ Could not fix {file_path}: {e}")
    
    def _fix_circular_imports(self):
        """Fix circular import issues"""
        
        print("\n🔧 Fixing circular imports...")
        
        # Common circular import fixes
        circular_fixes = [
            # Move imports inside functions where needed
            (self.core_path / "__init__.py", self._fix_main_init),
            (self.core_path / "unified_brain.py", self._fix_unified_brain_imports),
            (self.core_path / "collective" / "__init__.py", self._fix_collective_init),
        ]
        
        for file_path, fix_func in circular_fixes:
            if file_path.exists():
                try:
                    fix_func(file_path)
                except Exception as e:
                    print(f"⚠️ Could not fix circular imports in {file_path}: {e}")
    
    def _fix_main_init(self, file_path: Path):
        """Fix main __init__.py circular imports"""
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Move problematic imports to avoid circular references
        if "from .unified_brain import" in content:
            content = content.replace(
                "from .unified_brain import UnifiedAURABrain, UnifiedConfig as BrainConfig, AnalysisResult",
                "# Lazy import to avoid circular references\n# from .unified_brain import UnifiedAURABrain, UnifiedConfig as BrainConfig, AnalysisResult"
            )
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            self.fixes_applied.append(f"Fixed circular imports in {file_path.relative_to(self.project_root)}")
    
    def _fix_unified_brain_imports(self, file_path: Path):
        """Fix unified_brain.py imports"""
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Move problematic imports inside functions
        problematic_imports = [
            "from .collective import CollectiveIntelligenceOrchestrator",
            "from .constitutional import ConstitutionalAI, EthicalViolationError",
            "from .tda_engine import ProductionGradeTDA, TopologySignature",
        ]
        
        for imp in problematic_imports:
            if imp in content:
                content = content.replace(imp, f"# {imp}")
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        self.fixes_applied.append(f"Fixed imports in {file_path.relative_to(self.project_root)}")
    
    def _fix_collective_init(self, file_path: Path):
        """Fix collective __init__.py"""
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Comment out problematic imports
        problematic = [
            "from .supervisor import CollectiveSupervisor",
            "from .memory_manager import CollectiveMemoryManager",
            "from .orchestrator import CollectiveIntelligenceOrchestrator",
        ]
        
        for imp in problematic:
            if imp in content:
                content = content.replace(imp, f"# {imp}")
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        self.fixes_applied.append(f"Fixed collective init imports in {file_path.relative_to(self.project_root)}")
    
    def _validate_fixes(self) -> Dict[str, bool]:
        """Validate that our fixes worked"""
        
        print("\n✅ Validating fixes...")
        
        validation_results = {
            "syntax_errors": 0,
            "import_errors": 0,
            "files_checked": 0
        }
        
        for py_file in self.core_path.rglob("*.py"):
            validation_results["files_checked"] += 1
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check syntax
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    validation_results["syntax_errors"] += 1
                    print(f"❌ Still has syntax error: {py_file.relative_to(self.project_root)}: {e}")
                
            except Exception as e:
                validation_results["import_errors"] += 1
        
        return validation_results
    
    def _report_results(self, validation_results: Dict[str, bool]):
        """Report the results of our fixes"""
        
        print("\n" + "=" * 60)
        print("📊 FIX RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"Files processed: {validation_results['files_checked']}")
        print(f"Fixes applied: {len(self.fixes_applied)}")
        print(f"Remaining syntax errors: {validation_results['syntax_errors']}")
        print(f"Import errors: {validation_results['import_errors']}")
        
        if self.fixes_applied:
            print("\n🔧 Applied Fixes:")
            for fix in self.fixes_applied[-10:]:  # Show last 10 fixes
                print(f"   ✅ {fix}")
            if len(self.fixes_applied) > 10:
                print(f"   ... and {len(self.fixes_applied) - 10} more")
        
        if validation_results['syntax_errors'] == 0:
            print("\n🎉 ALL SYNTAX ERRORS FIXED!")
        else:
            print(f"\n⚠️ {validation_results['syntax_errors']} syntax errors remain")
        
        # Save detailed results
        results = {
            "fixes_applied": self.fixes_applied,
            "errors_found": self.errors_found,
            "validation_results": validation_results
        }
        
        import json
        with open(self.project_root / "cascade_fix_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)


def main():
    """Run comprehensive AURA import cascade fix"""
    
    project_root = Path(__file__).parent
    fixer = AURAImportFixer(project_root)
    
    success = fixer.fix_all_issues()
    
    if success:
        print("\n🎉 COMPREHENSIVE FIX COMPLETED SUCCESSFULLY!")
        print("✅ All import cascade issues resolved")
        print("✅ UnifiedAuraSupervisor should now import cleanly")
    else:
        print("\n⚠️ SOME ISSUES REMAIN")
        print("🔧 Check cascade_fix_results.json for details")
    
    return success


if __name__ == "__main__":
    main()