#!/usr/bin/env python3
"""
Comprehensive AURA Project Indexer
Indexes ALL files and understands the real structure
"""
import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import ast

class ComprehensiveProjectIndexer:
    def __init__(self):
        self.base_path = Path("/workspace/core/src/aura_intelligence")
        self.stats = defaultdict(int)
        self.components = defaultdict(list)
        self.imports = defaultdict(set)
        self.errors = []
        self.supervisor_files = []
        self.tda_files = []
        self.lnn_files = []
        self.real_implementations = []
        self.mock_implementations = []
        
    def index_project(self):
        """Index entire project structure"""
        print("🔍 COMPREHENSIVE AURA PROJECT INDEXING")
        print("=" * 70)
        
        # Index all Python files
        all_files = list(self.base_path.rglob("*.py"))
        print(f"\n📁 Total Python files: {len(all_files)}")
        
        for file_path in all_files:
            self.analyze_file(file_path)
        
        # Analyze patterns
        self.analyze_patterns()
        
        # Generate report
        self.generate_report()
        
    def analyze_file(self, file_path):
        """Analyze a single Python file"""
        relative_path = file_path.relative_to(self.base_path)
        self.stats['total_files'] += 1
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Count lines
            lines = content.split('\n')
            self.stats['total_lines'] += len(lines)
            
            # Check for key patterns
            if 'supervisor' in str(relative_path).lower() or 'Supervisor' in content:
                self.supervisor_files.append(str(relative_path))
                
            if 'tda' in str(relative_path).lower() or 'topolog' in content.lower():
                self.tda_files.append(str(relative_path))
                
            if 'lnn' in str(relative_path).lower() or 'liquid' in content.lower():
                self.lnn_files.append(str(relative_path))
            
            # Check for real vs mock implementations
            if any(pattern in content for pattern in ['# TODO', 'pass  # TODO', 'raise NotImplementedError', 'return {}', 'return []', 'return None  # TODO']):
                self.mock_implementations.append(str(relative_path))
            elif any(pattern in content for pattern in ['async def', 'torch.', 'np.', 'sklearn.', 'return result']):
                self.real_implementations.append(str(relative_path))
            
            # Try to parse AST
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        self.components['classes'].append(f"{relative_path}:{node.name}")
                    elif isinstance(node, ast.FunctionDef):
                        self.components['functions'].append(f"{relative_path}:{node.name}")
                    elif isinstance(node, ast.AsyncFunctionDef):
                        self.components['async_functions'].append(f"{relative_path}:{node.name}")
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            self.imports[str(relative_path)].add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            self.imports[str(relative_path)].add(node.module)
            except SyntaxError as e:
                self.errors.append(f"{relative_path}: {e}")
                self.stats['syntax_errors'] += 1
                
        except Exception as e:
            self.errors.append(f"{relative_path}: {e}")
            self.stats['read_errors'] += 1
    
    def analyze_patterns(self):
        """Analyze patterns in the codebase"""
        # Find main entry points
        self.entry_points = []
        for file_path in self.base_path.rglob("*.py"):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                if 'if __name__ == "__main__"' in content:
                    self.entry_points.append(str(file_path.relative_to(self.base_path)))
            except:
                pass
                
        # Find API files
        self.api_files = []
        for file_path in self.base_path.rglob("*.py"):
            try:
                relative = str(file_path.relative_to(self.base_path))
                if 'api' in relative.lower() or 'fastapi' in open(file_path).read():
                    self.api_files.append(relative)
            except:
                pass
    
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n📊 PROJECT STATISTICS")
        print("=" * 70)
        print(f"Total files: {self.stats['total_files']}")
        print(f"Total lines: {self.stats['total_lines']:,}")
        print(f"Syntax errors: {self.stats['syntax_errors']}")
        print(f"Read errors: {self.stats['read_errors']}")
        
        print(f"\n🏗️ COMPONENTS")
        print(f"Classes: {len(self.components['classes'])}")
        print(f"Functions: {len(self.components['functions'])}")
        print(f"Async functions: {len(self.components['async_functions'])}")
        
        print(f"\n🎯 KEY FILES")
        print(f"Supervisor files: {len(self.supervisor_files)}")
        print(f"TDA files: {len(self.tda_files)}")
        print(f"LNN files: {len(self.lnn_files)}")
        print(f"API files: {len(self.api_files)}")
        print(f"Entry points: {len(self.entry_points)}")
        
        print(f"\n💡 IMPLEMENTATION STATUS")
        print(f"Real implementations: {len(self.real_implementations)}")
        print(f"Mock/TODO implementations: {len(self.mock_implementations)}")
        
        print(f"\n🔝 TOP SUPERVISOR FILES")
        for f in self.supervisor_files[:10]:
            print(f"  - {f}")
            
        print(f"\n🧮 TOP TDA FILES")
        for f in self.tda_files[:10]:
            print(f"  - {f}")
            
        print(f"\n🧠 TOP LNN FILES")
        for f in self.lnn_files[:10]:
            print(f"  - {f}")
            
        print(f"\n🚀 ENTRY POINTS")
        for f in self.entry_points[:10]:
            print(f"  - {f}")
            
        print(f"\n🌐 API FILES")
        for f in self.api_files[:10]:
            print(f"  - {f}")
            
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)} total)")
            for e in self.errors[:10]:
                print(f"  - {e}")
        
        # Save detailed report
        report = {
            'stats': dict(self.stats),
            'supervisor_files': self.supervisor_files,
            'tda_files': self.tda_files,
            'lnn_files': self.lnn_files,
            'api_files': self.api_files,
            'entry_points': self.entry_points,
            'real_implementations': self.real_implementations[:50],
            'mock_implementations': self.mock_implementations[:50],
            'errors': self.errors[:50],
            'total_classes': len(self.components['classes']),
            'total_functions': len(self.components['functions']),
            'total_async_functions': len(self.components['async_functions'])
        }
        
        with open('/workspace/aura_project_index.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: /workspace/aura_project_index.json")

if __name__ == "__main__":
    indexer = ComprehensiveProjectIndexer()
    indexer.index_project()