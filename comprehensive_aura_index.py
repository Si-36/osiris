#!/usr/bin/env python3
"""
Comprehensive AURA Project Indexer
- Index ALL files
- Find ALL connections
- Identify REAL problems
- No fake testing
"""

import os
import ast
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import re

class AURAIndexer:
    def __init__(self):
        self.all_files = []
        self.syntax_errors = []
        self.imports = defaultdict(set)
        self.dependencies = defaultdict(set)
        self.api_keys_needed = set()
        self.test_files = []
        self.mock_implementations = []
        self.real_implementations = []
        self.research_mentions = []
        
    def index_project(self, root_path: str):
        """Index entire project"""
        print(f"🔍 Indexing ENTIRE AURA project from: {root_path}")
        
        # Walk through ALL directories
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Skip some directories
            if any(skip in dirpath for skip in ['.git', '__pycache__', 'venv', '.pytest_cache']):
                continue
                
            for filename in filenames:
                if filename.endswith('.py'):
                    filepath = os.path.join(dirpath, filename)
                    self.all_files.append(filepath)
                    self._analyze_file(filepath)
                elif filename.endswith('.md'):
                    filepath = os.path.join(dirpath, filename)
                    self._analyze_markdown(filepath)
                    
    def _analyze_file(self, filepath: str):
        """Analyze a Python file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check syntax
            try:
                tree = ast.parse(content)
                self._extract_imports(tree, filepath)
                self._check_implementations(content, filepath)
            except SyntaxError as e:
                self.syntax_errors.append({
                    'file': filepath,
                    'line': e.lineno,
                    'error': e.msg
                })
                
            # Check for API keys
            if any(key in content for key in ['api_key', 'API_KEY', 'token', 'TOKEN', 'credentials']):
                self._check_api_requirements(content, filepath)
                
            # Check if test file
            if 'test_' in filename or '_test.py' in filename:
                self.test_files.append(filepath)
                
            # Check for research mentions
            if any(term in content for term in ['2025', 'state-of-the-art', 'latest', 'research']):
                self.research_mentions.append(filepath)
                
        except Exception as e:
            pass
            
    def _extract_imports(self, tree: ast.AST, filepath: str):
        """Extract imports from AST"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    self.imports[filepath].add(name.name)
                    self.dependencies[name.name].add(filepath)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.imports[filepath].add(node.module)
                    self.dependencies[node.module].add(filepath)
                    
    def _check_implementations(self, content: str, filepath: str):
        """Check if file has real or mock implementations"""
        if any(term in content for term in ['mock', 'Mock', 'TODO', 'placeholder', 'not implemented']):
            self.mock_implementations.append(filepath)
        elif any(term in content for term in ['async def', 'class', 'def']) and len(content) > 500:
            self.real_implementations.append(filepath)
            
    def _check_api_requirements(self, content: str, filepath: str):
        """Check what API keys are needed"""
        # Common API patterns
        apis = {
            'openai': ['openai', 'gpt', 'chatgpt'],
            'neo4j': ['neo4j', 'graph_database', 'bolt://'],
            'redis': ['redis', 'Redis('],
            'kafka': ['kafka', 'bootstrap_servers'],
            'langchain': ['langchain', 'llm'],
            'pinecone': ['pinecone', 'vector_db'],
            'weaviate': ['weaviate', 'vector_store'],
            'cohere': ['cohere', 'co.Client'],
            'anthropic': ['anthropic', 'claude'],
            'huggingface': ['transformers', 'AutoModel']
        }
        
        for api, patterns in apis.items():
            if any(pattern in content for pattern in patterns):
                self.api_keys_needed.add(api)
                
    def _analyze_markdown(self, filepath: str):
        """Analyze markdown files for research and documentation"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Look for important sections
            if any(term in content.lower() for term in ['research', '2025', 'architecture', 'design']):
                self.research_mentions.append(filepath)
        except:
            pass
            
    def generate_report(self) -> Dict:
        """Generate comprehensive report"""
        # Find most imported modules
        most_imported = sorted(
            [(module, len(files)) for module, files in self.dependencies.items()],
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        # Find critical files (imported by many others)
        file_import_count = defaultdict(int)
        for filepath, imports in self.imports.items():
            for imp in imports:
                if 'aura_intelligence' in imp:
                    file_import_count[imp] += 1
                    
        critical_modules = sorted(
            file_import_count.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        return {
            'total_files': len(self.all_files),
            'syntax_errors': len(self.syntax_errors),
            'error_files': self.syntax_errors[:10],  # Top 10
            'test_files': len(self.test_files),
            'mock_implementations': len(self.mock_implementations),
            'real_implementations': len(self.real_implementations),
            'api_keys_needed': list(self.api_keys_needed),
            'most_imported': most_imported,
            'critical_modules': critical_modules,
            'research_files': self.research_mentions[:10]
        }

# Run indexer
indexer = AURAIndexer()

# Index different paths
paths_to_index = [
    'core/src/aura_intelligence',
    'looklooklook.md',
    'README.md'
]

for path in paths_to_index:
    if os.path.exists(path):
        indexer.index_project(path)
        
# Generate report
report = indexer.generate_report()

# Save report
with open('AURA_COMPLETE_INDEX.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n📊 AURA Complete Index Report")
print("=" * 60)
print(f"Total Python files: {report['total_files']}")
print(f"Files with syntax errors: {report['syntax_errors']}")
print(f"Test files: {report['test_files']}")
print(f"Mock implementations: {report['mock_implementations']}")
print(f"Real implementations: {report['real_implementations']}")
print(f"\n🔑 API Keys Needed: {', '.join(report['api_keys_needed'])}")
print(f"\n📚 Critical Modules (most imported):")
for module, count in report['critical_modules'][:5]:
    print(f"  - {module}: imported by {count} files")

print("\n🔍 Now analyzing what's REALLY important...")