#!/usr/bin/env python3
"""
Comprehensive Project Indexer V2 - Find all components and potential issues
"""

import os
import ast
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

class ProjectIndexer:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.stats = defaultdict(int)
        self.real_implementations = []
        self.dummy_implementations = []
        self.import_errors = []
        self.missing_features = []
        self.external_dependencies = set()
        self.api_endpoints = []
        self.gpu_features = []
        self.distributed_features = []
        self.production_features = []
        self.test_files = []
        self.docker_files = []
        self.config_files = []
        self.ml_models = []
        self.data_pipelines = []
        
    def index_project(self):
        """Index the entire project"""
        print("🔍 Starting Comprehensive Project Index...")
        print("=" * 80)
        
        # Scan all Python files
        for py_file in self.base_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            self.analyze_python_file(py_file)
        
        # Scan other important files
        self.scan_docker_files()
        self.scan_config_files()
        self.scan_requirements()
        
        # Generate report
        self.generate_report()
    
    def analyze_python_file(self, filepath: Path):
        """Analyze a single Python file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip empty files
            if not content.strip():
                return
            
            # Parse AST
            try:
                tree = ast.parse(content)
                self.analyze_ast(tree, filepath, content)
            except SyntaxError as e:
                self.import_errors.append({
                    'file': str(filepath),
                    'error': f"Syntax Error: {e}",
                    'line': e.lineno
                })
            
            # Check for patterns in content
            self.check_patterns(content, filepath)
            
        except Exception as e:
            self.import_errors.append({
                'file': str(filepath),
                'error': f"Read Error: {e}"
            })
    
    def analyze_ast(self, tree: ast.AST, filepath: Path, content: str):
        """Analyze the AST of a file"""
        for node in ast.walk(tree):
            # Count node types
            self.stats[type(node).__name__] += 1
            
            # Check for real vs dummy implementations
            if isinstance(node, ast.FunctionDef):
                self.check_implementation(node, filepath, content)
            
            # Check for external dependencies
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.check_imports(node, filepath)
            
            # Check for API endpoints
            if isinstance(node, ast.FunctionDef):
                self.check_api_endpoints(node, filepath)
            
            # Check for GPU/CUDA code
            if isinstance(node, ast.Name) and 'cuda' in node.id.lower():
                self.gpu_features.append({
                    'file': str(filepath),
                    'feature': node.id
                })
            
            # Check for distributed features
            if isinstance(node, ast.Name) and any(x in node.id.lower() for x in ['ray', 'dask', 'spark']):
                self.distributed_features.append({
                    'file': str(filepath),
                    'feature': node.id
                })
    
    def check_implementation(self, node: ast.FunctionDef, filepath: Path, content: str):
        """Check if function is real or dummy"""
        # Get function body
        if len(node.body) == 1:
            stmt = node.body[0]
            
            # Check for pass/NotImplementedError/TODO
            if isinstance(stmt, ast.Pass):
                self.dummy_implementations.append({
                    'file': str(filepath),
                    'function': node.name,
                    'type': 'pass'
                })
            elif isinstance(stmt, ast.Raise) and 'NotImplementedError' in ast.unparse(stmt):
                self.dummy_implementations.append({
                    'file': str(filepath),
                    'function': node.name,
                    'type': 'NotImplementedError'
                })
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if 'TODO' in str(stmt.value.value) or 'FIXME' in str(stmt.value.value):
                    self.dummy_implementations.append({
                        'file': str(filepath),
                        'function': node.name,
                        'type': 'TODO'
                    })
        
        # Check for real implementations
        if len(node.body) > 2 or (len(node.body) > 1 and not isinstance(node.body[0], ast.Expr)):
            # Check for specific patterns that indicate real implementation
            body_str = ast.unparse(node)
            if any(pattern in body_str for pattern in [
                'torch.', 'np.', 'faiss.', 'sklearn.', 'ray.', 'async def',
                'RipsComplex', 'LiquidNN', 'KNNIndex', 'compute', 'train', 'predict'
            ]):
                self.real_implementations.append({
                    'file': str(filepath),
                    'function': node.name,
                    'lines': len(node.body)
                })
    
    def check_imports(self, node, filepath: Path):
        """Check imports for external dependencies"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.external_dependencies.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            self.external_dependencies.add(node.module.split('.')[0])
    
    def check_api_endpoints(self, node: ast.FunctionDef, filepath: Path):
        """Check for API endpoints"""
        # Look for decorators
        for decorator in node.decorator_list:
            dec_str = ast.unparse(decorator)
            if any(method in dec_str for method in ['@app.', '@router.', '.get(', '.post(', '.put(', '.delete(']):
                self.api_endpoints.append({
                    'file': str(filepath),
                    'function': node.name,
                    'decorator': dec_str
                })
    
    def check_patterns(self, content: str, filepath: Path):
        """Check for specific patterns in content"""
        # Test files
        if 'test_' in filepath.name or '_test.py' in str(filepath):
            self.test_files.append(str(filepath))
        
        # ML Models
        if any(pattern in content for pattern in ['class.*Model', 'nn.Module', 'tf.keras.Model']):
            self.ml_models.append(str(filepath))
        
        # Data pipelines
        if any(pattern in content for pattern in ['Pipeline', 'DataLoader', 'Dataset', 'transform']):
            self.data_pipelines.append(str(filepath))
        
        # Production features
        if any(pattern in content for pattern in ['@cache', 'redis', 'kafka', 'production', 'monitoring']):
            self.production_features.append(str(filepath))
    
    def scan_docker_files(self):
        """Scan for Docker files"""
        for dockerfile in self.base_path.rglob("Dockerfile*"):
            self.docker_files.append(str(dockerfile))
        for compose in self.base_path.rglob("docker-compose*.yml"):
            self.docker_files.append(str(compose))
    
    def scan_config_files(self):
        """Scan for configuration files"""
        patterns = ["*.yml", "*.yaml", "*.json", "*.toml", "*.ini", "*.env*"]
        for pattern in patterns:
            for config in self.base_path.rglob(pattern):
                if "__pycache__" not in str(config):
                    self.config_files.append(str(config))
    
    def scan_requirements(self):
        """Scan requirements files"""
        for req in self.base_path.rglob("requirements*.txt"):
            self.analyze_requirements(req)
    
    def analyze_requirements(self, req_file: Path):
        """Analyze a requirements file"""
        try:
            with open(req_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        pkg = line.split('==')[0].split('>=')[0].split('<')[0].strip()
                        self.external_dependencies.add(pkg)
        except:
            pass
    
    def generate_report(self):
        """Generate comprehensive report"""
        report = {
            'summary': {
                'total_files': sum(1 for _ in self.base_path.rglob("*.py") if "__pycache__" not in str(_)),
                'real_implementations': len(self.real_implementations),
                'dummy_implementations': len(self.dummy_implementations),
                'import_errors': len(self.import_errors),
                'api_endpoints': len(self.api_endpoints),
                'test_files': len(self.test_files),
                'docker_files': len(self.docker_files),
                'config_files': len(self.config_files),
                'ml_models': len(self.ml_models),
                'gpu_features': len(self.gpu_features),
                'distributed_features': len(self.distributed_features)
            },
            'external_dependencies': sorted(list(self.external_dependencies)),
            'dummy_implementations': self.dummy_implementations[:20],  # Top 20
            'real_implementations': self.real_implementations[:20],   # Top 20
            'import_errors': self.import_errors,
            'api_endpoints': self.api_endpoints,
            'gpu_features': self.gpu_features[:10],
            'distributed_features': self.distributed_features[:10],
            'production_features': self.production_features[:10],
            'test_coverage': {
                'test_files': len(self.test_files),
                'examples': self.test_files[:10]
            }
        }
        
        # Print report
        print("\n📊 PROJECT INDEX REPORT")
        print("=" * 80)
        
        print("\n📈 SUMMARY:")
        for key, value in report['summary'].items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        print("\n⚠️  DUMMY IMPLEMENTATIONS:")
        for dummy in report['dummy_implementations'][:10]:
            print(f"   - {dummy['file'].split('/')[-1]}: {dummy['function']} ({dummy['type']})")
        
        print("\n✅ REAL IMPLEMENTATIONS:")
        for real in report['real_implementations'][:10]:
            print(f"   - {real['file'].split('/')[-1]}: {real['function']} ({real['lines']} lines)")
        
        print("\n🚨 IMPORT ERRORS:")
        for error in report['import_errors'][:5]:
            print(f"   - {error['file'].split('/')[-1]}: {error['error']}")
        
        print("\n🌐 API ENDPOINTS:")
        for endpoint in report['api_endpoints'][:10]:
            print(f"   - {endpoint['file'].split('/')[-1]}: {endpoint['function']}")
        
        print("\n🚀 GPU FEATURES:")
        for gpu in report['gpu_features'][:5]:
            print(f"   - {gpu['file'].split('/')[-1]}: {gpu['feature']}")
        
        print("\n📦 KEY DEPENDENCIES:")
        key_deps = [d for d in report['external_dependencies'] if d in [
            'torch', 'tensorflow', 'numpy', 'pandas', 'fastapi', 'redis', 
            'kafka', 'ray', 'faiss', 'neo4j', 'langchain', 'transformers'
        ]]
        print(f"   {', '.join(key_deps)}")
        
        # Save full report
        with open('/workspace/project_index_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n💾 Full report saved to: /workspace/project_index_report.json")
        
        # Identify missing components
        self.identify_missing_components(report)
    
    def identify_missing_components(self, report):
        """Identify what might be missing"""
        print("\n🔍 POTENTIAL MISSING COMPONENTS:")
        
        # Check for missing test coverage
        if report['test_coverage']['test_files'] < report['summary']['total_files'] * 0.1:
            print("   ⚠️  Low test coverage - only {:.1%} of files have tests".format(
                report['test_coverage']['test_files'] / report['summary']['total_files']
            ))
        
        # Check for missing Docker setup
        if len(self.docker_files) == 0:
            print("   ⚠️  No Docker files found - containerization missing")
        
        # Check for missing CI/CD
        ci_files = list(self.base_path.rglob(".github/workflows/*.yml"))
        if not ci_files:
            print("   ⚠️  No CI/CD workflows found")
        
        # Check for missing documentation
        docs = list(self.base_path.rglob("docs/*.md")) + list(self.base_path.rglob("*.md"))
        if len(docs) < 5:
            print("   ⚠️  Limited documentation found")
        
        # Check for missing monitoring
        if 'prometheus' not in report['external_dependencies'] and 'grafana' not in str(self.config_files):
            print("   ⚠️  No monitoring setup found (Prometheus/Grafana)")
        
        # Check for missing security
        if not any('security' in str(f).lower() for f in self.config_files):
            print("   ⚠️  No security configuration found")

if __name__ == "__main__":
    indexer = ProjectIndexer("/workspace")
    indexer.index_project()