#!/usr/bin/env python3
"""
Comprehensive AURA Intelligence Project Analyzer
===============================================
Analyzes ALL components to create a complete integration map
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Set
from datetime import datetime
import re


class ComprehensiveProjectAnalyzer:
    """Analyzes the entire AURA Intelligence project comprehensively"""
    
    def __init__(self):
        self.base_path = Path("/workspace/core/src/aura_intelligence")
        self.components = {}
        self.real_implementations = {}
        self.integrations = {}
        self.microservices = {}
        self.advanced_features = {}
        
    def analyze_project(self):
        """Analyze the entire project structure"""
        
        print("🔍 COMPREHENSIVE AURA INTELLIGENCE PROJECT ANALYSIS")
        print("=" * 80)
        
        # 1. Analyze each major component
        for directory in sorted(self.base_path.iterdir()):
            if directory.is_dir() and not directory.name.startswith(('__', '.')):
                self.analyze_component(directory)
        
        # 2. Find all real implementations
        self.find_real_implementations()
        
        # 3. Analyze microservices
        self.analyze_microservices()
        
        # 4. Generate comprehensive report
        self.generate_comprehensive_report()
        
    def analyze_component(self, component_path: Path):
        """Analyze a single component directory"""
        
        component_name = component_path.name
        component_info = {
            'path': str(component_path),
            'files': [],
            'classes': [],
            'functions': [],
            'real_implementations': [],
            'integrations': [],
            'has_tests': False,
            'has_docs': False,
            'key_features': []
        }
        
        # Analyze all Python files
        for py_file in component_path.rglob("*.py"):
            if '__pycache__' not in str(py_file):
                file_info = self.analyze_python_file(py_file)
                component_info['files'].append(file_info)
                
                # Extract key information
                if 'real' in py_file.name.lower() or 'impl' in py_file.name.lower():
                    component_info['real_implementations'].append(str(py_file))
                
                if 'test' in py_file.name.lower():
                    component_info['has_tests'] = True
                
                component_info['classes'].extend(file_info.get('classes', []))
                component_info['functions'].extend(file_info.get('functions', []))
        
        # Check for documentation
        for doc_file in component_path.rglob("*.md"):
            component_info['has_docs'] = True
            break
        
        # Identify key features based on content
        component_info['key_features'] = self.identify_key_features(component_info)
        
        self.components[component_name] = component_info
        
    def analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a Python file for classes, functions, and patterns"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            file_info = {
                'path': str(file_path),
                'classes': [],
                'functions': [],
                'imports': [],
                'has_async': False,
                'has_gpu': False,
                'has_real_impl': False,
                'patterns': []
            }
            
            # Analyze AST
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    file_info['classes'].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    file_info['functions'].append(node.name)
                    if isinstance(node, ast.AsyncFunctionDef):
                        file_info['has_async'] = True
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        file_info['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        file_info['imports'].append(node.module)
            
            # Check for patterns
            if 'torch' in content or 'cuda' in content.lower():
                file_info['has_gpu'] = True
            
            if 'real' in content.lower() and 'implementation' in content.lower():
                file_info['has_real_impl'] = True
            
            # Identify design patterns
            patterns = []
            if re.search(r'class.*Factory', content):
                patterns.append('Factory')
            if re.search(r'class.*Singleton', content):
                patterns.append('Singleton')
            if re.search(r'class.*Observer', content):
                patterns.append('Observer')
            if re.search(r'@ray\.remote', content):
                patterns.append('Distributed')
            
            file_info['patterns'] = patterns
            
            return file_info
            
        except Exception as e:
            return {'path': str(file_path), 'error': str(e)}
    
    def find_real_implementations(self):
        """Find all real implementations across the project"""
        
        real_impl_patterns = [
            'real_', '_real', 'Real', 'impl', 'engine', 'system',
            'production', 'unified', 'complete', 'enhanced'
        ]
        
        for component_name, component_info in self.components.items():
            for file_info in component_info['files']:
                file_path = file_info['path']
                
                # Check if it's a real implementation
                is_real = any(pattern in file_path for pattern in real_impl_patterns)
                
                if is_real or file_info.get('has_real_impl'):
                    self.real_implementations[file_path] = {
                        'component': component_name,
                        'classes': file_info.get('classes', []),
                        'has_gpu': file_info.get('has_gpu', False),
                        'has_async': file_info.get('has_async', False),
                        'patterns': file_info.get('patterns', [])
                    }
    
    def analyze_microservices(self):
        """Analyze microservices directory"""
        
        microservices_path = Path("/workspace/aura-microservices")
        if microservices_path.exists():
            for service_dir in microservices_path.iterdir():
                if service_dir.is_dir() and not service_dir.name.startswith('.'):
                    self.microservices[service_dir.name] = {
                        'path': str(service_dir),
                        'has_dockerfile': (service_dir / 'Dockerfile').exists(),
                        'has_api': (service_dir / 'main.py').exists() or (service_dir / 'app.py').exists(),
                        'port': self.extract_port(service_dir)
                    }
    
    def extract_port(self, service_dir: Path) -> int:
        """Extract port number from service files"""
        
        # Check common files for port configuration
        for file_name in ['main.py', 'app.py', 'config.py', 'Dockerfile']:
            file_path = service_dir / file_name
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        # Look for port patterns
                        port_match = re.search(r'port["\s=:]+(\d{4,5})', content, re.IGNORECASE)
                        if port_match:
                            return int(port_match.group(1))
                except:
                    pass
        return 0
    
    def identify_key_features(self, component_info: Dict) -> List[str]:
        """Identify key features of a component"""
        
        features = []
        
        # Check classes for key patterns
        for class_name in component_info['classes']:
            if 'TDA' in class_name or 'Topolog' in class_name:
                features.append('Topological Data Analysis')
            elif 'LNN' in class_name or 'Liquid' in class_name:
                features.append('Liquid Neural Networks')
            elif 'Agent' in class_name:
                features.append('Multi-Agent Systems')
            elif 'Memory' in class_name:
                features.append('Memory Management')
            elif 'Graph' in class_name:
                features.append('Graph Processing')
            elif 'Stream' in class_name:
                features.append('Stream Processing')
        
        # Check for GPU features
        if any(f.get('has_gpu') for f in component_info['files']):
            features.append('GPU Acceleration')
        
        # Check for async features
        if any(f.get('has_async') for f in component_info['files']):
            features.append('Async/Await')
        
        return list(set(features))
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_overview': {
                'total_components': len(self.components),
                'total_files': sum(len(c['files']) for c in self.components.values()),
                'real_implementations': len(self.real_implementations),
                'microservices': len(self.microservices),
                'components_with_tests': sum(1 for c in self.components.values() if c['has_tests']),
                'components_with_docs': sum(1 for c in self.components.values() if c['has_docs'])
            },
            'key_components': {},
            'real_implementations': self.real_implementations,
            'microservices': self.microservices,
            'integration_map': self.build_integration_map(),
            'technology_stack': self.analyze_tech_stack()
        }
        
        # Identify key components
        key_components = ['tda', 'lnn', 'agents', 'memory', 'orchestration', 
                         'streaming', 'consensus', 'neural', 'graph']
        
        for comp_name in key_components:
            if comp_name in self.components:
                report['key_components'][comp_name] = {
                    'classes': self.components[comp_name]['classes'][:10],  # Top 10
                    'real_implementations': self.components[comp_name]['real_implementations'],
                    'key_features': self.components[comp_name]['key_features']
                }
        
        # Save report
        with open('/workspace/comprehensive_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self.print_summary(report)
        
        return report
    
    def build_integration_map(self) -> Dict[str, List[str]]:
        """Build a map of component integrations"""
        
        integration_map = {}
        
        # Analyze imports to understand dependencies
        for comp_name, comp_info in self.components.items():
            dependencies = set()
            
            for file_info in comp_info['files']:
                for import_name in file_info.get('imports', []):
                    if 'aura_intelligence' in import_name:
                        # Extract component name from import
                        parts = import_name.split('.')
                        if len(parts) > 2:
                            dep_component = parts[2]
                            if dep_component != comp_name:
                                dependencies.add(dep_component)
            
            integration_map[comp_name] = list(dependencies)
        
        return integration_map
    
    def analyze_tech_stack(self) -> Dict[str, List[str]]:
        """Analyze technology stack used"""
        
        tech_stack = {
            'ml_frameworks': set(),
            'databases': set(),
            'messaging': set(),
            'web_frameworks': set(),
            'distributed': set(),
            'monitoring': set()
        }
        
        # Analyze all imports
        for comp_info in self.components.values():
            for file_info in comp_info['files']:
                for import_name in file_info.get('imports', []):
                    # ML Frameworks
                    if 'torch' in import_name:
                        tech_stack['ml_frameworks'].add('PyTorch')
                    elif 'tensorflow' in import_name:
                        tech_stack['ml_frameworks'].add('TensorFlow')
                    elif 'sklearn' in import_name:
                        tech_stack['ml_frameworks'].add('Scikit-learn')
                    
                    # Databases
                    elif 'neo4j' in import_name:
                        tech_stack['databases'].add('Neo4j')
                    elif 'redis' in import_name:
                        tech_stack['databases'].add('Redis')
                    elif 'postgres' in import_name:
                        tech_stack['databases'].add('PostgreSQL')
                    
                    # Messaging
                    elif 'kafka' in import_name:
                        tech_stack['messaging'].add('Kafka')
                    elif 'nats' in import_name:
                        tech_stack['messaging'].add('NATS')
                    
                    # Web Frameworks
                    elif 'fastapi' in import_name:
                        tech_stack['web_frameworks'].add('FastAPI')
                    elif 'flask' in import_name:
                        tech_stack['web_frameworks'].add('Flask')
                    
                    # Distributed
                    elif 'ray' in import_name:
                        tech_stack['distributed'].add('Ray')
                    elif 'dask' in import_name:
                        tech_stack['distributed'].add('Dask')
                    
                    # Monitoring
                    elif 'prometheus' in import_name:
                        tech_stack['monitoring'].add('Prometheus')
                    elif 'opentelemetry' in import_name:
                        tech_stack['monitoring'].add('OpenTelemetry')
        
        # Convert sets to lists
        return {k: list(v) for k, v in tech_stack.items()}
    
    def print_summary(self, report: Dict):
        """Print a summary of the analysis"""
        
        print("\n📊 PROJECT OVERVIEW")
        print("=" * 80)
        for key, value in report['project_overview'].items():
            print(f"  {key}: {value}")
        
        print("\n🔧 KEY COMPONENTS")
        print("=" * 80)
        for comp_name, comp_info in report['key_components'].items():
            print(f"\n{comp_name.upper()}:")
            print(f"  Classes: {len(comp_info['classes'])}")
            print(f"  Real Implementations: {len(comp_info['real_implementations'])}")
            print(f"  Features: {', '.join(comp_info['key_features'])}")
        
        print("\n🚀 MICROSERVICES")
        print("=" * 80)
        for service_name, service_info in report['microservices'].items():
            print(f"  {service_name}: Port {service_info['port']}")
        
        print("\n💻 TECHNOLOGY STACK")
        print("=" * 80)
        for category, techs in report['technology_stack'].items():
            if techs:
                print(f"  {category}: {', '.join(techs)}")
        
        print("\n✅ REAL IMPLEMENTATIONS FOUND")
        print("=" * 80)
        print(f"  Total: {len(report['real_implementations'])}")
        
        # Group by component
        by_component = {}
        for path, info in report['real_implementations'].items():
            comp = info['component']
            if comp not in by_component:
                by_component[comp] = 0
            by_component[comp] += 1
        
        for comp, count in sorted(by_component.items(), key=lambda x: x[1], reverse=True):
            print(f"  {comp}: {count} files")


if __name__ == "__main__":
    analyzer = ComprehensiveProjectAnalyzer()
    analyzer.analyze_project()