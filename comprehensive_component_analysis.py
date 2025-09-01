#!/usr/bin/env python3
"""
AURA Intelligence - Comprehensive Component Analysis
Deep analysis of all project components to identify next integration targets
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComponentAnalyzer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.analysis = {
            'timestamp': time.time(),
            'project_structure': {},
            'core_components': {},
            'working_systems': {},
            'integration_priorities': [],
            'next_targets': {}
        }
    
    def analyze_directory_structure(self):
        """Analyze the complete directory structure"""
        logger.info("Analyzing directory structure...")
        
        key_directories = {
            'core_intelligence': 'core/src/aura_intelligence',
            'main_source': 'src/aura', 
            'demos': 'demos',
            'benchmarks': 'benchmarks',
            'infrastructure': 'infrastructure',
            'monitoring': 'monitoring',
            'scripts': 'scripts',
            'tests': 'tests'
        }
        
        structure = {}
        for name, path in key_directories.items():
            full_path = self.project_root / path
            if full_path.exists():
                structure[name] = {
                    'path': str(path),
                    'exists': True,
                    'files': self._count_files(full_path),
                    'size_mb': self._get_directory_size(full_path)
                }
            else:
                structure[name] = {'path': str(path), 'exists': False}
        
        self.analysis['project_structure'] = structure
        return structure
    
    def analyze_core_components(self):
        """Analyze core AURA Intelligence components"""
        logger.info("Analyzing core components...")
        
        core_path = self.project_root / 'core/src/aura_intelligence'
        if not core_path.exists():
            self.analysis['core_components'] = {'error': 'Core path not found'}
            return {}
        
        components = {}
        
        # Key component directories
        component_dirs = [
            'components', 'tda', 'lnn', 'memory', 'agents', 
            'api', 'observability', 'orchestration', 'consensus'
        ]
        
        for comp_dir in component_dirs:
            comp_path = core_path / comp_dir
            if comp_path.exists():
                analysis = self._analyze_component_directory(comp_path)
                components[comp_dir] = analysis
        
        # Analyze real_components.py specifically
        real_components_file = core_path / 'components/real_components.py'
        if real_components_file.exists():
            components['real_components_analysis'] = self._analyze_real_components_file(real_components_file)
        
        self.analysis['core_components'] = components
        return components
    
    def analyze_working_systems(self):
        """Identify currently working systems"""
        logger.info("Identifying working systems...")
        
        working = {}
        
        # Test TDA system
        try:
            result = subprocess.run(['python3', 'test_real_tda_direct.py'], 
                                  capture_output=True, text=True, timeout=30, cwd=self.project_root)
            working['tda_system'] = {
                'status': 'working' if result.returncode == 0 else 'broken',
                'test_output': result.stdout[-500:] if result.stdout else '',
                'errors': result.stderr[-500:] if result.stderr else ''
            }
        except Exception as e:
            working['tda_system'] = {'status': 'error', 'error': str(e)}
        
        # Test API system
        try:
            import requests
            response = requests.get('http://localhost:8080/health', timeout=5)
            working['api_system'] = {
                'status': 'working' if response.status_code == 200 else 'broken',
                'response_time': response.elapsed.total_seconds() * 1000,
                'data': response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            working['api_system'] = {'status': 'not_running', 'error': str(e)}
        
        # Check for other working components
        working['dependencies'] = self._check_dependencies()
        
        self.analysis['working_systems'] = working
        return working
    
    def identify_integration_priorities(self):
        """Identify next best components to integrate"""
        logger.info("Identifying integration priorities...")
        
        priorities = []
        
        # Priority 1: LNN (Liquid Neural Networks)
        lnn_files = self._find_files('lnn', ['*.py'])
        if lnn_files:
            priorities.append({
                'name': 'Liquid Neural Networks (LNN)',
                'priority': 1,
                'rationale': 'Core differentiator, real MIT implementation available',
                'files': lnn_files[:5],  # Top 5 files
                'estimated_effort': 'Medium (2-3 days)',
                'dependencies': ['ncps', 'torch'],
                'integration_target': 'working_tda_api.py'
            })
        
        # Priority 2: Memory Systems
        memory_files = self._find_files('memory', ['*memory*.py'])
        if memory_files:
            priorities.append({
                'name': 'Shape-Aware Memory Systems',
                'priority': 2,
                'rationale': 'Supports TDA with topological feature caching',
                'files': memory_files[:5],
                'estimated_effort': 'Medium (2-3 days)',
                'dependencies': ['redis', 'faiss'],
                'integration_target': 'working_tda_api.py + Redis'
            })
        
        # Priority 3: Multi-Agent System
        agent_files = self._find_files('agents', ['*agent*.py'])
        if agent_files:
            priorities.append({
                'name': 'Multi-Agent System',
                'priority': 3,
                'rationale': 'Enables distributed failure prediction',
                'files': agent_files[:5],
                'estimated_effort': 'High (4-5 days)',
                'dependencies': ['asyncio', 'websockets'],
                'integration_target': 'New orchestration layer'
            })
        
        # Priority 4: Advanced Monitoring
        monitoring_files = self._find_files('observability', ['*.py'])
        if monitoring_files:
            priorities.append({
                'name': 'Advanced Monitoring & Observability',
                'priority': 4,
                'rationale': 'Production monitoring and metrics',
                'files': monitoring_files[:5],
                'estimated_effort': 'Medium (2-3 days)',
                'dependencies': ['prometheus', 'opentelemetry'],
                'integration_target': 'working_tda_api.py + Grafana'
            })
        
        # Priority 5: GPU Acceleration
        gpu_files = self._find_files('tda', ['*gpu*.py', '*cuda*.py'])
        if gpu_files:
            priorities.append({
                'name': 'GPU-Accelerated TDA',
                'priority': 5,
                'rationale': 'Performance optimization for large datasets',
                'files': gpu_files[:3],
                'estimated_effort': 'High (3-4 days)',
                'dependencies': ['cupy', 'torch', 'cuda'],
                'integration_target': 'TDA engine optimization'
            })
        
        self.analysis['integration_priorities'] = priorities
        return priorities
    
    def generate_next_steps(self):
        """Generate specific next steps"""
        logger.info("Generating next steps...")
        
        next_steps = {
            'immediate_actions': [],
            'this_week': [],
            'this_month': [],
            'specific_files_to_work_on': []
        }
        
        # Immediate actions (today)
        next_steps['immediate_actions'] = [
            {
                'action': 'Test LNN Integration',
                'command': 'python3 -c "from core.src.aura_intelligence.lnn.real_mit_lnn import LiquidNeuralNetwork; print(\'LNN working\')"',
                'expected_outcome': 'Verify LNN imports and basic functionality'
            },
            {
                'action': 'Test Redis Connection',
                'command': 'python3 -c "import redis; r=redis.Redis(); r.ping(); print(\'Redis working\')"',
                'expected_outcome': 'Confirm Redis is accessible'
            },
            {
                'action': 'Create LNN API Endpoint',
                'file': 'working_tda_api.py',
                'modification': 'Add /analyze/lnn endpoint with real neural processing'
            }
        ]
        
        # This week targets
        next_steps['this_week'] = [
            'Integrate working LNN with TDA API',
            'Add Redis-based memory system',
            'Create combined TDA+LNN analysis endpoint',
            'Build real-time monitoring dashboard',
            'Add load testing for performance validation'
        ]
        
        # This month targets
        next_steps['this_month'] = [
            'Multi-agent orchestration system',
            'GPU acceleration for TDA computations',
            'Advanced failure prediction models',
            'Production deployment pipeline',
            'Comprehensive test automation'
        ]
        
        # Specific files to work on next
        priorities = self.analysis.get('integration_priorities', [])
        if priorities:
            top_priority = priorities[0]
            next_steps['specific_files_to_work_on'] = [
                {
                    'file': f,
                    'priority': top_priority['name'],
                    'action': 'Integrate with working_tda_api.py'
                }
                for f in top_priority['files']
            ]
        
        self.analysis['next_targets'] = next_steps
        return next_steps
    
    def _count_files(self, directory: Path) -> Dict[str, int]:
        """Count files by extension"""
        counts = {}
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    ext = file_path.suffix or 'no_extension'
                    counts[ext] = counts.get(ext, 0) + 1
        except PermissionError:
            counts['error'] = 'Permission denied'
        return counts
    
    def _get_directory_size(self, directory: Path) -> float:
        """Get directory size in MB"""
        try:
            total_size = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
            return round(total_size / (1024 * 1024), 2)
        except:
            return 0.0
    
    def _analyze_component_directory(self, comp_path: Path) -> Dict[str, Any]:
        """Analyze a specific component directory"""
        analysis = {
            'files': self._count_files(comp_path),
            'key_files': [],
            'has_real_implementation': False,
            'has_tests': False
        }
        
        # Find key files
        for file_path in comp_path.rglob('*.py'):
            file_name = file_path.name.lower()
            if any(keyword in file_name for keyword in ['real_', 'production_', 'working_']):
                analysis['key_files'].append(str(file_path.relative_to(self.project_root)))
                analysis['has_real_implementation'] = True
            elif 'test' in file_name:
                analysis['has_tests'] = True
        
        return analysis
    
    def _analyze_real_components_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze the real_components.py file specifically"""
        analysis = {
            'file_size_kb': round(file_path.stat().st_size / 1024, 2),
            'classes_found': [],
            'imports': [],
            'has_gpu_support': False,
            'has_redis_support': False
        }
        
        try:
            content = file_path.read_text()
            
            # Find class definitions
            import re
            classes = re.findall(r'^class (\w+)', content, re.MULTILINE)
            analysis['classes_found'] = classes
            
            # Find imports
            imports = re.findall(r'^(?:from|import) ([\w.]+)', content, re.MULTILINE)
            analysis['imports'] = imports[:10]  # First 10 imports
            
            # Check for specific features
            analysis['has_gpu_support'] = 'torch' in content or 'cuda' in content
            analysis['has_redis_support'] = 'redis' in content
            analysis['has_async_support'] = 'async' in content
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _find_files(self, component: str, patterns: List[str]) -> List[str]:
        """Find files matching patterns in component directories"""
        files = []
        search_paths = [
            f'core/src/aura_intelligence/{component}',
            f'src/aura/{component}'
        ]
        
        for search_path in search_paths:
            full_path = self.project_root / search_path
            if full_path.exists():
                for pattern in patterns:
                    files.extend([str(f.relative_to(self.project_root)) 
                                for f in full_path.rglob(pattern) if f.is_file()])
        
        return sorted(list(set(files)))  # Remove duplicates
    
    def _check_dependencies(self) -> Dict[str, bool]:
        """Check if key dependencies are available"""
        dependencies = {}
        required_packages = [
            'torch', 'numpy', 'scipy', 'scikit-learn',
            'ripser', 'gudhi', 'faiss', 'ncps', 
            'redis', 'neo4j', 'fastapi', 'uvicorn'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                dependencies[package] = True
            except ImportError:
                dependencies[package] = False
        
        return dependencies
    
    def save_analysis(self, filename: str = "component_analysis_report.json"):
        """Save analysis to JSON file"""
        output_path = self.project_root / filename
        with open(output_path, 'w') as f:
            json.dump(self.analysis, f, indent=2, default=str)
        
        logger.info(f"Analysis saved to {output_path}")
        return output_path
    
    def print_summary(self):
        """Print analysis summary"""
        print("\n🧠 AURA Intelligence - Component Analysis Summary")
        print("=" * 60)
        
        # Project structure
        structure = self.analysis.get('project_structure', {})
        print(f"\n📁 Project Structure:")
        for name, info in structure.items():
            if info.get('exists'):
                files = info.get('files', {})
                total_files = sum(files.values()) if isinstance(files, dict) else 0
                print(f"  ✅ {name}: {total_files} files ({info.get('size_mb', 0)}MB)")
            else:
                print(f"  ❌ {name}: Missing")
        
        # Working systems
        working = self.analysis.get('working_systems', {})
        print(f"\n🚀 Working Systems:")
        for system, status in working.items():
            if isinstance(status, dict):
                state = status.get('status', 'unknown')
                if state == 'working':
                    print(f"  ✅ {system}: {state}")
                elif state == 'broken':
                    print(f"  ⚠️  {system}: {state}")
                else:
                    print(f"  ❌ {system}: {state}")
        
        # Integration priorities
        priorities = self.analysis.get('integration_priorities', [])
        print(f"\n🎯 Next Integration Targets:")
        for i, priority in enumerate(priorities[:3], 1):
            print(f"  {i}. {priority['name']}")
            print(f"     └─ {priority['rationale']}")
            print(f"     └─ Effort: {priority['estimated_effort']}")
        
        # Next immediate actions
        next_steps = self.analysis.get('next_targets', {})
        immediate = next_steps.get('immediate_actions', [])
        print(f"\n⚡ Immediate Next Actions:")
        for i, action in enumerate(immediate[:3], 1):
            print(f"  {i}. {action['action']}")
            if 'command' in action:
                print(f"     └─ Command: {action['command']}")

def main():
    """Run comprehensive component analysis"""
    print("🔍 Starting comprehensive project analysis...")
    
    analyzer = ComponentAnalyzer()
    
    # Run all analyses
    analyzer.analyze_directory_structure()
    analyzer.analyze_core_components()
    analyzer.analyze_working_systems()
    analyzer.identify_integration_priorities()
    analyzer.generate_next_steps()
    
    # Save and display results
    analyzer.save_analysis()
    analyzer.print_summary()
    
    print(f"\n💾 Full analysis saved to component_analysis_report.json")
    print(f"🎯 Ready to proceed with next integration phase!")

if __name__ == "__main__":
    main()