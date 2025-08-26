#!/usr/bin/env python3
"""
COMPREHENSIVE PROJECT EVALUATION 2025
Complete analysis of AURA Intelligence system architecture, working components, and issues
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import importlib.util

class ProjectEvaluator:
    def __init__(self):
        self.project_root = Path("/home/sina/projects/osiris-2")
        self.core_path = self.project_root / "core" / "src" / "aura_intelligence"
        self.evaluation_results = {
            "timestamp": time.time(),
            "project_structure": {},
            "component_analysis": {},
            "working_systems": {},
            "critical_issues": {},
            "recommendations": {}
        }
    
    def analyze_project_structure(self):
        """Analyze complete project structure"""
        print("📁 Analyzing project structure...")
        
        structure = {
            "total_python_files": 0,
            "directories": {},
            "key_components": {},
            "config_files": [],
            "api_files": [],
            "docker_files": []
        }
        
        # Count Python files by directory
        for root, dirs, files in os.walk(self.core_path):
            py_files = [f for f in files if f.endswith('.py')]
            if py_files:
                rel_path = os.path.relpath(root, self.core_path)
                structure["directories"][rel_path] = len(py_files)
                structure["total_python_files"] += len(py_files)
        
        # Find key configuration files
        for file_path in self.project_root.rglob("*.yaml"):
            structure["config_files"].append(str(file_path.relative_to(self.project_root)))
        for file_path in self.project_root.rglob("*.yml"):
            structure["config_files"].append(str(file_path.relative_to(self.project_root)))
        for file_path in self.project_root.rglob("Dockerfile*"):
            structure["docker_files"].append(str(file_path.relative_to(self.project_root)))
        for file_path in self.project_root.rglob("*api*.py"):
            if file_path.is_file():
                structure["api_files"].append(str(file_path.relative_to(self.project_root)))
        
        self.evaluation_results["project_structure"] = structure
        
        print(f"   📊 Total Python files: {structure['total_python_files']}")
        print(f"   📁 Directories with code: {len(structure['directories'])}")
        print(f"   ⚙️  Config files: {len(structure['config_files'])}")
        print(f"   🐳 Docker files: {len(structure['docker_files'])}")
        print(f"   🌐 API files: {len(structure['api_files'])}")
    
    def test_component_imports(self):
        """Test all major component imports"""
        print("\n🧪 Testing component imports...")
        
        components_to_test = {
            # Core Systems
            "config": [
                ("config/base.py", "Config Base"),
                ("config/api.py", "API Config"), 
                ("config/memory.py", "Memory Config"),
                ("config/aura.py", "AURA Config")
            ],
            
            # Intelligence Core
            "intelligence": [
                ("core/system.py", "Core System"),
                ("core/interfaces.py", "Core Interfaces"),
                ("core/unified_system.py", "Unified System"),
                ("core/types.py", "Core Types")
            ],
            
            # TDA (Topological Data Analysis)
            "tda": [
                ("tda/algorithms.py", "TDA Algorithms"),
                ("tda/core.py", "TDA Core"),
                ("tda/real_tda.py", "Real TDA"),
                ("tda/models.py", "TDA Models"),
                ("tda/service.py", "TDA Service")
            ],
            
            # LNN (Liquid Neural Networks)
            "lnn": [
                ("lnn/core.py", "LNN Core"),
                ("lnn/dynamics.py", "LNN Dynamics"),
                ("lnn/real_mit_lnn.py", "Real MIT LNN"),
                ("neural/lnn.py", "Neural LNN"),
                ("neural/liquid_real.py", "Liquid Real")
            ],
            
            # Memory Systems
            "memory": [
                ("memory/shape_memory_v2.py", "Shape Memory V2"),
                ("memory/redis_store.py", "Redis Store"),
                ("memory/storage_interface.py", "Storage Interface"),
                ("memory/hybrid_manager.py", "Hybrid Manager")
            ],
            
            # Agents
            "agents": [
                ("agents/base.py", "Agent Base"),
                ("agents/supervisor.py", "Agent Supervisor"),
                ("agents/council/agent_council.py", "Agent Council"),
                ("agents/schemas/base.py", "Agent Schema Base")
            ],
            
            # Observability & Monitoring
            "observability": [
                ("observability/core.py", "Observability Core"),
                ("observability/prometheus_metrics.py", "Prometheus Metrics"),
                ("monitoring/production_monitor.py", "Production Monitor")
            ],
            
            # Orchestration
            "orchestration": [
                ("orchestration/workflows.py", "Workflows"),
                ("orchestration/langgraph_workflows.py", "LangGraph Workflows"),
                ("orchestration/temporal_signalfirst.py", "Temporal")
            ]
        }
        
        component_results = {}
        
        for category, components in components_to_test.items():
            print(f"\n🔍 Testing {category} components:")
            category_results = {"working": [], "failed": [], "not_found": []}
            
            for file_path, name in components:
                full_path = self.core_path / file_path
                
                if not full_path.exists():
                    category_results["not_found"].append(name)
                    print(f"   ❓ {name}: Not found")
                    continue
                
                try:
                    # Try to compile first
                    with open(full_path, 'r') as f:
                        code = f.read()
                    compile(code, str(full_path), 'exec')
                    
                    # Try to load module
                    spec = importlib.util.spec_from_file_location(name.replace(' ', '_'), str(full_path))
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    classes = [item for item in dir(module) if not item.startswith('_') and item[0].isupper()]
                    category_results["working"].append({
                        "name": name,
                        "file": file_path,
                        "classes": len(classes),
                        "classes_list": classes[:5]  # First 5 classes
                    })
                    print(f"   ✅ {name}: SUCCESS ({len(classes)} classes)")
                    
                except Exception as e:
                    category_results["failed"].append({
                        "name": name,
                        "file": file_path,
                        "error": str(e)[:100]
                    })
                    print(f"   ❌ {name}: {str(e)[:80]}")
            
            component_results[category] = category_results
        
        self.evaluation_results["component_analysis"] = component_results
        
        # Summary
        total_working = sum(len(cat["working"]) for cat in component_results.values())
        total_tested = sum(len(cat["working"]) + len(cat["failed"]) + len(cat["not_found"]) 
                          for cat in component_results.values())
        success_rate = total_working / total_tested if total_tested > 0 else 0
        
        print(f"\n📊 Component Import Summary:")
        print(f"   Working: {total_working}")
        print(f"   Failed: {total_tested - total_working}")
        print(f"   Success Rate: {success_rate*100:.1f}%")
    
    def check_running_systems(self):
        """Check what systems are actually running"""
        print("\n🏃 Checking running systems...")
        
        running_systems = {
            "apis": [],
            "processes": [],
            "ports": []
        }
        
        # Check for running Python processes
        try:
            result = subprocess.run(['pgrep', '-f', 'python.*api'], 
                                  capture_output=True, text=True)
            if result.stdout:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        try:
                            cmd_result = subprocess.run(['ps', '-p', pid, '-o', 'cmd='], 
                                                      capture_output=True, text=True)
                            if cmd_result.stdout:
                                running_systems["processes"].append({
                                    "pid": pid,
                                    "command": cmd_result.stdout.strip()
                                })
                        except:
                            pass
        except:
            pass
        
        # Check common ports
        ports_to_check = [8000, 8001, 8080, 3000, 9090, 9091]
        for port in ports_to_check:
            try:
                result = subprocess.run(['lsof', '-i', f':{port}'], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout:
                    running_systems["ports"].append({
                        "port": port,
                        "details": result.stdout.strip().split('\n')[1] if len(result.stdout.strip().split('\n')) > 1 else ""
                    })
                    print(f"   🌐 Port {port}: ACTIVE")
            except:
                pass
        
        # Check for APIs by trying to connect
        api_endpoints = [
            ("http://localhost:8000", "Main API"),
            ("http://localhost:8001", "Working AURA API"),
            ("http://localhost:8080", "Alternative API"),
            ("http://localhost:3000", "Frontend"),
            ("http://localhost:9090", "Prometheus"),
            ("http://localhost:9091", "Grafana")
        ]
        
        for url, name in api_endpoints:
            try:
                import urllib.request
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        running_systems["apis"].append({
                            "url": url,
                            "name": name,
                            "status": "ACTIVE"
                        })
                        print(f"   ✅ {name}: ACTIVE at {url}")
            except:
                pass
        
        self.evaluation_results["working_systems"] = running_systems
    
    def identify_critical_issues(self):
        """Identify critical blocking issues"""
        print("\n🚨 Identifying critical issues...")
        
        critical_files_to_check = [
            "infrastructure/kafka_event_mesh.py",
            "orchestration/workflows/nodes/supervisor.py", 
            "tda/algorithms.py",
            "lnn/real_mit_lnn.py",
            "observability/prometheus_integration.py"
        ]
        
        issues = {
            "syntax_errors": [],
            "import_blockers": [],
            "missing_dependencies": [],
            "configuration_issues": []
        }
        
        for file_path in critical_files_to_check:
            full_path = self.core_path / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        code = f.read()
                    compile(code, str(full_path), 'exec')
                    print(f"   ✅ {file_path}: Syntax OK")
                except SyntaxError as e:
                    issues["syntax_errors"].append({
                        "file": file_path,
                        "error": str(e),
                        "line": getattr(e, 'lineno', 'unknown')
                    })
                    print(f"   ❌ {file_path}: Syntax Error - {e}")
                except Exception as e:
                    issues["import_blockers"].append({
                        "file": file_path,
                        "error": str(e)[:150]
                    })
                    print(f"   ⚠️  {file_path}: Other Issue - {str(e)[:80]}")
        
        # Check for missing key dependencies
        dependencies_to_check = [
            "fastapi", "uvicorn", "redis", "prometheus_client", 
            "numpy", "torch", "transformers", "langchain"
        ]
        
        for dep in dependencies_to_check:
            try:
                __import__(dep)
                print(f"   ✅ {dep}: Available")
            except ImportError:
                issues["missing_dependencies"].append(dep)
                print(f"   ❌ {dep}: Missing")
        
        self.evaluation_results["critical_issues"] = issues
    
    def generate_recommendations(self):
        """Generate actionable recommendations"""
        print("\n💡 Generating recommendations...")
        
        recommendations = {
            "immediate_fixes": [],
            "system_improvements": [],
            "architecture_suggestions": [],
            "next_steps": []
        }
        
        # Analyze results and generate recommendations
        component_analysis = self.evaluation_results.get("component_analysis", {})
        critical_issues = self.evaluation_results.get("critical_issues", {})
        
        # Immediate fixes
        if critical_issues.get("syntax_errors"):
            recommendations["immediate_fixes"].append({
                "priority": "HIGH",
                "action": "Fix syntax errors in critical files",
                "files": [issue["file"] for issue in critical_issues["syntax_errors"]],
                "impact": "Enables basic system imports and functionality"
            })
        
        # Count working components per category
        working_by_category = {}
        total_by_category = {}
        for category, results in component_analysis.items():
            working_by_category[category] = len(results.get("working", []))
            total_by_category[category] = len(results.get("working", [])) + len(results.get("failed", []))
        
        # System improvements
        best_categories = sorted(working_by_category.items(), key=lambda x: x[1], reverse=True)
        if best_categories:
            recommendations["system_improvements"].append({
                "priority": "MEDIUM",
                "action": f"Build production system using working {best_categories[0][0]} components",
                "details": f"{best_categories[0][1]} components working",
                "impact": "Immediate functional system deployment"
            })
        
        # Architecture suggestions
        recommendations["architecture_suggestions"].append({
            "priority": "MEDIUM", 
            "action": "Implement microservices architecture",
            "rationale": "Isolate working components from broken ones",
            "impact": "Reduce import cascade failures"
        })
        
        # Next steps
        working_systems = self.evaluation_results.get("working_systems", {})
        if working_systems.get("apis"):
            recommendations["next_steps"].append({
                "priority": "LOW",
                "action": "Scale working API with load balancer",
                "details": f"{len(working_systems['apis'])} APIs currently running"
            })
        
        recommendations["next_steps"].extend([
            {
                "priority": "HIGH",
                "action": "Fix import blocker files (kafka_event_mesh.py, supervisor.py)",
                "impact": "Unlocks access to remaining 40+ components"
            },
            {
                "priority": "MEDIUM", 
                "action": "Implement comprehensive testing pipeline",
                "impact": "Prevent regression of syntax fixes"
            },
            {
                "priority": "LOW",
                "action": "Set up monitoring with working Prometheus components", 
                "impact": "Production observability"
            }
        ])
        
        self.evaluation_results["recommendations"] = recommendations
        
        for category, recs in recommendations.items():
            print(f"\n   {category.upper()}:")
            for rec in recs:
                print(f"     [{rec['priority']}] {rec['action']}")
    
    def save_evaluation(self):
        """Save complete evaluation to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_evaluation_{timestamp}.json"
        filepath = self.project_root / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        print(f"\n💾 Complete evaluation saved to: {filename}")
        return filepath
    
    def run_complete_evaluation(self):
        """Run complete project evaluation"""
        print("🔍 COMPREHENSIVE PROJECT EVALUATION - AURA INTELLIGENCE 2025")
        print("=" * 70)
        
        self.analyze_project_structure()
        self.test_component_imports()
        self.check_running_systems()
        self.identify_critical_issues()
        self.generate_recommendations()
        
        filepath = self.save_evaluation()
        
        print("\n" + "=" * 70)
        print("📋 EVALUATION SUMMARY")
        print("=" * 70)
        
        # Key metrics
        structure = self.evaluation_results["project_structure"]
        print(f"📁 Project Structure: {structure['total_python_files']} Python files")
        
        component_analysis = self.evaluation_results["component_analysis"]
        total_working = sum(len(cat["working"]) for cat in component_analysis.values())
        total_tested = sum(len(cat["working"]) + len(cat["failed"]) + len(cat["not_found"]) 
                          for cat in component_analysis.values())
        success_rate = total_working / total_tested if total_tested > 0 else 0
        
        print(f"🧪 Components: {total_working}/{total_tested} working ({success_rate*100:.1f}%)")
        
        working_systems = self.evaluation_results["working_systems"]
        print(f"🏃 Running Systems: {len(working_systems['apis'])} APIs, {len(working_systems['ports'])} active ports")
        
        critical_issues = self.evaluation_results["critical_issues"]
        print(f"🚨 Critical Issues: {len(critical_issues['syntax_errors'])} syntax errors, {len(critical_issues['missing_dependencies'])} missing deps")
        
        print(f"\n📄 Full report: {filepath}")
        
        return self.evaluation_results

if __name__ == "__main__":
    evaluator = ProjectEvaluator()
    results = evaluator.run_complete_evaluation()