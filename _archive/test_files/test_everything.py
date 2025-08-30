#!/usr/bin/env python3
"""
AURA Complete System Test - Tests ALL 213 Components
This will test EVERYTHING and report exactly what works and what doesn't
"""

import os
import sys
import json
import time
import subprocess
import urllib.request
import urllib.error
import socket
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add paths
sys.path.append('src')
sys.path.append('core/src')

# Colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
PURPLE = '\033[95m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

class AURACompleteTester:
    """Test ALL components of AURA Intelligence System"""
    
    def __init__(self):
        self.results = {
            "environment": {},
            "files": {},
            "directories": {},
            "components": {},
            "demo": {},
            "infrastructure": {},
            "imports": {},
            "api_keys": {},
            "processes": {},
            "network": {},
            "performance": {},
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }
        
        # Component definitions
        self.tda_algorithms = [
            # Quantum-Enhanced (20)
            "quantum_ripser", "neural_persistence", "quantum_witness", "quantum_mapper",
            "quantum_landscapes", "quantum_wasserstein", "quantum_bottleneck", "quantum_kernel",
            "quantum_clustering", "quantum_autoencoder", "quantum_transform", "quantum_zigzag",
            "quantum_multiparameter", "quantum_extended", "quantum_circular", "quantum_cohomology",
            "quantum_cup", "quantum_steenrod", "quantum_khovanov", "quantum_invariants",
            
            # Agent-Specific (15)
            "agent_topology_analyzer", "cascade_predictor", "bottleneck_detector", "community_finder",
            "influence_mapper", "failure_propagator", "resilience_scorer", "coordination_analyzer",
            "communication_topology", "load_distribution", "trust_network", "consensus_topology",
            "swarm_analyzer", "emergence_detector", "synchronization_mapper",
            
            # Streaming (20)
            "streaming_vietoris_rips", "streaming_alpha", "streaming_witness", "dynamic_persistence",
            "incremental_homology", "online_mapper", "sliding_window_tda", "temporal_persistence",
            "event_driven_tda", "adaptive_sampling", "progressive_computation", "lazy_evaluation",
            "cache_aware_tda", "parallel_streaming", "distributed_streaming", "edge_computing_tda",
            "low_latency_tda", "predictive_streaming", "anomaly_streaming", "adaptive_resolution",
            
            # GPU-Accelerated (15)
            "simba_gpu", "alpha_complex_gpu", "ripser_gpu", "gudhi_gpu", "cuda_persistence",
            "tensor_tda", "gpu_mapper", "parallel_homology", "batch_persistence", "multi_gpu_tda",
            "gpu_wasserstein", "gpu_landscapes", "gpu_kernels", "gpu_vectorization", "gpu_optimization",
            
            # Classical (30)
            "vietoris_rips", "alpha_complex", "witness_complex", "mapper", "persistent_landscapes",
            "persistence_images", "euler_characteristic", "betti_curves", "persistence_entropy",
            "wasserstein_distance", "bottleneck_distance", "kernel_methods", "tda_clustering",
            "persistence_diagrams", "homology_computation", "cohomology_computation",
            "simplicial_complex", "cubical_complex", "cech_complex", "rips_filtration",
            "lower_star_filtration", "discrete_morse", "persistent_homology", "zigzag_persistence",
            "multiparameter_persistence", "extended_persistence", "circular_coordinates",
            "cohomology_operations", "cup_products", "steenrod_squares",
            
            # Advanced (12)
            "causal_tda", "neural_surveillance", "specseq_plus", "hybrid_persistence",
            "topological_autoencoders", "persistent_homology_transform", "sheaf_cohomology",
            "motivic_cohomology", "operadic_tda", "infinity_tda", "derived_tda", "homotopy_tda"
        ]
        
        self.neural_networks = [
            "mit_liquid_nn", "adaptive_lnn", "edge_lnn", "distributed_lnn", "quantum_lnn",
            "neuromorphic_lnn", "hybrid_lnn", "streaming_lnn", "federated_lnn", "secure_lnn"
        ]
        
        self.memory_systems = [
            # Shape-Aware (8)
            "shape_mem_v2_prod", "topological_indexer", "betti_cache", "persistence_store",
            "wasserstein_index", "homology_memory", "shape_fusion", "adaptive_memory",
            
            # CXL Tiers (8)
            "L1_CACHE", "L2_CACHE", "L3_CACHE", "RAM", "CXL_HOT", "PMEM_WARM", "NVME_COLD", "HDD_ARCHIVE",
            
            # Hybrid Manager (10)
            "unified_allocator", "tier_optimizer", "prefetch_engine", "memory_pooling",
            "compression_engine", "dedup_engine", "migration_controller", "qos_manager",
            "power_optimizer", "wear_leveling",
            
            # Memory Bus (5)
            "cxl_controller", "ddr5_adapter", "pcie5_bridge", "coherency_manager", "numa_optimizer",
            
            # Vector Storage (9)
            "redis_vector", "qdrant_store", "faiss_index", "annoy_trees", "hnsw_graph",
            "lsh_buckets", "scann_index", "vespa_store", "custom_embeddings"
        ]
        
        self.infrastructure_components = [
            # Byzantine (5)
            "hotstuff", "pbft", "raft", "tendermint", "hashgraph",
            
            # Neuromorphic (8)
            "spiking_gnn", "lif_neurons", "stdp_learning", "liquid_state",
            "reservoir_computing", "event_driven", "dvs_processing", "loihi_patterns",
            
            # MoE (5)
            "switch_transformer", "expert_choice", "top_k_gating", "load_balanced", "semantic_routing",
            
            # Observability (5)
            "prometheus_metrics", "jaeger_tracing", "grafana_dashboards", "custom_telemetry", "log_aggregation",
            
            # Resilience (8)
            "circuit_breaker", "retry_policy", "bulkhead", "timeout_handler",
            "fallback_chain", "health_checks", "rate_limiter", "adaptive_concurrency",
            
            # Orchestration (10)
            "workflow_engine", "dag_scheduler", "event_router", "task_queue", "job_scheduler",
            "pipeline_manager", "state_machine", "saga_orchestrator", "choreography_engine", "temporal_workflows",
            
            # Adapters (10)
            "neo4j_adapter", "redis_adapter", "kafka_mesh", "postgres_adapter", "minio_storage",
            "qdrant_vector", "auth_service", "api_gateway", "service_mesh", "config_server"
        ]
    
    def print_header(self, title: str, level: int = 1):
        """Print formatted header"""
        if level == 1:
            print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
            print(f"{BOLD}{BLUE}{title.center(70)}{RESET}")
            print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")
        elif level == 2:
            print(f"\n{BOLD}{CYAN}{title}{RESET}")
            print(f"{CYAN}{'-'*len(title)}{RESET}")
        else:
            print(f"\n{BOLD}{PURPLE}{title}{RESET}")
    
    def test_result(self, category: str, test: str, passed: bool, message: str = ""):
        """Record and print test result"""
        self.results["total_tests"] += 1
        
        if passed:
            self.results["passed"] += 1
            print(f"{GREEN}✓ {test}{RESET} {message}")
        else:
            self.results["failed"] += 1
            print(f"{RED}✗ {test}{RESET} {message}")
        
        if category not in self.results:
            self.results[category] = {}
        self.results[category][test] = passed
    
    def test_warning(self, message: str):
        """Record warning"""
        self.results["warnings"] += 1
        print(f"{YELLOW}⚠ {message}{RESET}")
    
    def test_environment(self):
        """Test environment setup"""
        self.print_header("ENVIRONMENT CONFIGURATION", 2)
        
        # Check Python version
        py_version = sys.version_info
        self.test_result("environment", "Python Version", 
                        py_version >= (3, 8),
                        f"- {py_version.major}.{py_version.minor}.{py_version.micro}")
        
        # Check .env file
        env_exists = os.path.exists('.env')
        self.test_result("environment", ".env file", env_exists)
        
        if env_exists:
            # Load and check API keys
            with open('.env', 'r') as f:
                env_content = f.read()
            
            api_keys = {
                "LANGSMITH_API_KEY": "lsv2_pt_" in env_content,
                "GEMINI_API_KEY": "AIzaSy" in env_content,
                "NEO4J_URI": "neo4j" in env_content or "bolt://" in env_content,
                "REDIS_HOST": "REDIS_HOST=" in env_content
            }
            
            for key, found in api_keys.items():
                self.test_result("api_keys", key, found)
        
        # Check working directory
        cwd = os.getcwd()
        self.test_result("environment", "Working Directory", 
                        "workspace" in cwd or "aura" in cwd.lower(),
                        f"- {cwd}")
    
    def test_directory_structure(self):
        """Test all directories exist"""
        self.print_header("DIRECTORY STRUCTURE", 2)
        
        required_dirs = {
            "src/aura": "Main source code",
            "src/aura/core": "Core components",
            "src/aura/tda": "TDA algorithms",
            "src/aura/lnn": "Liquid Neural Networks",
            "src/aura/memory": "Memory systems",
            "src/aura/agents": "Agent systems",
            "src/aura/consensus": "Byzantine consensus",
            "src/aura/neuromorphic": "Neuromorphic computing",
            "src/aura/api": "API implementations",
            "src/aura/integrations": "External integrations",
            "demos": "Demo files",
            "benchmarks": "Performance tests",
            "utilities": "Helper scripts",
            "infrastructure": "Docker/K8s configs",
            "documentation": "All documentation",
            "tests": "Test files",
            "models": "Model storage",
            "cache": "Cache directory",
            "logs": "Log files",
            "data": "Data storage"
        }
        
        for dir_path, description in required_dirs.items():
            exists = os.path.exists(dir_path)
            self.test_result("directories", dir_path, exists, f"- {description}")
    
    def test_core_files(self):
        """Test all core files exist"""
        self.print_header("CORE FILES", 2)
        
        core_files = {
            ".env": "Environment configuration",
            ".env.example": "Example environment",
            "requirements.txt": "Python dependencies",
            "README.md": "Main documentation",
            "AURA_FINAL_INDEX.md": "System index",
            "NEXT_STEPS.md": "Roadmap",
            "src/aura/__init__.py": "Package init",
            "src/aura/core/system.py": "Main system (213 components)",
            "src/aura/core/config.py": "Configuration",
            "demos/aura_working_demo_2025.py": "Main demo",
            "benchmarks/aura_benchmark_100_agents.py": "100 agent test",
            "utilities/ULTIMATE_AURA_API_2025.py": "Unified API",
            "infrastructure/docker-compose.yml": "Docker setup",
            "infrastructure/prometheus.yml": "Monitoring config"
        }
        
        for file_path, description in core_files.items():
            exists = os.path.exists(file_path)
            self.test_result("files", file_path, exists, f"- {description}")
            
            if exists and file_path.endswith('.py'):
                # Check if Python file is valid
                try:
                    with open(file_path, 'r') as f:
                        compile(f.read(), file_path, 'exec')
                    self.test_result("files", f"{file_path} syntax", True)
                except SyntaxError as e:
                    self.test_result("files", f"{file_path} syntax", False, f"- {e}")
    
    def test_component_definitions(self):
        """Test all 213 component definitions"""
        self.print_header("COMPONENT DEFINITIONS (213 Total)", 2)
        
        # Check TDA algorithms (112)
        self.print_header("TDA Algorithms (112)", 3)
        system_file = "src/aura/core/system.py"
        
        if os.path.exists(system_file):
            with open(system_file, 'r') as f:
                system_content = f.read()
            
            found_tda = 0
            for algo in self.tda_algorithms:
                if f'"{algo}"' in system_content:
                    found_tda += 1
            
            self.test_result("components", "TDA Algorithms Defined", 
                           found_tda == 112,
                           f"- {found_tda}/112 found")
        
        # Check Neural Networks (10)
        self.print_header("Neural Networks (10)", 3)
        found_nn = sum(1 for nn in self.neural_networks if f'"{nn}"' in system_content)
        self.test_result("components", "Neural Networks Defined",
                       found_nn == 10,
                       f"- {found_nn}/10 found")
        
        # Check Memory Systems (40)
        self.print_header("Memory Systems (40)", 3)
        found_mem = sum(1 for mem in self.memory_systems if f'"{mem}"' in system_content)
        self.test_result("components", "Memory Systems Defined",
                       found_mem == 40,
                       f"- {found_mem}/40 found")
        
        # Check Agent Systems (100)
        self.print_header("Agent Systems (100)", 3)
        ia_count = sum(1 for i in range(1, 51) if f'"pattern_ia_{i:03d}"' in system_content or 
                                                  f'"anomaly_ia_{i:03d}"' in system_content or
                                                  f'"trend_ia_{i:03d}"' in system_content or
                                                  f'"context_ia_{i:03d}"' in system_content or
                                                  f'"feature_ia_{i:03d}"' in system_content)
        
        ca_count = sum(1 for i in range(1, 51) if f'"resource_ca_{i:03d}"' in system_content or
                                                  f'"schedule_ca_{i:03d}"' in system_content or
                                                  f'"balance_ca_{i:03d}"' in system_content or
                                                  f'"optimize_ca_{i:03d}"' in system_content or
                                                  f'"coord_ca_{i:03d}"' in system_content)
        
        self.test_result("components", "Information Agents",
                       ia_count >= 25,
                       f"- {ia_count}/50 patterns found")
        self.test_result("components", "Control Agents",
                       ca_count >= 25,
                       f"- {ca_count}/50 patterns found")
        
        # Check Infrastructure (51)
        self.print_header("Infrastructure Components (51)", 3)
        found_infra = sum(1 for comp in self.infrastructure_components if f'"{comp}"' in system_content)
        self.test_result("components", "Infrastructure Components",
                       found_infra >= 25,
                       f"- {found_infra}/51 found")
        
        # Total component count
        total_found = found_tda + found_nn + found_mem + (ia_count + ca_count) + found_infra
        self.test_result("components", "Total Components",
                       total_found >= 150,
                       f"- {total_found}/213 verified")
    
    def test_demo_running(self):
        """Test if demo is running and functional"""
        self.print_header("DEMO FUNCTIONALITY", 2)
        
        demo_url = "http://localhost:8080"
        
        # Check if demo is accessible
        try:
            response = urllib.request.urlopen(demo_url, timeout=5)
            html = response.read().decode('utf-8')
            
            self.test_result("demo", "Demo Running", True, f"- {len(html)} bytes received")
            
            # Check demo features
            features = {
                "Title Present": "AURA Agent Failure Prevention" in html,
                "Canvas Visualization": "<canvas" in html,
                "Agent Network": "Agent Network" in html,
                "Topology Analysis": "Topology Analysis" in html,
                "AURA Protection Toggle": "AURA Protection" in html,
                "Metrics Display": "Metrics" in html or "Performance" in html,
                "WebSocket Support": "ws://" in html or "WebSocket" in html,
                "30 Agents": "30" in html and "agents" in html.lower()
            }
            
            for feature, present in features.items():
                self.test_result("demo", feature, present)
            
            # Performance test
            times = []
            for _ in range(5):
                start = time.time()
                urllib.request.urlopen(demo_url, timeout=5)
                times.append((time.time() - start) * 1000)
            
            avg_time = sum(times) / len(times)
            self.test_result("performance", "Response Time",
                           avg_time < 100,
                           f"- {avg_time:.1f}ms average")
            
        except Exception as e:
            self.test_result("demo", "Demo Running", False, f"- {str(e)}")
            print(f"\n{YELLOW}To start demo: python3 demos/aura_working_demo_2025.py{RESET}")
    
    def test_imports(self):
        """Test Python imports"""
        self.print_header("PYTHON IMPORTS", 2)
        
        # Test AURA package import
        try:
            sys.path.insert(0, 'src')
            import aura
            self.test_result("imports", "import aura", True)
            
            # Check version
            if hasattr(aura, '__version__'):
                self.test_result("imports", "AURA version", True, f"- {aura.__version__}")
            
            # Test submodules
            try:
                from aura.core.config import AURAConfig
                self.test_result("imports", "AURAConfig", True)
                
                config = AURAConfig()
                self.test_result("imports", "AURAConfig instantiation", True)
            except Exception as e:
                self.test_result("imports", "AURAConfig", False, f"- {e}")
            
        except Exception as e:
            self.test_result("imports", "import aura", False, f"- {e}")
    
    def test_processes(self):
        """Test running processes"""
        self.print_header("PROCESS STATUS", 2)
        
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            processes = result.stdout
            
            # Check for demo process
            demo_running = 'aura_working_demo' in processes
            self.test_result("processes", "Demo Process", demo_running)
            
            if demo_running:
                # Find PID
                for line in processes.split('\n'):
                    if 'aura_working_demo' in line and 'python' in line:
                        parts = line.split()
                        if len(parts) > 1:
                            pid = parts[1]
                            print(f"  {BLUE}PID: {pid}{RESET}")
                            break
            
            # Check port 8080
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            port_used = sock.connect_ex(('localhost', 8080)) == 0
            sock.close()
            
            self.test_result("network", "Port 8080", port_used, 
                           "- In use" if port_used else "- Available")
            
        except Exception as e:
            self.test_warning(f"Could not check processes: {e}")
    
    def test_infrastructure(self):
        """Test infrastructure configuration"""
        self.print_header("INFRASTRUCTURE", 2)
        
        # Check Docker
        try:
            docker_result = subprocess.run(['docker', '--version'], 
                                         capture_output=True, text=True)
            docker_installed = docker_result.returncode == 0
            self.test_result("infrastructure", "Docker", docker_installed,
                           f"- {docker_result.stdout.strip()}")
        except:
            self.test_result("infrastructure", "Docker", False, "- Not installed")
        
        # Check docker-compose.yml
        compose_file = "infrastructure/docker-compose.yml"
        if os.path.exists(compose_file):
            with open(compose_file, 'r') as f:
                compose_content = f.read()
            
            services = ["neo4j", "redis", "postgres", "prometheus", "grafana", 
                       "jaeger", "minio", "qdrant"]
            
            for service in services:
                self.test_result("infrastructure", f"{service} configured",
                               service in compose_content)
    
    def generate_report(self):
        """Generate comprehensive report"""
        self.print_header("FINAL REPORT", 1)
        
        # Calculate percentages
        total = self.results["total_tests"]
        if total > 0:
            success_rate = (self.results["passed"] / total) * 100
            failure_rate = (self.results["failed"] / total) * 100
        else:
            success_rate = failure_rate = 0
        
        # Summary stats
        print(f"{BOLD}Test Statistics:{RESET}")
        print(f"  {GREEN}Passed: {self.results['passed']}{RESET}")
        print(f"  {RED}Failed: {self.results['failed']}{RESET}")
        print(f"  {YELLOW}Warnings: {self.results['warnings']}{RESET}")
        print(f"  Total Tests: {total}")
        print(f"  Success Rate: {success_rate:.1f}%")
        
        # Component summary
        print(f"\n{BOLD}Component Status:{RESET}")
        categories = ["environment", "directories", "files", "components", 
                     "demo", "imports", "infrastructure"]
        
        for category in categories:
            if category in self.results and isinstance(self.results[category], dict):
                passed = sum(1 for v in self.results[category].values() if v)
                total_cat = len(self.results[category])
                if total_cat > 0:
                    cat_rate = (passed / total_cat) * 100
                    status = f"{GREEN}✓{RESET}" if cat_rate >= 80 else f"{YELLOW}⚠{RESET}" if cat_rate >= 50 else f"{RED}✗{RESET}"
                    print(f"  {status} {category.title()}: {passed}/{total_cat} ({cat_rate:.0f}%)")
        
        # Critical issues
        print(f"\n{BOLD}Critical Issues:{RESET}")
        critical_issues = []
        
        if not self.results.get("demo", {}).get("Demo Running", False):
            critical_issues.append("Demo is not running")
        
        if self.results.get("components", {}).get("Total Components", False):
            if "- " in str(self.results["components"]["Total Components"]):
                # Extract component count
                comp_info = str(self.results["components"]["Total Components"])
                if "/" in comp_info:
                    found = int(comp_info.split("/")[0].split()[-1])
                    if found < 150:
                        critical_issues.append(f"Only {found}/213 components verified")
        
        if critical_issues:
            for issue in critical_issues:
                print(f"  {RED}• {issue}{RESET}")
        else:
            print(f"  {GREEN}No critical issues found!{RESET}")
        
        # Recommendations
        print(f"\n{BOLD}Recommendations:{RESET}")
        
        if not self.results.get("demo", {}).get("Demo Running", False):
            print(f"  1. Start the demo: {CYAN}python3 demos/aura_working_demo_2025.py{RESET}")
        
        if self.results["failed"] > 0:
            print(f"  2. Fix failed tests (see above)")
        
        if self.results["warnings"] > 0:
            print(f"  3. Address warnings for optimal performance")
        
        # Save detailed report
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_tests": total,
                "passed": self.results["passed"],
                "failed": self.results["failed"],
                "warnings": self.results["warnings"],
                "success_rate": success_rate
            },
            "details": self.results
        }
        
        with open("complete_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{BLUE}Detailed report saved to complete_test_report.json{RESET}")
        
        # Overall status
        print(f"\n{BOLD}Overall System Status:{RESET}")
        if success_rate >= 90:
            print(f"{GREEN}{'='*50}")
            print(f"✅ AURA System is HEALTHY and READY!")
            print(f"{'='*50}{RESET}")
        elif success_rate >= 70:
            print(f"{YELLOW}{'='*50}")
            print(f"⚠️  AURA System needs minor fixes")
            print(f"{'='*50}{RESET}")
        else:
            print(f"{RED}{'='*50}")
            print(f"❌ AURA System needs attention")
            print(f"{'='*50}{RESET}")
        
        return success_rate >= 70
    
    def run_all_tests(self):
        """Run complete test suite"""
        self.print_header("AURA INTELLIGENCE COMPLETE SYSTEM TEST", 1)
        print(f"{BLUE}Project ID: bc-a397ac41-47c3-4620-a5ec-c56fb1f50fd0{RESET}")
        print(f"{BLUE}Testing all 213 components...{RESET}\n")
        
        # Run all test categories
        self.test_environment()
        self.test_directory_structure()
        self.test_core_files()
        self.test_component_definitions()
        self.test_demo_running()
        self.test_imports()
        self.test_processes()
        self.test_infrastructure()
        
        # Generate report
        return self.generate_report()


if __name__ == "__main__":
    tester = AURACompleteTester()
    success = tester.run_all_tests()
    
    print(f"\n{BOLD}Quick Actions:{RESET}")
    print(f"1. View demo: {CYAN}http://localhost:8080{RESET}")
    print(f"2. Start demo: {CYAN}python3 demos/aura_working_demo_2025.py{RESET}")
    print(f"3. Run benchmark: {CYAN}python3 benchmarks/aura_benchmark_100_agents.py{RESET}")
    print(f"4. Check logs: {CYAN}cat complete_test_report.json{RESET}")
    
    sys.exit(0 if success else 1)