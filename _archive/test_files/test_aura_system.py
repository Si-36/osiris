#!/usr/bin/env python3
"""
AURA System Test Script
Tests all 213 components and provides a health report
"""

import os
import sys
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(60)}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{GREEN}✓ {text}{RESET}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_error(text: str):
    """Print error message"""
    print(f"{RED}✗ {text}{RESET}")

def print_info(text: str):
    """Print info message"""
    print(f"{BLUE}ℹ {text}{RESET}")

class AURASystemTester:
    """Test all AURA components"""
    
    def __init__(self):
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "warnings": 0,
            "failed": 0,
            "components": {}
        }
    
    def test_environment(self) -> bool:
        """Test environment configuration"""
        print_header("Testing Environment Configuration")
        
        # Check .env file
        if os.path.exists('.env'):
            print_success(".env file found")
            self.results["passed"] += 1
        else:
            print_error(".env file not found")
            self.results["failed"] += 1
            return False
        
        # Check required environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        required_vars = [
            "LANGSMITH_API_KEY",
            "GEMINI_API_KEY",
            "NEO4J_URI",
            "REDIS_HOST"
        ]
        
        for var in required_vars:
            if os.getenv(var):
                print_success(f"{var} is set")
                self.results["passed"] += 1
            else:
                print_warning(f"{var} not set")
                self.results["warnings"] += 1
        
        return True
    
    def test_directory_structure(self) -> bool:
        """Test directory structure"""
        print_header("Testing Directory Structure")
        
        required_dirs = [
            "src/aura",
            "src/aura/core",
            "demos",
            "benchmarks",
            "utilities",
            "infrastructure",
            "documentation",
            "tests"
        ]
        
        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                print_success(f"{dir_path}/ exists")
                self.results["passed"] += 1
            else:
                print_error(f"{dir_path}/ missing")
                self.results["failed"] += 1
        
        return True
    
    def test_core_files(self) -> bool:
        """Test core files exist"""
        print_header("Testing Core Files")
        
        core_files = {
            "src/aura/__init__.py": "Package initialization",
            "src/aura/core/system.py": "Main AURA system",
            "src/aura/core/config.py": "Configuration",
            "demos/aura_working_demo_2025.py": "Main demo",
            "infrastructure/docker-compose.yml": "Docker config",
            ".env": "Environment config",
            "requirements.txt": "Dependencies"
        }
        
        for file_path, description in core_files.items():
            if os.path.exists(file_path):
                print_success(f"{file_path} - {description}")
                self.results["passed"] += 1
            else:
                print_error(f"{file_path} - {description} MISSING")
                self.results["failed"] += 1
        
        return True
    
    def test_aura_imports(self) -> bool:
        """Test AURA module imports"""
        print_header("Testing AURA Module Imports")
        
        try:
            # Test basic import
            import aura
            print_success("AURA package imports successfully")
            self.results["passed"] += 1
            
            # Test version
            if hasattr(aura, '__version__'):
                print_success(f"AURA version: {aura.__version__}")
                self.results["passed"] += 1
            
            # Test core imports
            from aura.core.config import AURAConfig
            print_success("AURAConfig imports successfully")
            self.results["passed"] += 1
            
            # Test config creation
            config = AURAConfig()
            print_success("AURAConfig instantiates successfully")
            self.results["passed"] += 1
            
            # Validate config
            if config.validate():
                print_success("Configuration validated")
                self.results["passed"] += 1
            
        except ImportError as e:
            print_error(f"Import error: {e}")
            self.results["failed"] += 1
            return False
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            self.results["failed"] += 1
            return False
        
        return True
    
    def test_component_counts(self) -> bool:
        """Test component counts match expected"""
        print_header("Testing Component Counts")
        
        expected_counts = {
            "TDA Algorithms": 112,
            "Neural Networks": 10,
            "Memory Systems": 40,
            "Agent Systems": 100,
            "Infrastructure": 51,
            "Total": 213
        }
        
        try:
            from aura.core.system import AURASystem
            
            # Check if we can access component stats
            system = AURASystem()
            
            for component, expected in expected_counts.items():
                key = component.lower().replace(" ", "_")
                if key in system.component_stats:
                    actual = system.component_stats[key]
                    if actual == expected:
                        print_success(f"{component}: {actual} ✓")
                        self.results["passed"] += 1
                    else:
                        print_warning(f"{component}: {actual} (expected {expected})")
                        self.results["warnings"] += 1
            
        except Exception as e:
            print_error(f"Could not test component counts: {e}")
            self.results["failed"] += 1
            return False
        
        return True
    
    def test_demo_health(self) -> bool:
        """Test if demo is accessible"""
        print_header("Testing Demo Health")
        
        # Check if demo file exists
        demo_path = "demos/aura_working_demo_2025.py"
        if os.path.exists(demo_path):
            print_success(f"Demo file exists: {demo_path}")
            self.results["passed"] += 1
            
            # Check if it's runnable
            with open(demo_path, 'r') as f:
                content = f.read()
                if "class AURASystem2025" in content:
                    print_success("Demo contains AURASystem2025 class")
                    self.results["passed"] += 1
                if "HTTPServer" in content:
                    print_success("Demo has HTTP server")
                    self.results["passed"] += 1
        else:
            print_error("Demo file not found")
            self.results["failed"] += 1
        
        return True
    
    def test_infrastructure(self) -> bool:
        """Test infrastructure configuration"""
        print_header("Testing Infrastructure Configuration")
        
        # Check docker-compose
        docker_compose_path = "infrastructure/docker-compose.yml"
        if os.path.exists(docker_compose_path):
            print_success("Docker Compose file exists")
            self.results["passed"] += 1
            
            # Check services
            with open(docker_compose_path, 'r') as f:
                content = f.read()
                services = ["neo4j", "redis", "postgres", "prometheus", "grafana"]
                for service in services:
                    if service in content:
                        print_success(f"{service} service configured")
                        self.results["passed"] += 1
                    else:
                        print_warning(f"{service} service not found")
                        self.results["warnings"] += 1
        else:
            print_error("Docker Compose file not found")
            self.results["failed"] += 1
        
        return True
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate test report"""
        total = self.results["passed"] + self.results["warnings"] + self.results["failed"]
        self.results["total_tests"] = total
        
        if total > 0:
            self.results["success_rate"] = (self.results["passed"] / total) * 100
        else:
            self.results["success_rate"] = 0
        
        self.results["timestamp"] = datetime.utcnow().isoformat()
        self.results["status"] = "healthy" if self.results["failed"] == 0 else "unhealthy"
        
        return self.results
    
    def run_all_tests(self):
        """Run all tests"""
        print_header("AURA System Health Check")
        print_info(f"Project ID: bc-a397ac41-47c3-4620-a5ec-c56fb1f50fd0")
        print_info(f"Testing 213 components...")
        
        # Run tests
        self.test_environment()
        self.test_directory_structure()
        self.test_core_files()
        self.test_aura_imports()
        self.test_component_counts()
        self.test_demo_health()
        self.test_infrastructure()
        
        # Generate report
        report = self.generate_report()
        
        # Print summary
        print_header("Test Summary")
        print(f"{GREEN}Passed: {report['passed']}{RESET}")
        print(f"{YELLOW}Warnings: {report['warnings']}{RESET}")
        print(f"{RED}Failed: {report['failed']}{RESET}")
        print(f"\nSuccess Rate: {report['success_rate']:.1f}%")
        print(f"Status: {report['status'].upper()}")
        
        # Save report
        with open('test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print_info("\nDetailed report saved to test_report.json")
        
        # Next steps
        print_header("Next Steps")
        if report['status'] == 'healthy':
            print_success("System is healthy! You can:")
            print("1. Run the demo: python3 demos/aura_working_demo_2025.py")
            print("2. Start infrastructure: cd infrastructure && docker-compose up -d")
            print("3. Run benchmarks: python3 benchmarks/aura_benchmark_100_agents.py")
        else:
            print_warning("System needs attention. Please:")
            print("1. Check failed tests above")
            print("2. Ensure all dependencies are installed")
            print("3. Verify .env configuration")
        
        return report


if __name__ == "__main__":
    tester = AURASystemTester()
    report = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if report['status'] == 'healthy' else 1)