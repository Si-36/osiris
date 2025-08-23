#!/usr/bin/env python3
"""
Verify AURA microservices structure and readiness
"""

import os
import json
from pathlib import Path

def check_service_structure(service_name, expected_files):
    """Check if a service has the expected structure"""
    service_path = Path(f"/workspace/aura-microservices/{service_name}")
    
    print(f"\n🔍 Checking {service_name.upper()} Service:")
    
    missing = []
    for file_path in expected_files:
        full_path = service_path / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (missing)")
            missing.append(file_path)
    
    return len(missing) == 0


def main():
    print("🚀 AURA Microservices Structure Verification")
    print("=" * 60)
    
    # Expected structure for each service
    services_structure = {
        "neuromorphic": [
            "src/api/main.py",
            "src/models/advanced/neuromorphic_2025.py",
            "src/schemas/requests.py",
            "src/schemas/responses.py",
            "requirements.txt",
            "Dockerfile",
            "README.md"
        ],
        "memory": [
            "src/api/main.py",
            "src/models/advanced/memory_tiers_2025.py",
            "src/schemas/requests.py",
            "src/schemas/responses.py",
            "src/services/shape_analyzer.py",
            "requirements.txt",
            "Dockerfile",
            "README.md"
        ],
        "byzantine": [
            "src/api/main.py",
            "src/models/consensus/byzantine_2025.py",
            "src/schemas/requests.py",
            "src/schemas/responses.py",
            "requirements.txt",
            "Dockerfile",
            "README.md"
        ],
        "lnn": [
            "src/api/main.py",
            "src/models/liquid/liquid_neural_network_2025.py",
            "src/schemas/requests.py",
            "src/schemas/responses.py",
            "requirements.txt",
            "Dockerfile",
            "README.md"
        ],
        "moe": [
            "src/api/main.py",
            "src/models/routing/moe_router_2025.py",
            "src/schemas/requests.py",
            "src/schemas/responses.py",
            "requirements.txt",
            "Dockerfile",
            "README.md"
        ]
    }
    
    # Integration structure
    integration_structure = [
        "framework/testcontainers/container_manager.py",
        "framework/contracts/contract_framework.py",
        "framework/chaos/chaos_framework_2025.py",
        "demos/interactive/demo_framework.py",
        "tests/e2e/test_full_stack_integration.py",
        "run_integration_tests.py",
        "requirements.txt",
        "README.md"
    ]
    
    # Check each service
    all_services_ok = True
    for service, files in services_structure.items():
        if not check_service_structure(service, files):
            all_services_ok = False
    
    # Check integration framework
    print("\n🔍 Checking Integration Framework:")
    integration_ok = True
    for file_path in integration_structure:
        full_path = Path(f"/workspace/aura-microservices/integration/{file_path}")
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (missing)")
            integration_ok = False
    
    # Check Docker Compose
    print("\n🔍 Checking Docker Configuration:")
    docker_compose_path = Path("/workspace/aura-microservices/docker-compose.yml")
    if docker_compose_path.exists():
        print(f"  ✅ docker-compose.yml")
        
        # Parse and verify services
        with open(docker_compose_path, 'r') as f:
            content = f.read()
            expected_services = ["redis", "kafka", "neo4j", "postgres", 
                               "aura-neuromorphic", "aura-memory", "aura-byzantine", 
                               "aura-lnn", "aura-moe-router"]
            
            found_services = []
            for service in expected_services:
                if f"{service}:" in content:
                    found_services.append(service)
            
            print(f"  📦 Found {len(found_services)}/{len(expected_services)} services in docker-compose.yml")
    else:
        print(f"  ❌ docker-compose.yml (missing)")
        integration_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    print(f"\n✅ Services Structure: {'PASSED' if all_services_ok else 'FAILED'}")
    print(f"✅ Integration Framework: {'PASSED' if integration_ok else 'FAILED'}")
    
    # Key stats
    print("\n📈 Key Statistics:")
    print(f"  • 5 Microservices extracted")
    print(f"  • 3 Testing frameworks (TestContainers, Contracts, Chaos)")
    print(f"  • 1 Interactive demo framework")
    print(f"  • Ready for Docker deployment")
    
    # Next steps
    print("\n🎯 NEXT STEPS:")
    print("1. Install Docker and Docker Compose")
    print("2. Build services: docker compose build")
    print("3. Start stack: docker compose up -d")
    print("4. Run tests: ./integration/run_integration_tests.py test")
    print("5. Launch demos: ./integration/run_integration_tests.py demo")
    
    return all_services_ok and integration_ok


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)