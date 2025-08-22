#!/usr/bin/env python3
"""
Verify all AURA components are properly created
"""

import sys
sys.path.insert(0, 'src')

from aura.core.system import AURASystem
from aura.core.config import AURAConfig

# Colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

def main():
    print(f"\n{BOLD}{BLUE}Verifying AURA Components...{RESET}\n")
    
    try:
        # Create system
        config = AURAConfig()
        system = AURASystem(config)
        
        # Get all components
        components = system.get_all_components()
        
        # Check each category
        print(f"{BOLD}Component Counts:{RESET}")
        print(f"  TDA Algorithms: {len(components['tda_algorithms'])}/112")
        print(f"  Neural Networks: {len(components['neural_networks'])}/10")
        print(f"  Memory Components: {len(components['memory_components'])}/40")
        print(f"  Agents: {len(components['agents'])}/100")
        print(f"  Consensus Protocols: {len(components['consensus_protocols'])}/5")
        print(f"  Neuromorphic: {len(components['neuromorphic_components'])}/8")
        print(f"  Infrastructure: {len(components['infrastructure'])}/51")
        
        # Check agent details
        print(f"\n{BOLD}Agent Details:{RESET}")
        agent_types = {
            "pattern_ia": 0,
            "anomaly_ia": 0,
            "trend_ia": 0,
            "context_ia": 0,
            "feature_ia": 0,
            "resource_ca": 0,
            "schedule_ca": 0,
            "balance_ca": 0,
            "optimize_ca": 0,
            "coord_ca": 0,
        }
        
        for agent in components['agents']:
            for agent_type in agent_types:
                if agent.startswith(agent_type):
                    agent_types[agent_type] += 1
        
        print(f"\n  Information Agents:")
        for agent_type in ["pattern_ia", "anomaly_ia", "trend_ia", "context_ia", "feature_ia"]:
            count = agent_types[agent_type]
            status = f"{GREEN}✓{RESET}" if count == 10 else f"{RED}✗{RESET}"
            print(f"    {status} {agent_type}: {count}/10")
        
        print(f"\n  Control Agents:")
        for agent_type in ["resource_ca", "schedule_ca", "balance_ca", "optimize_ca", "coord_ca"]:
            count = agent_types[agent_type]
            status = f"{GREEN}✓{RESET}" if count == 10 else f"{RED}✗{RESET}"
            print(f"    {status} {agent_type}: {count}/10")
        
        # Total component count
        total = sum(len(v) for v in components.values())
        print(f"\n{BOLD}Total Components: {total}/226{RESET}")
        
        # Note: 226 because we have 213 main components + 13 extra (5 consensus + 8 neuromorphic counted twice)
        
        # Sample some components
        print(f"\n{BOLD}Sample Components:{RESET}")
        print(f"  First 5 TDA algorithms: {components['tda_algorithms'][:5]}")
        print(f"  First 5 agents: {components['agents'][:5]}")
        print(f"  Infrastructure sample: {components['infrastructure'][:5]}")
        
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()