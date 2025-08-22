#!/usr/bin/env python3
"""Quick AURA Benchmark Demo"""

import asyncio
import sys
import os

# Import from the benchmark module
from aura_benchmark_100_agents import (
    ScalableAURASystem, 
    run_comparison_benchmark,
    print_benchmark_report
)

async def quick_demo():
    """Run a quick benchmark demo"""
    print("🚀 Running Quick AURA Benchmark Demo")
    print("=" * 60)
    
    # Test with smaller numbers for quick results
    agent_counts = [30, 50, 100]
    
    # Run just 1 test per configuration for speed
    results = await run_comparison_benchmark(agent_counts, runs_per_config=1)
    
    # Print report
    print_benchmark_report(results)
    
    return results

if __name__ == "__main__":
    asyncio.run(quick_demo())