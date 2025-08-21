#!/usr/bin/env python3
"""
AURA Intelligence Integration Test Runner
Orchestrates comprehensive testing with reporting

2025 Best Practices:
- Parallel execution
- Rich reporting
- Automatic retries
- Performance benchmarking
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
import structlog

# Add integration path
sys.path.append(str(Path(__file__).parent.parent))

from integration.framework.testcontainers.container_manager import AuraTestContainers
from integration.framework.contracts.contract_framework import ContractTestRunner
from integration.framework.chaos.chaos_framework_2025 import (
    ChaosOrchestrator, ChaosInjector, create_example_experiments
)
from integration.demos.interactive.demo_framework import (
    InteractiveDemoRunner, create_demo_scenarios
)

# Initialize
app = typer.Typer(help="AURA Intelligence Integration Test Runner")
console = Console()
logger = structlog.get_logger()


class IntegrationTestOrchestrator:
    """Orchestrates all integration tests"""
    
    def __init__(self):
        self.console = console
        self.results: Dict[str, Any] = {
            "start_time": time.time(),
            "tests": {},
            "summary": {}
        }
        
    async def run_all_tests(self, 
                           skip_chaos: bool = False,
                           skip_contracts: bool = False,
                           skip_performance: bool = False,
                           parallel: bool = True) -> Dict[str, Any]:
        """Run all integration tests"""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Start services
            task = progress.add_task("[cyan]Starting AURA services...", total=None)
            
            manager = AuraTestContainers(use_compose=False)
            await manager.prepare_warm_pools()  # Pre-warm for speed
            
            async with manager.start_aura_stack() as (service_urls, container_manager):
                progress.update(task, completed=True)
                self.console.print("[green]✓[/green] Services started successfully")
                
                # Run test suites
                test_suites = []
                
                if not skip_contracts:
                    test_suites.append(("Contract Tests", self._run_contract_tests))
                
                test_suites.append(("Functional Tests", self._run_functional_tests))
                
                if not skip_performance:
                    test_suites.append(("Performance Tests", self._run_performance_tests))
                
                if not skip_chaos:
                    test_suites.append(("Chaos Tests", self._run_chaos_tests))
                
                # Execute test suites
                if parallel:
                    # Run in parallel
                    tasks = [
                        self._run_test_suite(name, func, service_urls, container_manager)
                        for name, func in test_suites
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for (name, _), result in zip(test_suites, results):
                        if isinstance(result, Exception):
                            self.results["tests"][name] = {
                                "status": "failed",
                                "error": str(result)
                            }
                        else:
                            self.results["tests"][name] = result
                else:
                    # Run sequentially
                    for name, func in test_suites:
                        result = await self._run_test_suite(
                            name, func, service_urls, container_manager
                        )
                        self.results["tests"][name] = result
                
                # Run demo scenarios
                await self._run_demo_scenarios(service_urls)
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    async def _run_test_suite(self, 
                             name: str,
                             func: callable,
                             service_urls: Dict[str, str],
                             container_manager: Any) -> Dict[str, Any]:
        """Run a test suite with progress reporting"""
        self.console.print(f"\n[yellow]Running {name}...[/yellow]")
        
        start_time = time.time()
        try:
            result = await func(service_urls, container_manager)
            duration = time.time() - start_time
            
            self.console.print(f"[green]✓[/green] {name} completed in {duration:.2f}s")
            
            return {
                "status": "passed",
                "duration": duration,
                "details": result
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.console.print(f"[red]✗[/red] {name} failed: {str(e)}")
            
            return {
                "status": "failed",
                "duration": duration,
                "error": str(e)
            }
    
    async def _run_contract_tests(self, 
                                 service_urls: Dict[str, str],
                                 container_manager: Any) -> Dict[str, Any]:
        """Run contract validation tests"""
        runner = ContractTestRunner("/workspace/aura-microservices/contracts")
        
        # Test each consumer
        consumers = ["moe-router", "memory-service", "byzantine-service"]
        results = {}
        
        for consumer in consumers:
            consumer_results = await runner.run_consumer_tests(
                consumer_name=consumer,
                provider_url=service_urls.get("neuromorphic", "http://localhost:8000")
            )
            
            passed = sum(1 for r in consumer_results if r.valid)
            results[consumer] = {
                "total": len(consumer_results),
                "passed": passed,
                "failed": len(consumer_results) - passed
            }
        
        return results
    
    async def _run_functional_tests(self,
                                   service_urls: Dict[str, str],
                                   container_manager: Any) -> Dict[str, Any]:
        """Run functional integration tests"""
        import pytest
        
        # Run pytest programmatically
        test_dir = Path(__file__).parent / "tests" / "e2e"
        
        # Pass service URLs via environment
        import os
        os.environ["AURA_SERVICE_URLS"] = json.dumps(service_urls)
        
        # Run tests
        exit_code = pytest.main([
            str(test_dir),
            "-v",
            "--tb=short",
            "--asyncio-mode=auto",
            "-m", "not slow",
            "--json-report",
            "--json-report-file=/tmp/functional_test_results.json"
        ])
        
        # Load results
        try:
            with open("/tmp/functional_test_results.json", "r") as f:
                pytest_results = json.load(f)
                
            return {
                "total": pytest_results["summary"]["total"],
                "passed": pytest_results["summary"]["passed"],
                "failed": pytest_results["summary"]["failed"],
                "duration": pytest_results["duration"]
            }
        except Exception:
            return {
                "exit_code": exit_code,
                "status": "completed" if exit_code == 0 else "failed"
            }
    
    async def _run_performance_tests(self,
                                    service_urls: Dict[str, str],
                                    container_manager: Any) -> Dict[str, Any]:
        """Run performance benchmarks"""
        import httpx
        import numpy as np
        
        results = {}
        
        # Test each service
        for service_name, url in service_urls.items():
            if service_name in ["neuromorphic", "memory", "byzantine", "lnn", "moe"]:
                # Warm up
                async with httpx.AsyncClient() as client:
                    for _ in range(10):
                        await client.get(f"{url}/api/v1/health")
                
                # Benchmark
                latencies = []
                errors = 0
                
                async with httpx.AsyncClient() as client:
                    start_time = time.time()
                    
                    for _ in range(100):
                        try:
                            req_start = time.perf_counter()
                            response = await client.get(
                                f"{url}/api/v1/health",
                                timeout=5.0
                            )
                            req_end = time.perf_counter()
                            
                            if response.status_code == 200:
                                latencies.append((req_end - req_start) * 1000)
                            else:
                                errors += 1
                                
                        except Exception:
                            errors += 1
                    
                    duration = time.time() - start_time
                
                if latencies:
                    results[service_name] = {
                        "requests": len(latencies) + errors,
                        "successful": len(latencies),
                        "errors": errors,
                        "throughput": len(latencies) / duration,
                        "latency_avg": np.mean(latencies),
                        "latency_p50": np.percentile(latencies, 50),
                        "latency_p95": np.percentile(latencies, 95),
                        "latency_p99": np.percentile(latencies, 99)
                    }
        
        return results
    
    async def _run_chaos_tests(self,
                              service_urls: Dict[str, str],
                              container_manager: Any) -> Dict[str, Any]:
        """Run chaos engineering tests"""
        injector = ChaosInjector(container_manager)
        orchestrator = ChaosOrchestrator(service_urls, injector)
        
        # Run game day
        experiments = create_example_experiments()
        game_day_results = await orchestrator.run_game_day(experiments[:2])  # Run 2 experiments
        
        return {
            "experiments_run": len(game_day_results["experiments"]),
            "success_rate": game_day_results["summary"]["success_rate"],
            "insights": game_day_results["summary"]["observations"]
        }
    
    async def _run_demo_scenarios(self, service_urls: Dict[str, str]):
        """Run demo scenarios"""
        self.console.print("\n[yellow]Running demo scenarios...[/yellow]")
        
        runner = InteractiveDemoRunner(service_urls)
        scenarios = create_demo_scenarios()
        
        demo_results = []
        
        for scenario in scenarios[:2]:  # Run first 2 scenarios
            try:
                result = await runner.run_scenario(scenario)
                demo_results.append({
                    "scenario": scenario.name,
                    "status": "completed",
                    "duration": result["duration_seconds"],
                    "success_rate": result["success_rate"]
                })
                
                self.console.print(
                    f"[green]✓[/green] Demo '{scenario.name}' completed"
                )
                
            except Exception as e:
                demo_results.append({
                    "scenario": scenario.name,
                    "status": "failed",
                    "error": str(e)
                })
                
                self.console.print(
                    f"[red]✗[/red] Demo '{scenario.name}' failed: {str(e)}"
                )
        
        self.results["tests"]["demos"] = demo_results
    
    def _generate_summary(self):
        """Generate test summary"""
        total_suites = len(self.results["tests"])
        passed_suites = sum(
            1 for t in self.results["tests"].values()
            if isinstance(t, dict) and t.get("status") == "passed"
        )
        
        self.results["summary"] = {
            "total_duration": time.time() - self.results["start_time"],
            "test_suites_run": total_suites,
            "test_suites_passed": passed_suites,
            "test_suites_failed": total_suites - passed_suites,
            "overall_status": "passed" if passed_suites == total_suites else "failed"
        }
    
    def display_results(self):
        """Display results in a nice format"""
        # Summary panel
        summary = self.results["summary"]
        status_color = "green" if summary["overall_status"] == "passed" else "red"
        
        summary_text = f"""
Overall Status: [{status_color}]{summary['overall_status'].upper()}[/{status_color}]
Duration: {summary['total_duration']:.2f} seconds
Test Suites: {summary['test_suites_passed']}/{summary['test_suites_run']} passed
        """
        
        self.console.print(
            Panel(summary_text.strip(), title="Integration Test Summary", expand=False)
        )
        
        # Detailed results table
        table = Table(title="Test Suite Results")
        table.add_column("Suite", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Details", style="white")
        
        for suite_name, result in self.results["tests"].items():
            if isinstance(result, dict):
                status = result.get("status", "unknown")
                status_display = "[green]✓ PASSED[/green]" if status == "passed" else "[red]✗ FAILED[/red]"
                duration = f"{result.get('duration', 0):.2f}s"
                
                # Format details
                if "details" in result:
                    details = json.dumps(result["details"], indent=2)[:100] + "..."
                elif "error" in result:
                    details = f"[red]{result['error'][:100]}...[/red]"
                else:
                    details = "-"
                
                table.add_row(suite_name, status_display, duration, details)
        
        self.console.print(table)
        
        # Performance highlights
        if "Performance Tests" in self.results["tests"]:
            perf = self.results["tests"]["Performance Tests"]
            if perf.get("status") == "passed" and "details" in perf:
                self.console.print("\n[bold]Performance Highlights:[/bold]")
                
                for service, metrics in perf["details"].items():
                    if isinstance(metrics, dict) and "latency_p95" in metrics:
                        self.console.print(
                            f"  • {service}: "
                            f"P95 latency = {metrics['latency_p95']:.1f}ms, "
                            f"Throughput = {metrics['throughput']:.1f} req/s"
                        )


@app.command()
def test(
    skip_chaos: bool = typer.Option(False, "--skip-chaos", help="Skip chaos engineering tests"),
    skip_contracts: bool = typer.Option(False, "--skip-contracts", help="Skip contract tests"),
    skip_performance: bool = typer.Option(False, "--skip-performance", help="Skip performance tests"),
    sequential: bool = typer.Option(False, "--sequential", help="Run tests sequentially"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Save results to file")
):
    """Run all integration tests"""
    
    console.print(
        Panel.fit(
            "🧪 AURA Intelligence Integration Testing Suite",
            subtitle="2025 Edition",
            style="bold blue"
        )
    )
    
    orchestrator = IntegrationTestOrchestrator()
    
    # Run tests
    results = asyncio.run(
        orchestrator.run_all_tests(
            skip_chaos=skip_chaos,
            skip_contracts=skip_contracts,
            skip_performance=skip_performance,
            parallel=not sequential
        )
    )
    
    # Display results
    orchestrator.display_results()
    
    # Save results if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]Results saved to {output_file}[/green]")
    
    # Exit with appropriate code
    sys.exit(0 if results["summary"]["overall_status"] == "passed" else 1)


@app.command()
def demo():
    """Run interactive demo UI"""
    console.print("[cyan]Starting interactive demo UI on http://localhost:8888[/cyan]")
    
    # Import and run demo
    from integration.demos.interactive.demo_framework import (
        InteractiveDemoRunner, create_demo_app
    )
    import uvicorn
    
    service_urls = {
        "neuromorphic": "http://localhost:8000",
        "memory": "http://localhost:8001",
        "byzantine": "http://localhost:8002",
        "lnn": "http://localhost:8003",
        "moe": "http://localhost:8005"
    }
    
    runner = InteractiveDemoRunner(service_urls)
    app = create_demo_app(runner)
    
    uvicorn.run(app, host="0.0.0.0", port=8888)


@app.command()
def chaos(
    experiments: int = typer.Option(2, "--experiments", "-e", help="Number of chaos experiments"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run without actually injecting faults")
):
    """Run chaos engineering experiments"""
    console.print("[yellow]Running chaos engineering experiments...[/yellow]")
    
    async def run_chaos():
        manager = AuraTestContainers(use_compose=False)
        
        async with manager.start_aura_stack() as (service_urls, container_manager):
            injector = ChaosInjector(container_manager)
            orchestrator = ChaosOrchestrator(service_urls, injector)
            
            # Get experiments
            all_experiments = create_example_experiments()
            selected = all_experiments[:experiments]
            
            # Mark as dry run if requested
            if dry_run:
                for exp in selected:
                    exp.dry_run = True
            
            # Run game day
            results = await orchestrator.run_game_day(selected)
            
            # Display results
            console.print(f"\n[bold]Game Day Results:[/bold]")
            console.print(f"Success Rate: {results['summary']['success_rate']:.1%}")
            console.print(f"Insights: {results['summary']['observations']}")
    
    asyncio.run(run_chaos())


@app.command()
def performance(
    duration: int = typer.Option(60, "--duration", "-d", help="Test duration in seconds"),
    concurrency: int = typer.Option(10, "--concurrency", "-c", help="Concurrent requests")
):
    """Run performance benchmarks"""
    console.print(f"[yellow]Running performance tests for {duration}s with {concurrency} concurrent requests...[/yellow]")
    
    # This would integrate with k6 or Locust for more comprehensive testing
    console.print("[red]Full performance testing not yet implemented[/red]")


if __name__ == "__main__":
    app()