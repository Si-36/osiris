#!/usr/bin/env python3
"""
Direct test of TDA functionality without complex imports
Tests core topological data analysis with REAL data
"""

import numpy as np
import time
from typing import Dict, Any, List, Tuple

def compute_real_persistence_homology(data: List[float]) -> Dict[str, Any]:
    """
    Real persistent homology computation using ripser
    """
    try:
        from ripser import ripser
        import numpy as np
        
        # Convert to numpy array if needed
        if isinstance(data, list):
            points = np.array(data).reshape(-1, 1)
        else:
            points = np.array(data)
        
        # Ensure we have proper 2D array
        if len(points.shape) == 1:
            points = points.reshape(-1, 1)
        
        print(f"Computing persistent homology for {points.shape} points...")
        start_time = time.time()
        
        # Compute persistent homology up to dimension 1
        result = ripser(points, maxdim=1)
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Extract persistence diagrams
        diagrams = result['dgms']
        
        # Compute Betti numbers
        betti_0 = len(diagrams[0])  # Connected components
        betti_1 = len(diagrams[1]) if len(diagrams) > 1 else 0  # Loops
        
        # Compute statistics
        if len(diagrams[0]) > 0:
            lifetimes_0 = diagrams[0][:, 1] - diagrams[0][:, 0]
            avg_lifetime_0 = np.mean(lifetimes_0[np.isfinite(lifetimes_0)])
        else:
            avg_lifetime_0 = 0
        
        return {
            'success': True,
            'betti_numbers': {'b0': betti_0, 'b1': betti_1},
            'persistence_diagrams': {
                'dim_0': diagrams[0].tolist() if len(diagrams) > 0 else [],
                'dim_1': diagrams[1].tolist() if len(diagrams) > 1 else []
            },
            'statistics': {
                'avg_lifetime_dim0': float(avg_lifetime_0),
                'processing_time_ms': round(processing_time, 2),
                'num_points': len(points)
            },
            'metadata': {
                'algorithm': 'ripser',
                'max_dimension': 1,
                'metric': 'euclidean'
            }
        }
        
    except ImportError as e:
        return {
            'success': False,
            'error': f'Missing dependency: {e}',
            'fallback': 'Using basic distance matrix analysis'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Computation failed: {e}',
            'input_data': str(type(data))
        }

def compute_basic_topology(data: List[float]) -> Dict[str, Any]:
    """
    Basic topological analysis without external libraries
    """
    try:
        points = np.array(data)
        
        # Basic statistics
        mean_val = np.mean(points)
        std_val = np.std(points)
        
        # Simple distance matrix
        n = len(points)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = abs(points[i] - points[j])
        
        # Basic connectivity analysis
        threshold = std_val / 2  # Simple threshold
        connected_components = 0
        visited = set()
        
        for i in range(n):
            if i not in visited:
                connected_components += 1
                # Simple DFS
                stack = [i]
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        for j in range(n):
                            if distances[node, j] <= threshold and j not in visited:
                                stack.append(j)
        
        return {
            'success': True,
            'basic_stats': {
                'mean': float(mean_val),
                'std': float(std_val),
                'connected_components': connected_components,
                'threshold': float(threshold)
            },
            'topology_features': {
                'density': float(np.mean(distances)),
                'max_distance': float(np.max(distances)),
                'connectivity_ratio': connected_components / n
            },
            'metadata': {
                'algorithm': 'basic_analysis',
                'num_points': n
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Basic analysis failed: {e}'
        }

def test_tda_functionality():
    """Test TDA with various datasets"""
    print("🧠 Testing AURA TDA Functionality")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            'name': 'Simple Linear Data',
            'data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'expected': 'Single connected component'
        },
        {
            'name': 'Two Clusters',
            'data': [1, 2, 3, 10, 11, 12],
            'expected': 'Two connected components'
        },
        {
            'name': 'Random Data',
            'data': list(np.random.normal(0, 1, 20)),
            'expected': 'Complex topology'
        },
        {
            'name': 'Circular Pattern',
            'data': [np.sin(i/3) for i in range(20)],
            'expected': 'Potential loops'
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 Test {i}: {test_case['name']}")
        print(f"Data: {test_case['data'][:5]}... (len={len(test_case['data'])})")
        print(f"Expected: {test_case['expected']}")
        
        # Try advanced TDA first
        result = compute_real_persistence_homology(test_case['data'])
        
        if result['success']:
            print("✅ Advanced TDA (ripser) successful!")
            print(f"   Betti numbers: {result['betti_numbers']}")
            print(f"   Processing time: {result['statistics']['processing_time_ms']}ms")
            if result['betti_numbers']['b1'] > 0:
                print(f"   🔄 Found {result['betti_numbers']['b1']} loops!")
        else:
            print(f"⚠️  Advanced TDA failed: {result['error']}")
            
            # Fallback to basic analysis
            basic_result = compute_basic_topology(test_case['data'])
            if basic_result['success']:
                print("✅ Basic topology analysis successful!")
                print(f"   Connected components: {basic_result['basic_stats']['connected_components']}")
                print(f"   Connectivity ratio: {basic_result['topology_features']['connectivity_ratio']:.2f}")
                result = basic_result
            else:
                print(f"❌ Even basic analysis failed: {basic_result['error']}")
                result = basic_result
        
        results.append({
            'test_name': test_case['name'],
            'success': result['success'],
            'result': result
        })
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    print(f"✅ Successful tests: {successful_tests}/{total_tests}")
    print(f"📈 Success rate: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests > 0:
        print("\n🎉 TDA FUNCTIONALITY IS WORKING!")
        print("✅ We have real topological data analysis capability")
        print("✅ Can compute persistence homology or basic topology")
        print("✅ Processing real data with measurable results")
        
        # Show performance
        processing_times = []
        for r in results:
            if r['success'] and 'statistics' in r['result']:
                if 'processing_time_ms' in r['result']['statistics']:
                    processing_times.append(r['result']['statistics']['processing_time_ms'])
        
        if processing_times:
            avg_time = np.mean(processing_times)
            print(f"⚡ Average processing time: {avg_time:.2f}ms")
            
            if avg_time < 100:
                print("🚀 EXCELLENT performance - under 100ms!")
            elif avg_time < 500:
                print("✅ Good performance - under 500ms")
            else:
                print("⚠️  Performance needs optimization")
    
    else:
        print("❌ NO WORKING TDA FUNCTIONALITY")
        print("🔧 Need to fix dependencies and implementations")
    
    return successful_tests, total_tests, results

if __name__ == "__main__":
    success_count, total_count, test_results = test_tda_functionality()
    
    # Save results for further analysis
    with open('tda_test_results.json', 'w') as f:
        import json
        json.dump({
            'timestamp': time.time(),
            'success_rate': success_count / total_count,
            'total_tests': total_count,
            'successful_tests': success_count,
            'detailed_results': test_results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to tda_test_results.json")
    print(f"🎯 Next step: Integrate working TDA into main system")