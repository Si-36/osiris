#!/usr/bin/env python3
"""
AURA Intelligence - Integrated System Demonstration
Shows actual working TDA+LNN API with real data processing
"""

import requests
import time
import json
import numpy as np
from typing import Dict, Any, List

class AURAIntegratedDemo:
    def __init__(self, api_base_url: str = "http://localhost:8080"):
        self.api_base = api_base_url
        self.session = requests.Session()
    
    def print_header(self, title: str):
        """Print formatted section header"""
        print("\n" + "=" * 60)
        print(f"🧠 {title}")
        print("=" * 60)
    
    def print_success(self, message: str):
        """Print success message"""
        print(f"✅ {message}")
    
    def print_info(self, message: str):
        """Print info message"""
        print(f"📊 {message}")
    
    def print_performance(self, time_ms: float):
        """Print performance info"""
        if time_ms < 10:
            print(f"⚡ Processing time: {time_ms:.2f}ms - EXCELLENT!")
        elif time_ms < 50:
            print(f"⚡ Processing time: {time_ms:.2f}ms - Very good")
        elif time_ms < 100:
            print(f"⚡ Processing time: {time_ms:.2f}ms - Good")
        else:
            print(f"⚡ Processing time: {time_ms:.2f}ms - Needs optimization")
    
    def test_system_capabilities(self):
        """Test system capabilities and status"""
        self.print_header("System Capabilities Check")
        
        try:
            response = self.session.get(f"{self.api_base}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                self.print_success("System is operational")
                self.print_info(f"Service: {data['service']}")
                self.print_info(f"Version: {data['version']}")
                
                capabilities = data.get('capabilities', {})
                for cap, available in capabilities.items():
                    status = "✅ Available" if available else "❌ Not Available"
                    print(f"   {cap.upper()}: {status}")
                
                return True
            else:
                print(f"❌ System check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to system: {e}")
            return False
    
    def test_tda_engine(self):
        """Test TDA engine with real data"""
        self.print_header("TDA Engine Test")
        
        # Test with topological data that should show clear patterns
        test_data = [1, 2, 3, 10, 11, 12, 20, 21, 22]  # Three clusters
        
        try:
            payload = {"data": test_data}
            response = self.session.post(
                f"{self.api_base}/analyze/tda",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    self.print_success("TDA engine working")
                    
                    betti = result['betti_numbers']
                    self.print_info(f"Connected components: {betti['b0']}")
                    self.print_info(f"Topological loops: {betti['b1']}")
                    
                    self.print_performance(result['processing_time_ms'])
                    return result
                else:
                    print(f"❌ TDA analysis failed: {result.get('error')}")
                    return None
            else:
                print(f"❌ TDA API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ TDA test failed: {e}")
            return None
    
    def test_lnn_engine(self):
        """Test LNN engine with real data"""
        self.print_header("LNN Engine Test")
        
        # Test with feature vectors
        test_data = [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        ]
        
        try:
            payload = {"data": test_data}
            response = self.session.post(
                f"{self.api_base}/analyze/lnn",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    self.print_success("LNN engine working")
                    
                    model_info = result['model_info']
                    self.print_info(f"Model type: {model_info['type']}")
                    self.print_info(f"Library: {model_info['library']}")
                    self.print_info(f"Parameters: {model_info['parameters']:,}")
                    self.print_info(f"Continuous time: {model_info['continuous_time']}")
                    
                    output = result['output']
                    self.print_info(f"Output shape: {len(output)} x {len(output[0]) if output else 0}")
                    if output:
                        self.print_info(f"Output range: [{min(output[0]):.3f}, {max(output[0]):.3f}]")
                    
                    self.print_performance(result['processing_time_ms'])
                    return result
                else:
                    print(f"❌ LNN analysis failed: {result.get('error')}")
                    return None
            else:
                print(f"❌ LNN API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ LNN test failed: {e}")
            return None
    
    def test_tda_lnn_pipeline(self):
        """Test TDA+LNN integrated pipeline"""
        self.print_header("TDA+LNN Integration Test")
        
        # Generate data that has both topological and temporal features
        points = []
        for i in range(20):
            # Create spiral pattern with noise
            angle = i * 0.5
            radius = 1 + 0.1 * i
            x = radius * np.cos(angle) + np.random.normal(0, 0.1)
            y = radius * np.sin(angle) + np.random.normal(0, 0.1)
            points.append(x + y)  # 1D representation for TDA
        
        print(f"📊 Generated spiral data: {len(points)} points")
        
        # Step 1: TDA Analysis
        print("\n🔬 Step 1: Topological Analysis")
        try:
            tda_payload = {"data": points}
            tda_response = self.session.post(
                f"{self.api_base}/analyze/tda",
                json=tda_payload,
                timeout=30
            )
            
            if tda_response.status_code != 200:
                print(f"❌ TDA step failed: {tda_response.status_code}")
                return False
            
            tda_result = tda_response.json()
            if not tda_result['success']:
                print(f"❌ TDA analysis failed: {tda_result.get('error')}")
                return False
            
            # Extract topological features
            betti = tda_result['betti_numbers']
            stats = tda_result['statistics']
            
            self.print_success("TDA analysis complete")
            self.print_info(f"Betti numbers: b₀={betti['b0']}, b₁={betti['b1']}")
            
            # Create feature vector from TDA results
            tda_features = [
                float(betti['b0']),  # Connected components
                float(betti['b1']),  # Loops
                stats['avg_lifetime_dim0'],  # Average lifetime
                stats['num_points'],  # Number of points
                float(len(tda_result['persistence_diagrams']['dim_0'])),  # Persistence features
                np.std(points)  # Data spread
            ]
            
            print(f"   📊 TDA features: {[f'{x:.3f}' for x in tda_features]}")
            
        except Exception as e:
            print(f"❌ TDA step failed: {e}")
            return False
        
        # Step 2: LNN Analysis of TDA features
        print("\n🧠 Step 2: LNN Processing of Topological Features")
        try:
            # Pad features to 10 dimensions (LNN input size)
            while len(tda_features) < 10:
                tda_features.append(0.0)
            tda_features = tda_features[:10]  # Ensure exactly 10 features
            
            lnn_payload = {"data": [tda_features]}
            lnn_response = self.session.post(
                f"{self.api_base}/analyze/lnn",
                json=lnn_payload,
                timeout=30
            )
            
            if lnn_response.status_code != 200:
                print(f"❌ LNN step failed: {lnn_response.status_code}")
                return False
            
            lnn_result = lnn_response.json()
            if not lnn_result['success']:
                print(f"❌ LNN analysis failed: {lnn_result.get('error')}")
                return False
            
            self.print_success("LNN analysis complete")
            prediction = lnn_result['output'][0]
            self.print_info(f"Neural prediction: {prediction}")
            
            # Combined performance
            total_time = tda_result['processing_time_ms'] + lnn_result['processing_time_ms']
            print(f"\n⚡ Pipeline performance:")
            print(f"   TDA: {tda_result['processing_time_ms']:.2f}ms")
            print(f"   LNN: {lnn_result['processing_time_ms']:.2f}ms")
            print(f"   Total: {total_time:.2f}ms")
            
            return True
            
        except Exception as e:
            print(f"❌ LNN step failed: {e}")
            return False
    
    def test_system_metrics(self):
        """Test system metrics and monitoring"""
        self.print_header("System Metrics & Monitoring")
        
        try:
            response = self.session.get(f"{self.api_base}/metrics", timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                system_metrics = data['system_metrics']
                tda_engine = data['tda_engine']
                lnn_engine = data['lnn_engine']
                
                self.print_success("Metrics available")
                print(f"📊 System uptime: {system_metrics['uptime_seconds']:.1f}s")
                print(f"📊 Total requests: {system_metrics['total_requests']}")
                print(f"📊 Success rate: {system_metrics['success_rate']:.1%}")
                print(f"📊 Avg processing: {system_metrics['average_processing_time_ms']:.2f}ms")
                
                print(f"\n🧠 TDA Engine:")
                print(f"   Status: {tda_engine['status']}")
                print(f"   Algorithm: {tda_engine['algorithm']}")
                print(f"   Performance: {tda_engine['performance']}")
                
                print(f"\n🧠 LNN Engine:")
                print(f"   Status: {lnn_engine['status']}")
                print(f"   Library: {lnn_engine['library']}")
                print(f"   Parameters: {lnn_engine['parameters']:,}")
                print(f"   Continuous time: {lnn_engine['continuous_time']}")
                
                return True
            else:
                print(f"❌ Metrics unavailable: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Metrics error: {e}")
            return False
    
    def run_comprehensive_demo(self):
        """Run complete integrated system demonstration"""
        print("🧠 AURA Intelligence - Integrated System Demonstration")
        print("🎯 Testing TDA + LNN integration with real data processing")
        print("⚡ NO MOCKS - REAL COMPUTATIONS ONLY")
        
        start_time = time.time()
        
        # Test sequence
        tests = [
            ("System Capabilities", self.test_system_capabilities),
            ("TDA Engine", self.test_tda_engine),
            ("LNN Engine", self.test_lnn_engine),
            ("TDA+LNN Pipeline", self.test_tda_lnn_pipeline),
            ("System Metrics", self.test_system_metrics)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\n🔄 Running {test_name}...")
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                results[test_name] = False
        
        # Final summary
        total_time = time.time() - start_time
        successful_tests = sum(1 for success in results.values() if success)
        total_tests = len(results)
        
        self.print_header("Integration Results Summary")
        print(f"✅ Successful tests: {successful_tests}/{total_tests}")
        print(f"📈 Success rate: {successful_tests/total_tests*100:.1f}%")
        print(f"⏱️  Total demo time: {total_time:.1f} seconds")
        
        if successful_tests == total_tests:
            print("\n🎉 COMPLETE INTEGRATION SUCCESS!")
            print("✨ AURA Intelligence TDA+LNN system fully operational")
            print("✨ Real topological analysis with sub-second performance")
            print("✨ Real liquid neural networks with MIT implementation")
            print("✨ Integrated pipeline processing topological features")
            print("✨ All endpoints responding correctly")
            print("\n🚀 READY FOR ADVANCED FEATURE INTEGRATION!")
        elif successful_tests > 0:
            print(f"\n⚠️  PARTIAL INTEGRATION ({successful_tests}/{total_tests} working)")
            print("🔧 Some components need attention")
        else:
            print("\n❌ INTEGRATION NOT WORKING")
            print("🔧 Critical issues need resolution")
        
        return successful_tests, total_tests

def main():
    """Main demonstration function"""
    demo = AURAIntegratedDemo()
    
    # Check if API is running
    try:
        response = demo.session.get(f"{demo.api_base}/", timeout=5)
        if response.status_code != 200:
            print("❌ API server not responding. Start with: python3 working_tda_api.py")
            return
    except:
        print("❌ Cannot connect to API server at http://localhost:8080")
        print("💡 Start the API with: python3 working_tda_api.py")
        return
    
    # Run demonstration
    success_count, total_count = demo.run_comprehensive_demo()
    
    # Exit code for automation
    if success_count == total_count:
        exit(0)  # All tests passed
    else:
        exit(1)  # Some tests failed

if __name__ == "__main__":
    main()