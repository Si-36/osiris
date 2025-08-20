#!/usr/bin/env python3
"""
AURA Intelligence Infrastructure Connectivity Test
Tests all infrastructure components with real connections.
"""

import asyncio
import os
import sys
import redis
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add core to Python path
sys.path.insert(0, str(Path(__file__).parent / "core" / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class InfrastructureTest:
    def __init__(self):
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0

    async def test_redis_connection(self):
        """Test Redis connectivity and operations."""
        print("🔴 Testing Redis Connection...")
        try:
            # Connect to Redis (using new port)
            redis_client = redis.Redis(
                host='localhost',
                port=6380,  # Changed to new port
                password=os.getenv('REDIS_PASSWORD', 'aura_redis_2025_secure'),
                decode_responses=True
            )
            
            # Test basic operations
            redis_client.ping()
            redis_client.set('test_key', 'test_value')
            value = redis_client.get('test_key')
            redis_client.delete('test_key')
            
            if value == 'test_value':
                self.record_result('redis_connection', True, "Redis CRUD operations successful")
                print("  ✅ Redis: Connected and operational")
            else:
                self.record_result('redis_connection', False, f"Value mismatch: {value}")
                print("  ❌ Redis: Value mismatch error")
                
        except Exception as e:
            self.record_result('redis_connection', False, str(e))
            print(f"  ❌ Redis: {e}")

    async def test_neo4j_connection(self):
        """Test Neo4j connectivity and operations."""
        print("🟢 Testing Neo4j Connection...")
        try:
            from neo4j import GraphDatabase
            
            # Connect to Neo4j (using new port)
            driver = GraphDatabase.driver(
                "bolt://localhost:7688",  # Changed to new port
                auth=("neo4j", os.getenv('NEO4J_PASSWORD', 'aura_neo4j_2025_secure'))
            )
            
            # Test basic operations
            with driver.session() as session:
                # Create a test node
                result = session.run(
                    "CREATE (test:TestNode {name: 'infrastructure_test', timestamp: $timestamp}) "
                    "RETURN test.name as name",
                    timestamp=int(time.time())
                )
                
                record = result.single()
                if record and record['name'] == 'infrastructure_test':
                    # Clean up test node
                    session.run("MATCH (test:TestNode {name: 'infrastructure_test'}) DELETE test")
                    
                    self.record_result('neo4j_connection', True, "Neo4j CRUD operations successful")
                    print("  ✅ Neo4j: Connected and operational")
                else:
                    self.record_result('neo4j_connection', False, "Could not create/read test node")
                    print("  ❌ Neo4j: CRUD operation failed")
            
            driver.close()
            
        except Exception as e:
            self.record_result('neo4j_connection', False, str(e))
            print(f"  ❌ Neo4j: {e}")

    async def test_kafka_connection(self):
        """Test Kafka connectivity and operations."""
        print("🟡 Testing Kafka Connection...")
        try:
            from kafka import KafkaProducer, KafkaConsumer
            from kafka.admin import KafkaAdminClient, NewTopic
            
            # Test producer (using new port)
            producer = KafkaProducer(
                bootstrap_servers=['localhost:9093'],  # Changed to new port
                value_serializer=lambda v: str(v).encode('utf-8')
            )
            
            # Send test message
            future = producer.send('infrastructure_test', 'test_message')
            producer.flush()
            
            # Test admin client (using new port)
            admin_client = KafkaAdminClient(bootstrap_servers=['localhost:9093'])
            
            # List topics to verify connection
            metadata = admin_client.list_consumer_groups()
            
            producer.close()
            
            self.record_result('kafka_connection', True, "Kafka producer and admin operations successful")
            print("  ✅ Kafka: Connected and operational")
            
        except Exception as e:
            self.record_result('kafka_connection', False, str(e))
            print(f"  ❌ Kafka: {e}")

    async def test_postgres_connection(self):
        """Test PostgreSQL connectivity and operations."""
        print("🔵 Testing PostgreSQL Connection...")
        try:
            import psycopg2
            
            # Connect to PostgreSQL (using new port)
            conn = psycopg2.connect(
                host="localhost",
                port=5433,  # Changed to new port
                database="aura_intelligence",
                user="aura_user",
                password=os.getenv('POSTGRES_PASSWORD', 'aura_postgres_2025_secure')
            )
            
            cursor = conn.cursor()
            
            # Test basic operations
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            
            # Create test table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS infrastructure_test (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert test data
            cursor.execute(
                "INSERT INTO infrastructure_test (name) VALUES (%s) RETURNING id",
                ('test_record',)
            )
            test_id = cursor.fetchone()[0]
            
            # Query test data
            cursor.execute("SELECT name FROM infrastructure_test WHERE id = %s", (test_id,))
            result = cursor.fetchone()
            
            # Clean up
            cursor.execute("DROP TABLE infrastructure_test")
            conn.commit()
            
            if result and result[0] == 'test_record':
                self.record_result('postgres_connection', True, f"PostgreSQL operations successful: {version[0][:50]}")
                print("  ✅ PostgreSQL: Connected and operational")
            else:
                self.record_result('postgres_connection', False, "CRUD operation failed")
                print("  ❌ PostgreSQL: CRUD operation failed")
                
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.record_result('postgres_connection', False, str(e))
            print(f"  ❌ PostgreSQL: {e}")

    async def test_gemini_api(self):
        """Test Gemini API connectivity."""
        print("🤖 Testing Gemini API Connection...")
        try:
            import google.generativeai as genai
            
            # Configure Gemini
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
            
            genai.configure(api_key=api_key)
            
            # Test with a simple request
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content("Say 'AURA Intelligence test successful'")
            
            if response.text and 'AURA Intelligence test successful' in response.text:
                self.record_result('gemini_api', True, "Gemini API response successful")
                print("  ✅ Gemini API: Connected and responding")
            else:
                self.record_result('gemini_api', False, f"Unexpected response: {response.text[:100]}")
                print("  ❌ Gemini API: Unexpected response")
                
        except Exception as e:
            self.record_result('gemini_api', False, str(e))
            print(f"  ❌ Gemini API: {e}")

    async def test_langsmith_tracing(self):
        """Test LangSmith tracing connectivity."""
        print("📊 Testing LangSmith Tracing...")
        try:
            import langsmith
            from langsmith import Client
            
            # Initialize LangSmith client
            api_key = os.getenv('LANGSMITH_API_KEY')
            if not api_key:
                raise ValueError("LANGSMITH_API_KEY not found in environment")
            
            client = Client(api_key=api_key)
            
            # Test connection by creating a simple run
            with langsmith.trace(name="infrastructure_test") as run:
                run.info = {"test": "AURA Infrastructure Connectivity"}
                run.outputs = {"status": "testing"}
            
            self.record_result('langsmith_tracing', True, "LangSmith tracing operational")
            print("  ✅ LangSmith: Connected and tracing")
            
        except Exception as e:
            self.record_result('langsmith_tracing', False, str(e))
            print(f"  ❌ LangSmith: {e}")

    async def test_aura_component_imports(self):
        """Test AURA component imports with real infrastructure."""
        print("🧠 Testing AURA Component Imports...")
        try:
            # Test key adapter imports
            from aura_intelligence.adapters.redis_adapter import RedisAdapter
            from aura_intelligence.adapters.neo4j_adapter import Neo4jAdapter
            
            # Test Redis adapter with real connection (using new port)
            redis_adapter = RedisAdapter({
                'host': 'localhost',
                'port': 6380,  # Changed to new port
                'password': os.getenv('REDIS_PASSWORD', 'aura_redis_2025_secure')
            })
            
            await redis_adapter.store('test_key', {'test': 'data'})
            result = await redis_adapter.retrieve('test_key')
            await redis_adapter.delete('test_key')
            
            if result and result.get('test') == 'data':
                self.record_result('aura_components', True, "AURA adapters working with real infrastructure")
                print("  ✅ AURA Components: Adapters functional with real infrastructure")
            else:
                self.record_result('aura_components', False, f"Adapter test failed: {result}")
                print("  ❌ AURA Components: Adapter test failed")
                
        except Exception as e:
            self.record_result('aura_components', False, str(e))
            print(f"  ❌ AURA Components: {e}")

    def record_result(self, test_name: str, success: bool, details: str = ""):
        """Record test result."""
        self.results[test_name] = {
            'success': success,
            'details': details,
            'timestamp': time.time()
        }
        self.total_tests += 1
        if success:
            self.passed_tests += 1

    async def run_all_tests(self):
        """Run all infrastructure tests."""
        print("🏗️  AURA Intelligence Infrastructure Connectivity Test")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all tests
        await self.test_redis_connection()
        await self.test_neo4j_connection()
        await self.test_kafka_connection()
        await self.test_postgres_connection()
        await self.test_gemini_api()
        await self.test_langsmith_tracing()
        await self.test_aura_component_imports()
        
        # Generate report
        await self.generate_report(time.time() - start_time)

    async def generate_report(self, execution_time: float):
        """Generate comprehensive test report."""
        print("\n" + "=" * 60)
        print("📊 INFRASTRUCTURE CONNECTIVITY REPORT")
        print("=" * 60)
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"📈 OVERALL RESULTS:")
        print(f"   Tests Run: {self.total_tests}")
        print(f"   Passed: {self.passed_tests}")
        print(f"   Failed: {self.total_tests - self.passed_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Execution Time: {execution_time:.2f}s")
        
        print(f"\n✅ PASSED TESTS:")
        for test_name, result in self.results.items():
            if result['success']:
                print(f"   ✅ {test_name}: {result['details']}")
        
        print(f"\n❌ FAILED TESTS:")
        failed_tests = [(name, result) for name, result in self.results.items() if not result['success']]
        if failed_tests:
            for test_name, result in failed_tests:
                print(f"   ❌ {test_name}: {result['details']}")
        else:
            print("   🎉 All tests passed!")
        
        print(f"\n🎯 NEXT STEPS:")
        if success_rate >= 85:
            print("   🚀 Infrastructure is ready for AURA deployment!")
            print("   → Run: python test_all_real_components.py")
            print("   → Start: docker-compose up -d aura-api")
        elif success_rate >= 60:
            print("   ⚠️  Some issues detected. Fix failed tests before proceeding.")
            print("   → Check service configurations and credentials")
        else:
            print("   🚨 Major infrastructure issues. Address all failures.")
            print("   → Verify Docker containers are running")
            print("   → Check network connectivity and credentials")
        
        print("\n" + "=" * 60)

async def main():
    """Main test execution."""
    tester = InfrastructureTest()
    await tester.run_all_tests()

if __name__ == "__main__":
    # Install missing packages if needed
    try:
        import redis
        import psycopg2
        import google.generativeai
        import neo4j
        import kafka
        import langsmith
        import dotenv
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install redis psycopg2-binary google-generativeai neo4j kafka-python langsmith python-dotenv")
        sys.exit(1)
    
    asyncio.run(main())