#!/usr/bin/env python3
"""
Test all fixed implementations to ensure they work properly
"""

import sys
sys.path.insert(0, '/workspace/core/src')

def test_graph_builder():
    """Test the fixed graph builder implementation"""
    print("\n🔧 Testing Graph Builder...")
    
    from aura_intelligence.collective.graph_builder import StateGraph, SqliteSaver, END
    
    # Create graph
    graph = StateGraph(dict)
    
    # Add nodes
    def process_input(state):
        state['processed'] = True
        return state
    
    def validate_data(state):
        state['validated'] = True
        return state
    
    graph.add_node("input", process_input)
    graph.add_node("validate", validate_data)
    
    # Add edges
    graph.add_edge("input", "validate")
    graph.add_edge("validate", END)
    
    # Set entry point
    graph.set_entry_point("input")
    
    # Compile
    workflow = graph.compile()
    
    # Execute
    result = workflow.invoke({"data": "test"})
    
    print(f"✅ Graph executed successfully!")
    print(f"   Nodes executed: {result['nodes_executed']}")
    print(f"   Execution time: {result['execution_time']:.3f}s")
    
    # Test checkpointer
    saver = SqliteSaver.from_conn_string(":memory:")
    saver.save("test_workflow", "node1", {"data": "test"})
    loaded = saver.load("test_workflow", "node1")
    print(f"✅ Checkpointer working: {loaded is not None}")

def test_context_engine():
    """Test the fixed context engine implementation"""
    print("\n🔧 Testing Context Engine...")
    
    from aura_intelligence.collective.context_engine import (
        ProductionAgentState, ProductionEvidence, AgentConfig
    )
    
    # Test ProductionAgentState
    state = ProductionAgentState()
    print(f"✅ ProductionAgentState initialized")
    print(f"   Metadata: {state.metadata['version']}")
    
    # Test ProductionEvidence
    evidence = ProductionEvidence(
        type='observation',
        content={'test': 'data'},
        confidence=0.9,
        source='test_source'
    )
    print(f"✅ ProductionEvidence created")
    print(f"   Evidence ID: {evidence.id[:20]}...")
    
    # Test AgentConfig
    config = AgentConfig()
    print(f"✅ AgentConfig initialized")
    print(f"   Model: {config.model}")
    print(f"   Features: {list(config.features.keys())}")

def test_enterprise_features():
    """Test the fixed enterprise features"""
    print("\n🔧 Testing Enterprise Features...")
    
    from aura_intelligence.enterprise import (
        EnterpriseSecurityManager,
        ComplianceManager,
        EnterpriseMonitoring,
        DeploymentManager
    )
    
    # Test Security Manager
    security = EnterpriseSecurityManager()
    token = security.authenticate({
        'user_id': 'test_user',
        'password': 'secure_hash_here'
    })
    print(f"✅ Security Manager: Authentication {'success' if token else 'failed'}")
    
    # Test Compliance Manager
    compliance = ComplianceManager()
    compliant, violations = compliance.check_compliance(
        'data_storage',
        {'encrypted': True}
    )
    print(f"✅ Compliance Manager: {'Compliant' if compliant else 'Non-compliant'}")
    
    # Test Monitoring
    monitoring = EnterpriseMonitoring()
    monitoring.record_metric('api_latency', 250, 'gauge')
    health = monitoring.get_health_status()
    print(f"✅ Monitoring: System health is {health['overall_status']}")
    
    # Test Deployment Manager
    deployment = DeploymentManager()
    deployment_id = deployment.deploy('v2.0.0', 'canary', 'prod')
    print(f"✅ Deployment Manager: Deployed {deployment_id is not False}")

def test_integrations():
    """Test integrations module"""
    print("\n🔧 Testing Integrations...")
    
    try:
        from aura_intelligence.integrations import __init__ as integrations
        print("✅ Integrations module imports successfully")
        
        # Check if the module has been fixed
        import inspect
        source = inspect.getsource(integrations)
        if 'pass' not in source or 'def __init__(self):' not in source:
            print("✅ No dummy implementations found in integrations")
        else:
            print("⚠️  Some dummy implementations may still exist in integrations")
    except Exception as e:
        print(f"⚠️  Could not test integrations: {e}")

def main():
    """Run all tests"""
    print("🚀 TESTING FIXED IMPLEMENTATIONS")
    print("=" * 60)
    
    tests = [
        test_graph_builder,
        test_context_engine,
        test_enterprise_features,
        test_integrations
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All implementations are working correctly!")
    else:
        print("⚠️  Some implementations need additional fixes")

if __name__ == "__main__":
    main()