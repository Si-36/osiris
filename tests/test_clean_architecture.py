"""
🧪 Clean Architecture Tests

Comprehensive tests for the new clean AURA Intelligence architecture.
"""

import pytest
import asyncio
import torch
import numpy as np
from unittest.mock import Mock, patch

# Import AURA components
from aura.core.neural import NeuralSystem, NeuralSystemConfig, LNNConfig
from aura.core.consciousness import ConsciousnessSystem, ConsciousnessConfig
from aura.services.intelligence import IntelligenceService, IntelligenceConfig


class TestNeuralSystem:
    """Test neural system components."""
    
    @pytest.fixture
    def neural_config(self):
        """Create test neural configuration."""
        lnn_config = LNNConfig(
            input_size=64,
            hidden_size=128,
            output_size=64,
            num_layers=2,
            sparsity=0.5
        )
        
        return NeuralSystemConfig(
            lnn_config=lnn_config,
            enable_gpu=False,  # Use CPU for tests
            batch_size=16
        )
    
    @pytest.fixture
    def neural_system(self, neural_config):
        """Create test neural system."""
        return NeuralSystem(neural_config)
    
    def test_neural_system_initialization(self, neural_system):
        """Test neural system initializes correctly."""
        assert neural_system is not None
        assert neural_system.lnn is not None
        assert neural_system.device.type in ['cpu', 'cuda']
        
        info = neural_system.get_info()
        assert info["type"] == "NeuralSystem"
        assert "lnn_info" in info
        assert "statistics" in info
    
    @pytest.mark.asyncio
    async def test_neural_processing(self, neural_system):
        """Test neural processing functionality."""
        # Test with tensor input
        test_data = torch.randn(2, 64)
        result = await neural_system.process(test_data)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 2  # Batch size
        assert result.shape[1] == 64  # Output size
        
        # Test with numpy input
        np_data = np.random.randn(1, 64)
        result = await neural_system.process(np_data)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1
        assert result.shape[1] == 64
    
    @pytest.mark.asyncio
    async def test_neural_batched_processing(self, neural_system):
        """Test batched processing for large inputs."""
        # Create large input that exceeds batch size
        large_data = torch.randn(50, 64)  # Larger than batch_size=16
        
        result = await neural_system.process(large_data)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 50
        assert result.shape[1] == 64
    
    def test_neural_state_reset(self, neural_system):
        """Test neural state reset functionality."""
        # This should not raise an exception
        neural_system.reset_states()
        
        # Process some data
        test_data = torch.randn(1, 64)
        asyncio.run(neural_system.process(test_data))
        
        # Reset again
        neural_system.reset_states()


class TestConsciousnessSystem:
    """Test consciousness system components."""
    
    @pytest.fixture
    def consciousness_config(self):
        """Create test consciousness configuration."""
        return ConsciousnessConfig(
            max_workspace_size=10,
            attention_threshold=0.3,
            consciousness_threshold=0.6,
            processing_interval=0.05  # Faster for tests
        )
    
    @pytest.fixture
    def consciousness_system(self, consciousness_config):
        """Create test consciousness system."""
        return ConsciousnessSystem(consciousness_config)
    
    def test_consciousness_system_initialization(self, consciousness_system):
        """Test consciousness system initializes correctly."""
        assert consciousness_system is not None
        assert len(consciousness_system.workspace) == 0
        assert len(consciousness_system.attention_map) == 0
        
        info = consciousness_system.get_info()
        assert info["type"] == "ConsciousnessSystem"
        assert "workspace_summary" in info
        assert "statistics" in info
    
    @pytest.mark.asyncio
    async def test_consciousness_content_management(self, consciousness_system):
        """Test consciousness content management."""
        # Add content to workspace
        content_id = await consciousness_system.add_content(
            source="test",
            data={"message": "test content"},
            priority=5
        )
        
        assert content_id in consciousness_system.workspace
        assert content_id in consciousness_system.attention_map
        
        # Check workspace summary
        summary = consciousness_system.get_workspace_summary()
        assert summary["total_content"] == 1
        assert summary["max_attention"] > 0
    
    @pytest.mark.asyncio
    async def test_consciousness_decision_making(self, consciousness_system):
        """Test consciousness decision making."""
        # Add some context to workspace
        await consciousness_system.add_content(
            source="context",
            data={"context": "decision context"},
            priority=3
        )
        
        # Make a decision
        options = [
            {"name": "option1", "priority": 3},
            {"name": "option2", "priority": 7},
            {"name": "option3", "priority": 5}
        ]
        
        decision = await consciousness_system.make_decision(
            decision_id="test_decision",
            options=options,
            context={"test": "context"}
        )
        
        assert decision.decision_id == "test_decision"
        assert decision.chosen in options
        assert 0 <= decision.confidence <= 1
        assert len(decision.reasoning) > 0
        assert decision.chosen["name"] == "option2"  # Highest priority
    
    @pytest.mark.asyncio
    async def test_consciousness_processing_loop(self, consciousness_system):
        """Test consciousness processing loop."""
        # Start processing
        await consciousness_system.start_processing()
        assert consciousness_system.processing_active
        
        # Add content and let it process
        await consciousness_system.add_content(
            source="test",
            data={"test": "processing"},
            priority=8
        )
        
        # Wait for processing
        await asyncio.sleep(0.2)
        
        # Check that attention was updated
        assert len(consciousness_system.attention_map) > 0
        
        # Stop processing
        await consciousness_system.stop_processing()
        assert not consciousness_system.processing_active


class TestIntelligenceService:
    """Test intelligence service integration."""
    
    @pytest.fixture
    def intelligence_config(self):
        """Create test intelligence configuration."""
        lnn_config = LNNConfig(
            input_size=32,
            hidden_size=64,
            output_size=32,
            num_layers=1
        )
        
        neural_config = NeuralSystemConfig(
            lnn_config=lnn_config,
            enable_gpu=False,
            batch_size=8
        )
        
        consciousness_config = ConsciousnessConfig(
            max_workspace_size=5,
            processing_interval=0.05
        )
        
        return IntelligenceConfig(
            neural_config=neural_config,
            consciousness_config=consciousness_config,
            enable_neural=True,
            enable_consciousness=True
        )
    
    @pytest.fixture
    async def intelligence_service(self, intelligence_config):
        """Create and initialize test intelligence service."""
        service = IntelligenceService(intelligence_config)
        await service.initialize()
        return service
    
    @pytest.mark.asyncio
    async def test_intelligence_service_initialization(self, intelligence_service):
        """Test intelligence service initializes correctly."""
        assert intelligence_service.neural_system is not None
        assert intelligence_service.consciousness_system is not None
        
        status = await intelligence_service.get_system_status()
        assert status["service"] == "IntelligenceService"
        assert "systems" in status
        assert "neural" in status["systems"]
        assert "consciousness" in status["systems"]
    
    @pytest.mark.asyncio
    async def test_neural_inference_request(self, intelligence_service):
        """Test neural inference through intelligence service."""
        test_data = torch.randn(2, 32)
        
        result = await intelligence_service.process_intelligence_request(
            request_type="neural_inference",
            data=test_data
        )
        
        assert result["success"] is True
        assert "result" in result
        assert "processing_time" in result
        assert isinstance(result["result"], torch.Tensor)
    
    @pytest.mark.asyncio
    async def test_conscious_decision_request(self, intelligence_service):
        """Test conscious decision through intelligence service."""
        options = [
            {"name": "choice1", "priority": 2},
            {"name": "choice2", "priority": 8},
            {"name": "choice3", "priority": 5}
        ]
        
        result = await intelligence_service.process_intelligence_request(
            request_type="conscious_decision",
            data={
                "options": options,
                "decision_id": "test_decision"
            },
            context={"test": "context"}
        )
        
        assert result["success"] is True
        assert "result" in result
        assert result["result"]["chosen"]["name"] == "choice2"
        assert result["result"]["confidence"] > 0
    
    @pytest.mark.asyncio
    async def test_hybrid_processing_request(self, intelligence_service):
        """Test hybrid processing through intelligence service."""
        test_data = torch.randn(1, 32)
        
        result = await intelligence_service.process_intelligence_request(
            request_type="hybrid_processing",
            data=test_data,
            context={"hybrid": "test"}
        )
        
        assert result["success"] is True
        assert "result" in result
        assert "neural" in result["result"]
        assert "consciousness" in result["result"]
    
    @pytest.mark.asyncio
    async def test_health_check(self, intelligence_service):
        """Test intelligence service health check."""
        health = await intelligence_service.health_check()
        
        assert "overall" in health
        assert "systems" in health
        assert health["overall"] in ["healthy", "degraded"]
    
    @pytest.mark.asyncio
    async def test_error_handling(self, intelligence_service):
        """Test error handling in intelligence service."""
        # Test with invalid request type
        result = await intelligence_service.process_intelligence_request(
            request_type="invalid_type",
            data={}
        )
        
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, intelligence_service):
        """Test concurrent request handling."""
        # Create multiple concurrent requests
        tasks = []
        for i in range(5):
            task = intelligence_service.process_intelligence_request(
                request_type="neural_inference",
                data=torch.randn(1, 32)
            )
            tasks.append(task)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        for result in results:
            assert result["success"] is True
        
        # Check statistics
        assert intelligence_service.stats["total_requests"] >= 5
        assert intelligence_service.stats["successful_requests"] >= 5


class TestArchitectureIntegration:
    """Test overall architecture integration."""
    
    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """Test full system integration."""
        # Create minimal configuration
        lnn_config = LNNConfig(input_size=16, hidden_size=32, output_size=16, num_layers=1)
        neural_config = NeuralSystemConfig(lnn_config=lnn_config, enable_gpu=False)
        consciousness_config = ConsciousnessConfig(max_workspace_size=3)
        intelligence_config = IntelligenceConfig(
            neural_config=neural_config,
            consciousness_config=consciousness_config
        )
        
        # Initialize service
        service = IntelligenceService(intelligence_config)
        await service.initialize()
        
        try:
            # Test neural processing
            neural_result = await service.process_intelligence_request(
                "neural_inference",
                torch.randn(1, 16)
            )
            assert neural_result["success"]
            
            # Test decision making
            decision_result = await service.process_intelligence_request(
                "conscious_decision",
                {"options": [{"name": "test", "priority": 5}]}
            )
            assert decision_result["success"]
            
            # Test hybrid processing
            hybrid_result = await service.process_intelligence_request(
                "hybrid_processing",
                torch.randn(1, 16)
            )
            assert hybrid_result["success"]
            
            # Verify systems are working together
            status = await service.get_system_status()
            assert len(status["systems"]) >= 2
            
        finally:
            await service.shutdown()
    
    def test_import_structure(self):
        """Test that imports work correctly."""
        # Test core imports
        from aura.core.neural import NeuralSystem
        from aura.core.consciousness import ConsciousnessSystem
        
        # Test service imports
        from aura.services.intelligence import IntelligenceService
        
        # Test infrastructure imports
        from aura.infrastructure.api import APIServer
        
        # Test main package imports
        from aura import NeuralSystem as MainNeuralSystem
        
        assert NeuralSystem == MainNeuralSystem


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])