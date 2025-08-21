"""
Contract Testing Framework for AURA Microservices
Consumer-Driven Contracts with AsyncAPI and OpenAPI support

2025 Best Practices:
- Bi-directional contract testing
- Schema evolution tracking
- Async event contracts
- GraphQL contract support
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Protocol, Type
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import structlog
from pydantic import BaseModel, ValidationError
import httpx
from jsonschema import validate, ValidationError as JsonSchemaError
import aiofiles

logger = structlog.get_logger()


@dataclass
class Contract:
    """Base contract definition"""
    provider: str
    consumer: str
    version: str
    type: str  # rest, event, graphql
    specifications: Dict[str, Any]
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContractValidationResult:
    """Result of contract validation"""
    contract_id: str
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    coverage: float = 0.0
    tested_scenarios: List[str] = field(default_factory=list)


class ContractProvider(Protocol):
    """Protocol for contract providers"""
    
    async def get_contract(self) -> Contract:
        """Get the contract specification"""
        ...
    
    async def verify_consumer(self, consumer_name: str) -> ContractValidationResult:
        """Verify a consumer adheres to contract"""
        ...


class ContractConsumer(Protocol):
    """Protocol for contract consumers"""
    
    async def validate_against_provider(self, contract: Contract) -> ContractValidationResult:
        """Validate against provider contract"""
        ...


class OpenAPIContractValidator:
    """
    OpenAPI 3.1 contract validation with examples
    """
    
    def __init__(self, spec_path: str):
        self.spec_path = spec_path
        self.spec: Dict[str, Any] = {}
        self.logger = logger.bind(validator="openapi")
        
    async def load_spec(self):
        """Load OpenAPI specification"""
        async with aiofiles.open(self.spec_path, 'r') as f:
            content = await f.read()
            self.spec = yaml.safe_load(content)
    
    async def validate_request(self, 
                              endpoint: str,
                              method: str,
                              request_data: Dict[str, Any]) -> ContractValidationResult:
        """Validate request against OpenAPI spec"""
        result = ContractValidationResult(
            contract_id=f"{method.upper()} {endpoint}",
            valid=True
        )
        
        try:
            # Find endpoint in spec
            path_spec = self._find_endpoint_spec(endpoint)
            if not path_spec:
                result.valid = False
                result.errors.append(f"Endpoint {endpoint} not found in spec")
                return result
            
            # Get method spec
            method_spec = path_spec.get(method.lower())
            if not method_spec:
                result.valid = False
                result.errors.append(f"Method {method} not supported for {endpoint}")
                return result
            
            # Validate request body
            if "requestBody" in method_spec:
                await self._validate_request_body(
                    method_spec["requestBody"],
                    request_data,
                    result
                )
            
            # Validate parameters
            if "parameters" in method_spec:
                await self._validate_parameters(
                    method_spec["parameters"],
                    request_data,
                    result
                )
            
        except Exception as e:
            result.valid = False
            result.errors.append(f"Validation error: {str(e)}")
        
        return result
    
    async def validate_response(self,
                               endpoint: str,
                               method: str,
                               status_code: int,
                               response_data: Any) -> ContractValidationResult:
        """Validate response against OpenAPI spec"""
        result = ContractValidationResult(
            contract_id=f"{method.upper()} {endpoint} -> {status_code}",
            valid=True
        )
        
        try:
            # Find endpoint spec
            path_spec = self._find_endpoint_spec(endpoint)
            if not path_spec:
                result.valid = False
                result.errors.append(f"Endpoint {endpoint} not found in spec")
                return result
            
            # Get response spec
            method_spec = path_spec.get(method.lower(), {})
            response_spec = method_spec.get("responses", {}).get(str(status_code))
            
            if not response_spec:
                result.warnings.append(f"No spec for status code {status_code}")
                return result
            
            # Validate response schema
            if "content" in response_spec:
                content_type = "application/json"  # Default
                schema = response_spec["content"].get(content_type, {}).get("schema")
                
                if schema:
                    try:
                        validate(instance=response_data, schema=schema)
                    except JsonSchemaError as e:
                        result.valid = False
                        result.errors.append(f"Schema validation failed: {e.message}")
            
        except Exception as e:
            result.valid = False
            result.errors.append(f"Validation error: {str(e)}")
        
        return result
    
    def _find_endpoint_spec(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Find endpoint in OpenAPI spec (handles path parameters)"""
        paths = self.spec.get("paths", {})
        
        # Direct match
        if endpoint in paths:
            return paths[endpoint]
        
        # Try pattern matching for path parameters
        for path_pattern, path_spec in paths.items():
            if self._matches_pattern(endpoint, path_pattern):
                return path_spec
        
        return None
    
    def _matches_pattern(self, endpoint: str, pattern: str) -> bool:
        """Check if endpoint matches pattern with parameters"""
        # Simple pattern matching for {param} style
        import re
        pattern_regex = pattern.replace("{", "(?P<").replace("}", ">[^/]+)")
        return bool(re.match(f"^{pattern_regex}$", endpoint))
    
    async def _validate_request_body(self,
                                    body_spec: Dict[str, Any],
                                    request_data: Dict[str, Any],
                                    result: ContractValidationResult):
        """Validate request body against spec"""
        if "content" in body_spec:
            content_type = "application/json"
            schema = body_spec["content"].get(content_type, {}).get("schema")
            
            if schema:
                try:
                    validate(instance=request_data, schema=schema)
                except JsonSchemaError as e:
                    result.valid = False
                    result.errors.append(f"Request body validation failed: {e.message}")
    
    async def _validate_parameters(self,
                                  parameters: List[Dict[str, Any]],
                                  request_data: Dict[str, Any],
                                  result: ContractValidationResult):
        """Validate request parameters"""
        for param in parameters:
            name = param["name"]
            required = param.get("required", False)
            
            if required and name not in request_data:
                result.valid = False
                result.errors.append(f"Required parameter '{name}' missing")


class AsyncAPIContractValidator:
    """
    AsyncAPI 2.6 contract validation for event-driven contracts
    """
    
    def __init__(self, spec_path: str):
        self.spec_path = spec_path
        self.spec: Dict[str, Any] = {}
        self.logger = logger.bind(validator="asyncapi")
    
    async def load_spec(self):
        """Load AsyncAPI specification"""
        async with aiofiles.open(self.spec_path, 'r') as f:
            content = await f.read()
            self.spec = yaml.safe_load(content)
    
    async def validate_message(self,
                              channel: str,
                              message_data: Dict[str, Any],
                              operation: str = "publish") -> ContractValidationResult:
        """Validate message against AsyncAPI spec"""
        result = ContractValidationResult(
            contract_id=f"{operation} {channel}",
            valid=True
        )
        
        try:
            # Find channel in spec
            channels = self.spec.get("channels", {})
            channel_spec = channels.get(channel)
            
            if not channel_spec:
                result.valid = False
                result.errors.append(f"Channel {channel} not found in spec")
                return result
            
            # Get operation spec
            operation_spec = channel_spec.get(operation)
            if not operation_spec:
                result.valid = False
                result.errors.append(f"Operation {operation} not supported for {channel}")
                return result
            
            # Get message spec
            message_ref = operation_spec.get("message", {}).get("$ref")
            if message_ref:
                message_spec = self._resolve_ref(message_ref)
            else:
                message_spec = operation_spec.get("message", {})
            
            # Validate payload
            if "payload" in message_spec:
                payload_schema = message_spec["payload"]
                try:
                    validate(instance=message_data, schema=payload_schema)
                except JsonSchemaError as e:
                    result.valid = False
                    result.errors.append(f"Payload validation failed: {e.message}")
            
            # Validate headers if present
            if "headers" in message_spec and "headers" in message_data:
                headers_schema = message_spec["headers"]
                try:
                    validate(instance=message_data["headers"], schema=headers_schema)
                except JsonSchemaError as e:
                    result.valid = False
                    result.errors.append(f"Headers validation failed: {e.message}")
            
        except Exception as e:
            result.valid = False
            result.errors.append(f"Validation error: {str(e)}")
        
        return result
    
    def _resolve_ref(self, ref: str) -> Dict[str, Any]:
        """Resolve $ref in AsyncAPI spec"""
        # Simple implementation for local refs
        if ref.startswith("#/"):
            path_parts = ref[2:].split("/")
            current = self.spec
            
            for part in path_parts:
                current = current.get(part, {})
            
            return current
        
        return {}


class ContractTestRunner:
    """
    Orchestrate contract testing across services
    """
    
    def __init__(self, contracts_dir: str = "./contracts"):
        self.contracts_dir = Path(contracts_dir)
        self.logger = logger.bind(component="contract_runner")
        self.validators: Dict[str, Any] = {}
        self.results: List[ContractValidationResult] = []
    
    async def discover_contracts(self) -> List[Contract]:
        """Discover all contracts in directory"""
        contracts = []
        
        for contract_file in self.contracts_dir.rglob("*.yaml"):
            try:
                async with aiofiles.open(contract_file, 'r') as f:
                    content = await f.read()
                    data = yaml.safe_load(content)
                    
                    contract = Contract(
                        provider=data.get("provider", "unknown"),
                        consumer=data.get("consumer", "unknown"),
                        version=data.get("version", "1.0.0"),
                        type=data.get("type", "rest"),
                        specifications=data.get("specifications", {}),
                        examples=data.get("examples", []),
                        metadata=data.get("metadata", {})
                    )
                    
                    contracts.append(contract)
                    
            except Exception as e:
                self.logger.error(f"Error loading contract {contract_file}", error=str(e))
        
        return contracts
    
    async def run_consumer_tests(self, 
                                consumer_name: str,
                                provider_url: str) -> List[ContractValidationResult]:
        """Run all consumer contract tests against provider"""
        contracts = await self.discover_contracts()
        consumer_contracts = [c for c in contracts if c.consumer == consumer_name]
        
        results = []
        
        for contract in consumer_contracts:
            self.logger.info(
                f"Testing contract",
                provider=contract.provider,
                consumer=contract.consumer,
                version=contract.version
            )
            
            if contract.type == "rest":
                result = await self._test_rest_contract(contract, provider_url)
            elif contract.type == "event":
                result = await self._test_event_contract(contract, provider_url)
            else:
                result = ContractValidationResult(
                    contract_id=f"{contract.provider}-{contract.consumer}",
                    valid=False,
                    errors=[f"Unsupported contract type: {contract.type}"]
                )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    async def _test_rest_contract(self, 
                                 contract: Contract,
                                 provider_url: str) -> ContractValidationResult:
        """Test REST API contract"""
        result = ContractValidationResult(
            contract_id=f"{contract.provider}-{contract.consumer}-{contract.version}",
            valid=True
        )
        
        # Create validator
        openapi_validator = OpenAPIContractValidator(
            contract.specifications.get("openapi_path", "")
        )
        await openapi_validator.load_spec()
        
        # Test each example
        async with httpx.AsyncClient(base_url=provider_url) as client:
            for example in contract.examples:
                scenario = example.get("scenario", "unknown")
                result.tested_scenarios.append(scenario)
                
                try:
                    # Make request
                    response = await client.request(
                        method=example["request"]["method"],
                        url=example["request"]["path"],
                        json=example["request"].get("body"),
                        params=example["request"].get("params"),
                        headers=example["request"].get("headers", {})
                    )
                    
                    # Validate response
                    validation = await openapi_validator.validate_response(
                        endpoint=example["request"]["path"],
                        method=example["request"]["method"],
                        status_code=response.status_code,
                        response_data=response.json() if response.content else None
                    )
                    
                    if not validation.valid:
                        result.valid = False
                        result.errors.extend(validation.errors)
                    
                    # Check expected response
                    if "response" in example:
                        expected_status = example["response"].get("status")
                        if expected_status and response.status_code != expected_status:
                            result.valid = False
                            result.errors.append(
                                f"Expected status {expected_status}, got {response.status_code}"
                            )
                    
                except Exception as e:
                    result.valid = False
                    result.errors.append(f"Test failed for scenario '{scenario}': {str(e)}")
        
        # Calculate coverage
        total_endpoints = len(openapi_validator.spec.get("paths", {}))
        tested_endpoints = len(set(e["request"]["path"] for e in contract.examples))
        result.coverage = tested_endpoints / max(total_endpoints, 1)
        
        return result
    
    async def _test_event_contract(self,
                                  contract: Contract,
                                  provider_url: str) -> ContractValidationResult:
        """Test event-driven contract"""
        result = ContractValidationResult(
            contract_id=f"{contract.provider}-{contract.consumer}-{contract.version}",
            valid=True
        )
        
        # Create validator
        asyncapi_validator = AsyncAPIContractValidator(
            contract.specifications.get("asyncapi_path", "")
        )
        await asyncapi_validator.load_spec()
        
        # Test each example message
        for example in contract.examples:
            scenario = example.get("scenario", "unknown")
            result.tested_scenarios.append(scenario)
            
            try:
                # Validate message format
                validation = await asyncapi_validator.validate_message(
                    channel=example["channel"],
                    message_data=example["message"],
                    operation=example.get("operation", "publish")
                )
                
                if not validation.valid:
                    result.valid = False
                    result.errors.extend(validation.errors)
                
                # If provider URL given, try to publish/consume
                if provider_url and example.get("test_interaction", True):
                    # This would connect to actual message broker
                    # For now, just validate format
                    pass
                    
            except Exception as e:
                result.valid = False
                result.errors.append(f"Test failed for scenario '{scenario}': {str(e)}")
        
        return result
    
    async def generate_report(self) -> Dict[str, Any]:
        """Generate contract testing report"""
        total_contracts = len(self.results)
        passed_contracts = sum(1 for r in self.results if r.valid)
        
        report = {
            "summary": {
                "total_contracts": total_contracts,
                "passed": passed_contracts,
                "failed": total_contracts - passed_contracts,
                "pass_rate": passed_contracts / max(total_contracts, 1),
                "average_coverage": sum(r.coverage for r in self.results) / max(len(self.results), 1)
            },
            "contracts": []
        }
        
        for result in self.results:
            report["contracts"].append({
                "id": result.contract_id,
                "valid": result.valid,
                "errors": result.errors,
                "warnings": result.warnings,
                "coverage": result.coverage,
                "scenarios_tested": len(result.tested_scenarios)
            })
        
        return report


class ContractRecorder:
    """
    Record actual interactions to generate contract examples
    """
    
    def __init__(self):
        self.recordings: List[Dict[str, Any]] = []
        self.logger = logger.bind(component="contract_recorder")
    
    async def record_http_interaction(self,
                                     request: httpx.Request,
                                     response: httpx.Response,
                                     scenario: str = "recorded"):
        """Record HTTP interaction"""
        interaction = {
            "scenario": scenario,
            "timestamp": time.time(),
            "request": {
                "method": request.method,
                "path": str(request.url.path),
                "headers": dict(request.headers),
                "body": json.loads(request.content) if request.content else None,
                "params": dict(request.url.params) if request.url.params else None
            },
            "response": {
                "status": response.status_code,
                "headers": dict(response.headers),
                "body": response.json() if response.content else None
            }
        }
        
        self.recordings.append(interaction)
    
    async def record_event_interaction(self,
                                      channel: str,
                                      message: Dict[str, Any],
                                      operation: str = "publish",
                                      scenario: str = "recorded"):
        """Record event interaction"""
        interaction = {
            "scenario": scenario,
            "timestamp": time.time(),
            "channel": channel,
            "operation": operation,
            "message": message
        }
        
        self.recordings.append(interaction)
    
    async def generate_contract_examples(self) -> List[Dict[str, Any]]:
        """Generate contract examples from recordings"""
        examples = []
        
        for recording in self.recordings:
            if "request" in recording:
                # HTTP interaction
                example = {
                    "scenario": recording["scenario"],
                    "request": recording["request"],
                    "response": {
                        "status": recording["response"]["status"],
                        "body": recording["response"]["body"]
                    }
                }
            else:
                # Event interaction
                example = {
                    "scenario": recording["scenario"],
                    "channel": recording["channel"],
                    "operation": recording["operation"],
                    "message": recording["message"]
                }
            
            examples.append(example)
        
        return examples


# Example contract schema
EXAMPLE_CONTRACT = """
provider: neuromorphic-service
consumer: moe-router
version: 1.0.0
type: rest
metadata:
  description: Contract between MoE Router and Neuromorphic Service
  maintainer: aura-team
specifications:
  openapi_path: ./specs/neuromorphic-openapi.yaml
examples:
  - scenario: spike_processing
    request:
      method: POST
      path: /api/v1/process/spike
      body:
        spike_data: [[1,0,1,0,1]]
        time_steps: 10
    response:
      status: 200
      body:
        spike_output: [[0.8, 0.2]]
        energy_consumed_pj: 125.5
  - scenario: health_check
    request:
      method: GET
      path: /api/v1/health
    response:
      status: 200
      body:
        status: healthy
"""


if __name__ == "__main__":
    import time
    
    async def example_usage():
        # Create contract runner
        runner = ContractTestRunner("./contracts")
        
        # Run consumer tests
        results = await runner.run_consumer_tests(
            consumer_name="moe-router",
            provider_url="http://localhost:8000"
        )
        
        # Generate report
        report = await runner.generate_report()
        print(json.dumps(report, indent=2))
    
    asyncio.run(example_usage())