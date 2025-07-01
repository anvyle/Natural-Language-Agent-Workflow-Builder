# This is a test file.

import unittest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Mock classes for testing
class MockLLM:
    def __init__(self, responses: List[str] | None = None):
        self.responses = responses or []
        self.call_count = 0
    
    def generate(self, prompt: str) -> str:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return '{"name": "test_agent", "description": "Test agent", "tasks": []}'

class MockToolRegistry:
    def __init__(self):
        self.tools = {
            "WebSearchBlock": {
                "name": "WebSearchBlock",
                "description": "Search the web for information",
                "inputs": {"query": "string"},
                "outputs": {"results": "array"}
            },
            "EmailWriterBlock": {
                "name": "EmailWriterBlock", 
                "description": "Write personalized emails",
                "inputs": {"recipient": "string", "context": "string"},
                "outputs": {"email": "string"}
            },
            "NotionExportBlock": {
                "name": "NotionExportBlock",
                "description": "Export data to Notion",
                "inputs": {"content": "string", "database_id": "string"},
                "outputs": {"page_id": "string"}
            }
        }
    
    def get_tool_summaries(self) -> str:
        return json.dumps(self.tools, indent=2)
    
    def validate_tool(self, tool_name: str) -> bool:
        return tool_name in self.tools

class AgentGenerator:
    """Core class for generating agents from natural language descriptions"""
    
    def __init__(self, llm: MockLLM, tool_registry: MockToolRegistry):
        self.llm = llm
        self.tool_registry = tool_registry
    
    def generate_agent(self, description: str) -> Dict[str, Any]:
        """Generate agent configuration from natural language description"""
        system_prompt = self._build_system_prompt()
        full_prompt = f"{system_prompt}\n\nUser Description: {description}"
        
        response = self.llm.generate(full_prompt)
        
        try:
            agent_config = json.loads(response)
            return self._validate_and_clean_config(agent_config)
        except json.JSONDecodeError:
            return self._create_fallback_config(description)
    
    def _build_system_prompt(self) -> str:
        tool_summaries = self.tool_registry.get_tool_summaries()
        return f"""You are an expert AutoGPT agent architect. Generate a complete agent.json with subtasks, memory, prompt, and tools.

Available tools:
{tool_summaries}

Output format:
{{
    "name": "string",
    "description": "string", 
    "systemPrompt": "string",
    "tasks": [{{"id": "string", "name": "string", "blockName": "string", "inputs": {{}}}}],
    "memory": {{"enabled": true, "keys": []}}
}}"""
    
    def _validate_and_clean_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean the generated agent configuration"""
        required_fields = ["name", "description", "systemPrompt", "tasks"]
        
        for field in required_fields:
            if field not in config:
                config[field] = self._get_default_value(field)
        
        # Validate tasks
        if not isinstance(config.get("tasks"), list):
            config["tasks"] = []
        
        # Validate memory
        if "memory" not in config:
            config["memory"] = {"enabled": True, "keys": []}
        
        return config
    
    def _get_default_value(self, field: str) -> Any:
        """Get default values for missing fields"""
        defaults = {
            "name": "Generated Agent",
            "description": "Auto-generated agent",
            "systemPrompt": "You are a helpful AI agent.",
            "tasks": []
        }
        return defaults.get(field, "")
    
    def _create_fallback_config(self, description: str) -> Dict[str, Any]:
        """Create a fallback configuration when JSON parsing fails"""
        return {
            "name": "Fallback Agent",
            "description": f"Fallback agent for: {description}",
            "systemPrompt": "You are a helpful AI agent.",
            "tasks": [],
            "memory": {"enabled": True, "keys": []}
        }

class AgentValidator:
    """Validate generated agent configurations"""
    
    def __init__(self, tool_registry: MockToolRegistry):
        self.tool_registry = tool_registry
    
    def validate_agent(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an agent configuration and return validation results"""
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ["name", "description", "systemPrompt", "tasks"]
        for field in required_fields:
            if field not in agent_config:
                errors.append(f"Missing required field: {field}")
            elif not agent_config[field]:
                warnings.append(f"Empty field: {field}")
        
        # Validate tasks
        if "tasks" in agent_config:
            task_errors = self._validate_tasks(agent_config["tasks"])
            errors.extend(task_errors)
        
        # Validate memory configuration
        if "memory" in agent_config:
            memory_errors = self._validate_memory(agent_config["memory"])
            errors.extend(memory_errors)
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _validate_tasks(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Validate task configurations"""
        errors = []
        
        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                errors.append(f"Task {i}: Must be a dictionary")
                continue
            
            required_task_fields = ["id", "name", "blockName"]
            for field in required_task_fields:
                if field not in task:
                    errors.append(f"Task {i}: Missing required field '{field}'")
            
            # Validate block name exists in registry
            if "blockName" in task and task["blockName"]:
                if not self.tool_registry.validate_tool(task["blockName"]):
                    errors.append(f"Task {i}: Unknown tool '{task['blockName']}'")
        
        return errors
    
    def _validate_memory(self, memory: Dict[str, Any]) -> List[str]:
        """Validate memory configuration"""
        errors = []
        
        if not isinstance(memory, dict):
            errors.append("Memory must be a dictionary")
            return errors
        
        if "enabled" not in memory:
            errors.append("Memory configuration missing 'enabled' field")
        
        if "keys" not in memory:
            errors.append("Memory configuration missing 'keys' field")
        elif not isinstance(memory["keys"], list):
            errors.append("Memory keys must be a list")
        
        return errors

# Test Cases
class TestAgentGenerator(unittest.TestCase):
    """Test cases for the AgentGenerator class"""
    
    def setUp(self):
        self.tool_registry = MockToolRegistry()
        self.llm = MockLLM()
        self.generator = AgentGenerator(self.llm, self.tool_registry)
    
    def test_generate_agent_basic(self):
        """Test basic agent generation"""
        description = "Create an agent that searches the web and writes emails"
        
        # Mock LLM response
        mock_response = '''{
            "name": "Web Search Email Agent",
            "description": "Agent that searches web and writes emails",
            "systemPrompt": "You are a helpful assistant that searches the web and writes emails.",
            "tasks": [
                {
                    "id": "search",
                    "name": "Search Web",
                    "blockName": "WebSearchBlock",
                    "inputs": {"query": "{{user_query}}"}
                },
                {
                    "id": "write_email",
                    "name": "Write Email", 
                    "blockName": "EmailWriterBlock",
                    "inputs": {"recipient": "{{recipient}}", "context": "{{search_results}}"}
                }
            ],
            "memory": {"enabled": true, "keys": ["search_results"]}
        }'''
        
        self.llm.responses = [mock_response]
        
        result = self.generator.generate_agent(description)
        
        self.assertEqual(result["name"], "Web Search Email Agent")
        self.assertEqual(len(result["tasks"]), 2)
        self.assertTrue(result["memory"]["enabled"])
    
    def test_generate_agent_invalid_json(self):
        """Test agent generation with invalid JSON response"""
        description = "Create a test agent"
        
        # Mock invalid JSON response
        self.llm.responses = ["Invalid JSON response"]
        
        result = self.generator.generate_agent(description)
        
        self.assertEqual(result["name"], "Fallback Agent")
        self.assertIn("Fallback agent for:", result["description"])
    
    def test_generate_agent_missing_fields(self):
        """Test agent generation with missing fields"""
        description = "Create a minimal agent"
        
        # Mock response with missing fields
        mock_response = '{"name": "Minimal Agent"}'
        self.llm.responses = [mock_response]
        
        result = self.generator.generate_agent(description)
        
        self.assertEqual(result["name"], "Minimal Agent")
        self.assertIn("description", result)
        self.assertIn("systemPrompt", result)
        self.assertIn("tasks", result)
        self.assertIn("memory", result)

class TestAgentValidator(unittest.TestCase):
    """Test cases for the AgentValidator class"""
    
    def setUp(self):
        self.tool_registry = MockToolRegistry()
        self.validator = AgentValidator(self.tool_registry)
    
    def test_validate_valid_agent(self):
        """Test validation of a valid agent configuration"""
        valid_agent = {
            "name": "Test Agent",
            "description": "A test agent",
            "systemPrompt": "You are a helpful agent.",
            "tasks": [
                {
                    "id": "task1",
                    "name": "Search Web",
                    "blockName": "WebSearchBlock",
                    "inputs": {"query": "test"}
                }
            ],
            "memory": {"enabled": True, "keys": []}
        }
        
        result = self.validator.validate_agent(valid_agent)
        
        self.assertTrue(result["is_valid"])
        self.assertEqual(len(result["errors"]), 0)
    
    def test_validate_invalid_agent(self):
        """Test validation of an invalid agent configuration"""
        invalid_agent = {
            "name": "Test Agent",
            # Missing required fields
            "tasks": [
                {
                    "id": "task1",
                    # Missing required fields
                    "blockName": "NonExistentBlock"
                }
            ]
        }
        
        result = self.validator.validate_agent(invalid_agent)
        
        self.assertFalse(result["is_valid"])
        self.assertGreater(len(result["errors"]), 0)
    
    def test_validate_unknown_tool(self):
        """Test validation with unknown tool"""
        agent_with_unknown_tool = {
            "name": "Test Agent",
            "description": "A test agent",
            "systemPrompt": "You are a helpful agent.",
            "tasks": [
                {
                    "id": "task1",
                    "name": "Unknown Task",
                    "blockName": "NonExistentBlock",
                    "inputs": {}
                }
            ],
            "memory": {"enabled": True, "keys": []}
        }
        
        result = self.validator.validate_agent(agent_with_unknown_tool)
        
        self.assertFalse(result["is_valid"])
        self.assertTrue(any("Unknown tool" in error for error in result["errors"]))

class TestToolRegistry(unittest.TestCase):
    """Test cases for the MockToolRegistry class"""
    
    def setUp(self):
        self.registry = MockToolRegistry()
    
    def test_get_tool_summaries(self):
        """Test getting tool summaries"""
        summaries = self.registry.get_tool_summaries()
        
        self.assertIsInstance(summaries, str)
        parsed = json.loads(summaries)
        self.assertIn("WebSearchBlock", parsed)
        self.assertIn("EmailWriterBlock", parsed)
    
    def test_validate_tool(self):
        """Test tool validation"""
        self.assertTrue(self.registry.validate_tool("WebSearchBlock"))
        self.assertTrue(self.registry.validate_tool("EmailWriterBlock"))
        self.assertFalse(self.registry.validate_tool("NonExistentBlock"))

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow"""
    
    def setUp(self):
        self.tool_registry = MockToolRegistry()
        self.llm = MockLLM()
        self.generator = AgentGenerator(self.llm, self.tool_registry)
        self.validator = AgentValidator(self.tool_registry)
    
    def test_complete_workflow(self):
        """Test the complete workflow from description to validated agent"""
        description = "Create a market research agent that searches the web and exports to Notion"
        
        # Mock LLM response
        mock_response = '''{
            "name": "Market Research Agent",
            "description": "Agent that researches markets and exports to Notion",
            "systemPrompt": "You are a market research assistant.",
            "tasks": [
                {
                    "id": "search",
                    "name": "Search Market Data",
                    "blockName": "WebSearchBlock",
                    "inputs": {"query": "market trends"}
                },
                {
                    "id": "export",
                    "name": "Export to Notion",
                    "blockName": "NotionExportBlock", 
                    "inputs": {"content": "{{search_results}}", "database_id": "test_db"}
                }
            ],
            "memory": {"enabled": true, "keys": ["search_results"]}
        }'''
        
        self.llm.responses = [mock_response]
        
        # Generate agent
        agent = self.generator.generate_agent(description)
        
        # Validate agent
        validation = self.validator.validate_agent(agent)
        
        # Assertions
        self.assertTrue(validation["is_valid"])
        self.assertEqual(agent["name"], "Market Research Agent")
        self.assertEqual(len(agent["tasks"]), 2)
        self.assertTrue(agent["memory"]["enabled"])

def run_performance_test():
    """Run a simple performance test"""
    import time
    
    tool_registry = MockToolRegistry()
    llm = MockLLM()
    generator = AgentGenerator(llm, tool_registry)
    
    # Test multiple generations
    descriptions = [
        "Create a sales outreach agent",
        "Build a content scheduler",
        "Make a data analysis agent",
        "Create a customer support agent",
        "Build a research assistant"
    ]
    
    start_time = time.time()
    
    for description in descriptions:
        agent = generator.generate_agent(description)
        # Basic validation
        assert "name" in agent
        assert "tasks" in agent
    
    end_time = time.time()
    
    print(f"Performance test completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per agent: {(end_time - start_time) / len(descriptions):.3f} seconds")

if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run performance test
    print("\nRunning performance test...")
    run_performance_test()
    
    print("\nAll tests completed!")