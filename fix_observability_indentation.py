#!/usr/bin/env python3
"""Fix all remaining indentation issues in observability.py"""

import re

# Read the file
with open('/workspace/core/src/aura_intelligence/agents/observability.py', 'r') as f:
    content = f.read()

# Fix the trace_tool_call method body
content = re.sub(
    r'(async def trace_tool_call.*?:.*?\n)(.*?"""Context manager for tracing tool calls\."""\n)(.*?with self\.tracer\.start_as_current_span\(\n)(f"tool\.\{tool_name\}",)',
    r'\1\2        \3            \4',
    content,
    flags=re.DOTALL
)

# Fix remaining method definitions and their bodies
# Look for patterns where methods are not properly indented
content = re.sub(
    r'^(\s*)def\s+(\w+)\(self.*?\):\s*\n(\s*)"""',
    r'\1def \2(self):\n\1    """',
    content,
    flags=re.MULTILINE
)

# Fix extract_context method
content = content.replace(
    '    def extract_context(self, carrier: Dict[str, str]) -> context.Context:\n"""Extract trace context from carrier."""\nreturn self.propagator.extract(carrier)',
    '    def extract_context(self, carrier: Dict[str, str]) -> context.Context:\n        """Extract trace context from carrier."""\n        return self.propagator.extract(carrier)'
)

# Fix inject_context method
content = content.replace(
    '    def inject_context(self, carrier: Dict[str, str]) -> None:\n"""Inject current trace context into carrier."""\nself.propagator.inject(carrier)',
    '    def inject_context(self, carrier: Dict[str, str]) -> None:\n        """Inject current trace context into carrier."""\n        self.propagator.inject(carrier)'
)

# Fix create_span_link method
content = content.replace(
    '    def create_span_link(self, trace_id: str, span_id: str) -> Link:\n"""Create a span link for cross-agent correlation."""\nreturn Link(',
    '    def create_span_link(self, trace_id: str, span_id: str) -> Link:\n        """Create a span link for cross-agent correlation."""\n        return Link('
)

# Write the fixed content back
with open('/workspace/core/src/aura_intelligence/agents/observability.py', 'w') as f:
    f.write(content)

print("Fixed indentation issues in observability.py")