#!/usr/bin/env python3
"""
Fix NATS A2A implementation - remove pass statements and add missing features
"""

import re

# Read the file
with open('core/src/aura_intelligence/communication/nats_a2a.py', 'r') as f:
    content = f.read()

# Fix pattern: Remove pass statements that are followed by actual code
# Pattern matches pass followed by actual code on next line(s)
content = re.sub(r'(\s*)pass\n\1(?=\S)', r'\1', content)

# Add missing _publish_with_headers method at the end of NATSA2ASystem class
publish_with_headers = '''
    async def _publish_with_headers(
        self,
        subject: str,
        message: AgentMessage,
        headers: Dict[str, str]
    ) -> None:
        """
        Publish message with headers including deduplication and tracing.
        
        Implements exactly-once semantics with Nats-Msg-Id.
        """
        if not self.js:
            raise RuntimeError("JetStream not initialized")
        
        # Prepare message data
        data = message.to_bytes()
        
        # Add deduplication header
        if "Nats-Msg-Id" not in headers:
            headers["Nats-Msg-Id"] = message.id
        
        try:
            # Publish with headers
            ack = await self.js.publish(
                subject=subject,
                payload=data,
                headers=headers,
                timeout=5.0
            )
            
            self.metrics['messages_sent'] += 1
            
            logger.info(
                "Message published with headers",
                subject=subject,
                message_id=message.id,
                sequence=ack.seq,
                headers=list(headers.keys())
            )
            
        except Exception as e:
            self.metrics['messages_failed'] += 1
            logger.error(f"Failed to publish with headers: {e}")
            raise'''

# Find the end of NATSA2ASystem class and add the method
class_end_pattern = r'(class NATSA2ASystem:.*?)((?=\nclass\s|\Z))'
match = re.search(class_end_pattern, content, re.DOTALL)
if match:
    # Insert before the next class or end of file
    insertion_point = match.end(1)
    # Find the last method in the class
    last_method_pattern = r'(\n    async def.*?\n(?:        .*\n)*)'
    methods = list(re.finditer(last_method_pattern, content[:insertion_point]))
    if methods:
        last_method_end = methods[-1].end()
        content = content[:last_method_end] + '\n' + publish_with_headers + '\n' + content[last_method_end:]

# Fix subject patterns - remove HTML entities
content = content.replace('&gt;', '>')
content = content.replace('&lt;', '<')

# Add missing imports at the top
if 'from typing import Dict' not in content:
    import_section = '''from typing import Dict, Any, List, Optional, Callable, AsyncIterator, Set
'''
    content = re.sub(r'(from typing import.*)', import_section, content, count=1)

# Write the fixed content
with open('core/src/aura_intelligence/communication/nats_a2a.py', 'w') as f:
    f.write(content)

print("Fixed NATS A2A implementation:")
print("- Removed standalone pass statements")
print("- Added _publish_with_headers method")
print("- Fixed HTML entities in subjects")
print("- Added missing type imports")