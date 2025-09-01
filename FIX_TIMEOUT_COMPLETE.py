#!/usr/bin/env python3
"""Complete fix for timeout.py indentation"""

# Read the file
with open('core/src/aura_intelligence/resilience/timeout.py', 'r') as f:
    content = f.read()

# Fix specific patterns that are causing issues
import re

# Fix method bodies that are not indented properly
# Pattern: lines after method definition that should be indented
content = re.sub(
    r'(\n    def [^\n]+:\n        """[^"]+"""\n)([a-zA-Z])',
    r'\1        \2',
    content
)

# Fix if/else blocks without proper indentation
content = re.sub(
    r'\nif ([^\n]+):\n([^\s])',
    r'\n        if \1:\n            \2',
    content
)

content = re.sub(
    r'\nelse:\n([^\s])',
    r'\n        else:\n            \2',
    content
)

content = re.sub(
    r'\nelif ([^\n]+):\n([^\s])',
    r'\n        elif \1:\n            \2',
    content
)

# Fix return statements that are not indented
content = re.sub(
    r'\nreturn ([^\n]+)',
    r'\n        return \1',
    content
)

# Fix raise statements that are not indented
content = re.sub(
    r'\nraise ([^\n]+)',
    r'\n            raise \1',
    content
)

# Fix lines that start with variable assignments at wrong level
content = re.sub(
    r'\n([a-zA-Z_][a-zA-Z0-9_]* = [^\n]+)',
    r'\n        \1',
    content
)

# Fix lines starting with method calls
content = re.sub(
    r'\n(self\.[^\n]+)',
    r'\n        \1',
    content
)

# Fix logger and tracer calls
content = re.sub(
    r'\n(logger\.[^\n]+)',
    r'\n        \1',
    content
)

content = re.sub(
    r'\n(tracer\.[^\n]+)',
    r'\n        \1',
    content
)

# Fix async with blocks
content = re.sub(
    r'\nasync with ([^\n]+):\n([^\s])',
    r'\n        async with \1:\n            \2',
    content
)

# Fix with blocks
content = re.sub(
    r'\nwith ([^\n]+):\n([^\s])',
    r'\n        with \1:\n            \2',
    content
)

# Fix for loops
content = re.sub(
    r'\nfor ([^\n]+):\n([^\s])',
    r'\n        for \1:\n            \2',
    content
)

# Fix while loops
content = re.sub(
    r'\nwhile ([^\n]+):\n([^\s])',
    r'\n        while \1:\n            \2',
    content
)

# Fix try blocks
content = re.sub(
    r'\ntry:\n([^\s])',
    r'\n        try:\n            \1',
    content
)

content = re.sub(
    r'\nexcept([^\n]*):\n([^\s])',
    r'\n        except\1:\n            \2',
    content
)

content = re.sub(
    r'\nfinally:\n([^\s])',
    r'\n        finally:\n            \1',
    content
)

# Clean up any double indentation that might have been introduced
content = re.sub(r'\n        {8,}', '\n        ', content)

# Write the fixed content
with open('core/src/aura_intelligence/resilience/timeout.py', 'w') as f:
    f.write(content)

print("✅ Applied comprehensive fixes to timeout.py")