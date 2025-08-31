#!/usr/bin/env python3
"""
Fix the bulkhead.py file specifically
=====================================
This is blocking all persistence tests
"""

import re

def fix_bulkhead():
    """Fix the bulkhead.py syntax errors"""
    filepath = "core/src/aura_intelligence/agents/resilience/bulkhead.py"
    
    print(f"🔧 Fixing {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Split into lines for processing
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        in_class_bulkhead = False
        class_indent = ""
        
        while i < len(lines):
            line = lines[i]
            
            # Detect Bulkhead class
            if 'class Bulkhead:' in line:
                in_class_bulkhead = True
                class_indent = line[:len(line) - len(line.lstrip())]
                fixed_lines.append(line)
                i += 1
                continue
            
            # Fix __init__ method
            if in_class_bulkhead and '__init__' in line and 'def' in line:
                # Ensure proper indentation
                fixed_lines.append(f"{class_indent}    def __init__(self, config: BulkheadConfig):")
                i += 1
                
                # Skip the docstring and pass
                while i < len(lines) and (lines[i].strip().startswith('"""') or 
                                        lines[i].strip() == 'pass' or
                                        not lines[i].strip()):
                    if lines[i].strip().startswith('"""'):
                        fixed_lines.append(f'{class_indent}        """Initialize bulkhead."""')
                    i += 1
                
                # Add the __init__ body with proper indentation
                init_body = '''        config.validate()
        self.config = config
        self.stats = BulkheadStats()
        self.logger = structlog.get_logger().bind(bulkhead=config.name)
        
        # Semaphore for limiting concurrent executions
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        
        # Queue for waiting executions
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self._lock = asyncio.Lock()'''
                
                for body_line in init_body.split('\n'):
                    fixed_lines.append(f"{class_indent}    {body_line}")
                
                # Skip the original __init__ body
                while i < len(lines) and not (lines[i].strip().startswith('def') or 
                                             lines[i].strip().startswith('async def') or
                                             lines[i].strip().startswith('@') or
                                             (lines[i] and not lines[i].startswith(class_indent + ' '))):
                    i += 1
                continue
            
            # Fix execute method
            if in_class_bulkhead and 'def execute' in line and 'async' not in lines[i-1]:
                # Make it async and fix indentation
                fixed_lines.append(f"{class_indent}    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:")
                i += 1
                
                # Skip until we find the docstring
                while i < len(lines) and not lines[i].strip().startswith('"""'):
                    i += 1
                
                # Keep the rest of the method
                continue
            
            # Fix _execute_with_bulkhead method
            if in_class_bulkhead and 'def _execute_with_bulkhead' in line and 'async' not in lines[i-1]:
                fixed_lines.append(f"{class_indent}    async def _execute_with_bulkhead(self, func: Callable[..., T], *args, **kwargs) -> T:")
                i += 1
                continue
            
            # Fix health_check method
            if in_class_bulkhead and 'def health_check' in line and 'async' not in lines[i-1]:
                fixed_lines.append(f"{class_indent}    async def health_check(self) -> Dict[str, Any]:")
                i += 1
                # Skip pass statements after the method definition
                while i < len(lines) and lines[i].strip() == 'pass':
                    i += 1
                continue
            
            # Remove standalone pass statements that break indentation
            if line.strip() == 'pass' and i + 1 < len(lines):
                next_line = lines[i + 1]
                # If next line is a docstring or return statement, skip the pass
                if (next_line.strip().startswith('"""') or 
                    next_line.strip().startswith('return') or
                    next_line.strip().startswith('self.')):
                    i += 1
                    continue
            
            # Fix method indentation inside Bulkhead class
            if in_class_bulkhead and line.strip() and not line.startswith(class_indent):
                # Check if this should be inside the class
                if any(keyword in line for keyword in ['def ', 'async def', '@asynccontextmanager', 'return', 'if ', 'try:', 'except', 'finally:', 'with ']):
                    # This line is likely part of a method, fix indentation
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent < len(class_indent):
                        # This is wrongly dedented, fix it
                        line = class_indent + '    ' + line.lstrip()
            
            # End of class detection
            if in_class_bulkhead and line and not line.startswith(' ') and not line.startswith('\t'):
                in_class_bulkhead = False
            
            fixed_lines.append(line)
            i += 1
        
        # Join lines back
        fixed_content = '\n'.join(fixed_lines)
        
        # Write back
        with open(filepath, 'w') as f:
            f.write(fixed_content)
        
        print(f"✅ Fixed {filepath}")
        print("\nNow try running:")
        print("  python3 TEST_PERSISTENCE_NOW.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_bulkhead()