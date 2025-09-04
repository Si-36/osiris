#!/usr/bin/env python3
"""
Fix critical syntax errors in the import chain
==============================================
Focus on files that are blocking our persistence tests
"""

import ast
import os

# Critical files in our import chain based on the error messages
CRITICAL_FILES = [
    'core/src/aura_intelligence/agents/resilience/circuit_breaker.py',
    'core/src/aura_intelligence/agents/resilience/fallback_agent.py',
    'core/src/aura_intelligence/agents/resilience/bulkhead.py',
    'core/src/aura_intelligence/observability/gpu_monitoring.py',
    'core/src/aura_intelligence/observability/langsmith_integration.py',
    'core/src/aura_intelligence/observability/opentelemetry_integration.py',
]

def check_file(filepath):
    """Check if a file has syntax errors"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content, filename=filepath)
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)

def main():
    print("🔍 Checking critical files for syntax errors...")
    print("=" * 60)
    
    all_good = True
    
    for filepath in CRITICAL_FILES:
        if os.path.exists(filepath):
            success, error = check_file(filepath)
            if success:
                print(f"✅ {filepath}")
            else:
                print(f"❌ {filepath}")
                print(f"   Error: {error}")
                all_good = False
        else:
            print(f"⚠️  {filepath} - Not found")
    
    print("\n" + "=" * 60)
    
    if all_good:
        print("✅ All critical files are syntax error free!")
    else:
        print("❌ Some critical files still have syntax errors")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    exit(main())