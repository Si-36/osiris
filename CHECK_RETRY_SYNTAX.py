#!/usr/bin/env python3
import py_compile
import sys

try:
    py_compile.compile('core/src/aura_intelligence/resilience/retry.py', doraise=True)
    print("✅ retry.py syntax is valid!")
    sys.exit(0)
except py_compile.PyCompileError as e:
    print(f"❌ Syntax error in retry.py:")
    print(e)
    sys.exit(1)