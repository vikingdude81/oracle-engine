#!/usr/bin/env python3
"""Quick test to verify all imports work."""

import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import builtins
try:
    import psutil
    builtins.psutil = psutil
except ImportError:
    pass

import sys
from pathlib import Path

# Add the oracle-engine path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

print("=" * 60)
print("Testing Oracle Engine imports...")
print("=" * 60)

# Test 1: helios_metrics
try:
    from consciousness_circuit.helios_metrics import compute_lyapunov_exponent
    print("✅ helios_metrics: OK")
except Exception as e:
    print(f"❌ helios_metrics: {e}")

# Test 2: tame_metrics
try:
    from consciousness_circuit.tame_metrics import TAMEMetrics
    print("✅ tame_metrics: OK")
except Exception as e:
    print(f"❌ tame_metrics: {e}")

# Test 3: UniversalCircuit
try:
    from consciousness_circuit import UniversalCircuit
    print("✅ UniversalCircuit: OK")
except Exception as e:
    print(f"❌ UniversalCircuit: {e}")

# Test 4: Unsloth
try:
    from unsloth import FastLanguageModel
    print("✅ unsloth: OK")
except Exception as e:
    print(f"❌ unsloth: {e}")

# Test 5: Check if custom model exists
from oracle_config import CUSTOM_MODEL_PATH as custom_path
if Path(custom_path).exists():
    print(f"✅ Custom model found at: {custom_path}")
else:
    print(f"⚠️  Custom model not found at: {custom_path}")

print("=" * 60)
print("Import tests complete!")
print("=" * 60)
