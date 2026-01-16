#!/usr/bin/env python3
"""
Test Standalone Functionality
==============================

Verifies that each module works in isolation without importing siblings.
"""

import sys
import os
import subprocess

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def test_standalone_import(module_path, relative_to_root=False):
    """Test that a module can be imported standalone."""
    if relative_to_root:
        # Import from package
        cmd = f"python3 -c 'import {module_path}; print(\"OK\")'"
    else:
        # Run as standalone script
        cmd = f"python3 {module_path}"
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return True, "Success"
        else:
            return False, result.stderr[:200]
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def main():
    print("Testing Standalone Module Functionality")
    print("=" * 50)
    
    # Define modules to test (as standalone scripts with __main__)
    standalone_scripts = [
        "consciousness_circuit/metrics/lyapunov.py",
        "consciousness_circuit/metrics/hurst.py",
        "consciousness_circuit/metrics/msd.py",
        "consciousness_circuit/metrics/entropy.py",
        "consciousness_circuit/metrics/agency.py",
        "consciousness_circuit/classifiers/signal_class.py",
        "consciousness_circuit/plugins/base.py",
        "consciousness_circuit/plugins/attractor_lock.py",
        "consciousness_circuit/plugins/coherence_boost.py",
        "consciousness_circuit/plugins/goal_director.py",
        "consciousness_circuit/training/reward_model.py",
        "consciousness_circuit/training/preference_generator.py",
    ]
    
    # Define package imports to test
    package_imports = [
        "consciousness_circuit.metrics",
        "consciousness_circuit.classifiers",
        "consciousness_circuit.plugins",
        "consciousness_circuit.training",
        "consciousness_circuit.analyzers",
        "consciousness_circuit.benchmarks",
    ]
    
    results = []
    
    # Test standalone scripts
    print("\n1. Testing Standalone Scripts (with __main__):")
    print("-" * 50)
    for script in standalone_scripts:
        name = script.split('/')[-1]
        success, message = test_standalone_import(script)
        results.append((name, success))
        
        if success:
            print(f"   {GREEN}✓{RESET} {name:30s} - Standalone OK")
        else:
            print(f"   {RED}✗{RESET} {name:30s} - FAILED")
            print(f"      Error: {message}")
    
    # Test package imports
    print("\n2. Testing Package Imports:")
    print("-" * 50)
    for package in package_imports:
        name = package.split('.')[-1]
        success, message = test_standalone_import(package, relative_to_root=True)
        results.append((name, success))
        
        if success:
            print(f"   {GREEN}✓{RESET} {name:30s} - Import OK")
        else:
            print(f"   {RED}✗{RESET} {name:30s} - FAILED")
            print(f"      Error: {message}")
    
    # Test specific imports
    print("\n3. Testing Specific Function Imports:")
    print("-" * 50)
    
    test_imports = [
        "from consciousness_circuit.metrics import compute_lyapunov",
        "from consciousness_circuit.metrics import compute_hurst",
        "from consciousness_circuit.metrics import compute_msd",
        "from consciousness_circuit.classifiers import classify_signal",
        "from consciousness_circuit.plugins import AttractorLockPlugin",
        "from consciousness_circuit.training import ConsciousnessRewardModel",
        "from consciousness_circuit.benchmarks import get_test_suite",
    ]
    
    for import_stmt in test_imports:
        cmd = f"python3 -c '{import_stmt}; print(\"OK\")'"
        success, message = test_standalone_import(cmd, relative_to_root=False)
        results.append((import_stmt[:40], success))
        
        if success:
            print(f"   {GREEN}✓{RESET} {import_stmt[:40]:40s}")
        else:
            print(f"   {RED}✗{RESET} {import_stmt[:40]:40s}")
            print(f"      Error: {message}")
    
    # Summary
    print("\n" + "=" * 50)
    total = len(results)
    passed = sum(1 for _, success in results if success)
    failed = total - passed
    
    print(f"Total tests: {total}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    if failed > 0:
        print(f"{RED}Failed: {failed}{RESET}")
        return 1
    else:
        print(f"\n{GREEN}✓ All tests passed!{RESET}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
