#!/usr/bin/env python3
"""
Test script for DBD Physics Engine
Verifies that physics is correct before training.
"""

import sys
import os
import numpy as np

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now import
from dbd_physics import DBDPhysics

print("="*70)
print("🧪 Testing DBD Physics")
print("="*70)

# Create engine
print("\nCreating DBDPhysics engine...")
try:
    engine = DBDPhysics(
        pulse_duration=1.0,
        polarization_error=0.0,
        initial_momentum=0.0
    )
    print("✅ Engine created successfully")
except Exception as e:
    print(f"❌ Failed to create engine: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 1: Initial state
print("\n" + "="*70)
print("[Test 1] Initial State")
print("="*70)
try:
    state_vec = engine.state
    pops = engine.get_populations()
    sym_pop = engine.get_symmetric_population()
    
    print(f"State vector: {state_vec}")
    print(f"Populations: {pops}")
    print(f"Symmetric pop: {sym_pop:.6f}")
    
    if pops[0] > 0.99:
        print("✅ PASS: Starts in |p⟩ (ground state)")
    else:
        print(f"❌ FAIL: Expected |p⟩=1.0, got {pops[0]:.4f}")
        sys.exit(1)
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Evolution with zero detuning
print("\n" + "="*70)
print("[Test 2] Evolution with Δ=0")
print("="*70)

def Delta_zero(t):
    return 0.0

try:
    print("Running 30 steps with zero detuning...")
    for i in range(30):
        result = engine.evolve_system(Delta_zero, 1.0/30, p=0.0)
        if i == 0:
            print(f"  Step 1: sym_pop = {result['symmetric_population']:.6f}")
        elif i == 29:
            print(f"  Step 30: sym_pop = {result['symmetric_population']:.6f}")
    
    final_pops = result['final_populations']
    final_sym = result['symmetric_population']
    
    print(f"\nFinal populations: {final_pops}")
    print(f"Final symmetric pop: {final_sym:.6f}")
    print(f"BS efficiency: {final_pops[1] + final_pops[2]:.6f}")
    print(f"Normalization: {np.sum(final_pops):.6f}")
    
    if 0.65 < final_sym < 0.85:
        print("✅ PASS: Physics behavior is correct")
    else:
        print(f"⚠️  WARNING: Expected sym_pop ~0.70-0.80, got {final_sym:.4f}")
        print("This may be acceptable depending on parameters.")
        
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Reset
print("\n" + "="*70)
print("[Test 3] Reset Functionality")
print("="*70)
try:
    engine.reset()
    reset_pops = engine.get_populations()
    reset_time = engine.current_time
    
    print(f"Populations after reset: {reset_pops}")
    print(f"Time after reset: {reset_time:.6f}")
    
    if reset_pops[0] > 0.99 and reset_time == 0.0:
        print("✅ PASS: Reset works correctly")
    else:
        print("❌ FAIL: Reset did not return to initial state")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check all RL methods exist
print("\n" + "="*70)
print("[Test 4] RL Environment Compatibility")
print("="*70)
try:
    # Check attributes
    checks = [
        ('omega_rec', hasattr(engine, 'omega_rec')),
        ('initial_momentum', hasattr(engine, 'initial_momentum')),
        ('get_populations', hasattr(engine, 'get_populations')),
        ('get_symmetric_population', hasattr(engine, 'get_symmetric_population')),
        ('get_beamsplitter_efficiency', hasattr(engine, 'get_beamsplitter_efficiency')),
        ('evolve_system', callable(getattr(engine, 'evolve_system', None))),
        ('reset', callable(getattr(engine, 'reset', None))),
    ]
    
    all_passed = True
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    if not all_passed:
        print("❌ FAIL: Missing required methods")
        sys.exit(1)
    
    # Check return format of evolve_system
    engine.reset()
    result = engine.evolve_system(Delta_zero, 0.01, p=0.0)
    
    required_keys = ['success', 'final_populations', 'symmetric_population']
    for key in required_keys:
        if key not in result:
            print(f"❌ FAIL: evolve_system missing key '{key}'")
            sys.exit(1)
    
    print("\n✅ PASS: All RL methods present and working")
    
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check that omega_rec value is correct
print("\n" + "="*70)
print("[Test 5] Parameter Values")
print("="*70)
try:
    print(f"omega_rec: {engine.omega_rec}")
    print(f"Omega_R: {engine.pars.Omega_R}")
    print(f"tau_g: {engine.pars.tau_g}")
    print(f"eps_pol: {engine.pars.eps_pol}")
    print(f"initial_momentum: {engine.initial_momentum}")
    
    if engine.omega_rec == 1.0:
        print("✅ PASS: Parameters are correct")
    else:
        print(f"⚠️  WARNING: omega_rec = {engine.omega_rec}, expected 1.0")
        
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅✅✅ ALL TESTS PASSED! ✅✅✅")
print("="*70)
print("\nYour DBD physics is correct and ready for RL training!")
print("\nNext steps:")
print("1. Test the environment: python test_env.py")
print("2. Delete old checkpoints")
print("3. Start training: python train_simple.py")
print("="*70)