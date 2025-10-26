#!/usr/bin/env python3
"""
DBD Environment Test Script - Small Iterations

Test script for DBD environment.
Based on dbd_physics.py (unchanged).
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dbd_env import DBDEnv


def test_simple(episodes=3):
    """Simple console testing without visualization."""
    print(f"\n[TEST] DBD Environment - Small Iterations")
    print("=" * 70)

    env = DBDEnv()
    
    print(f"Configuration:")
    print(f"  Steps per episode: {env.steps_per_episode}")
    print(f"  Momentum grid: {env.mdist.p_grid_pts} points")
    print(f"  Polarization: random [0, 20%]")
    print(f"  Physics: omega_rec={env.params.omega_rec}, Omega_R={env.params.Omega_R}")
    print("=" * 70)

    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        print("-" * 50)
        
        obs, info = env.reset()
        
        print(f"Scenario: {info['scenario']}")
        print(f"Polarization Error: {info['polarization_error']:.3f}")
        print(f"Momentum: p0={info.get('momentum_p0', 0.0):.3f}, σp={info.get('momentum_sigma', 0.05):.3f} ℏkL")
        print("-" * 50)
        
        total_reward = 0.0
        
        for step in range(env.steps_per_episode):
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            beam_score = info['beam_splitting_score']
            
            if step % 5 == 0 or terminated:
                print(f"    Step {step+1:2d}/{env.steps_per_episode} | Beam: {beam_score:.4f} | Reward: {reward:.3f}")
        
        print(f"\nEpisode Complete! Total Reward: {total_reward:.4f}")
        print(f"Final populations: {info['populations']}")
        print(f"Beam splitting score: {info.get('beam_splitting_score', 0.0):.4f}")
        print("=" * 70)
    
    print("\n✓ Simple testing completed!")


def main():
    """Main test function."""
    print("=" * 70)
    print("DBD Environment Test Script - Small Iterations")
    print("Based on dbd_physics.py (UNCHANGED)")
    print("=" * 70)
    
    while True:
        print("\nSelect test mode:")
        print("1. Simple console testing (quick)")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == '1':
            episodes = input("Number of episodes to test (default 3): ").strip()
            episodes = int(episodes) if episodes else 3
            test_simple(episodes)
        elif choice == '2':
            print("\n✓ Goodbye!")
            return
        else:
            print("✗ Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()