#!/usr/bin/env python3
"""
DBD Environment Test Script - Real-Time Adaptation

Test the environment's real-time adaptation features.
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
    """Simple console testing."""
    print(f"\n[TEST] DBD Environment - Real-Time Adaptation")
    print("=" * 70)

    env = DBDEnv()
    
    print(f"Configuration:")
    print(f"  Steps per episode: {env.steps_per_episode}")
    print(f"  Momentum grid: 21 points")
    print(f"  Polarization: random [0, 20%]")
    print(f"  Momentum width: random {env.momentum_width_range} ℏkL")
    print(f"  Real-time adaptation: YES")
    print(f"  Parameter drift: YES")
    print("=" * 70)

    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        print("-" * 70)
        
        obs, info = env.reset()
        
        init_pol = info['initial_polarization_error']
        init_mom = info['initial_momentum_width']
        
        print(f"Scenario: {info['scenario']}")
        print(f"Initial Polarization: {init_pol:.3f}")
        print(f"Initial Momentum Width: {init_mom:.3f}")
        print("-" * 70)
        
        total_reward = 0.0
        
        for step in range(env.steps_per_episode):
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 5 == 0 or terminated:
                print(f"  Step {step+1:2d}/{env.steps_per_episode}: "
                      f"Beam={info['beam_splitting_score']:.4f}, "
                      f"Pol={info['current_polarization_error']:.3f}, "
                      f"Mom={info['current_momentum_width']:.3f}")
        
        print(f"\n✓ Episode Complete!")
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"  Beam Score: {info['beam_splitting_score']:.4f}")
        print(f"  Parameter Changes:")
        print(f"    Pol: {init_pol:.3f} → {info['current_polarization_error']:.3f} "
              f"(Δ={info['current_polarization_error']-init_pol:+.3f})")
        print(f"    Mom: {init_mom:.3f} → {info['current_momentum_width']:.3f} "
              f"(Δ={info['current_momentum_width']-init_mom:+.3f})")
        print(f"  Populations: {info['populations']}")
        print("=" * 70)
    
    print("\n✓ Testing completed!")


def main():
    """Main test function."""
    print("=" * 70)
    print("DBD Environment Test - Real-Time Adaptation")
    print("=" * 70)
    
    while True:
        print("\nSelect test mode:")
        print("1. Simple console testing")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == '1':
            episodes = input("Number of episodes (default 3): ").strip()
            episodes = int(episodes) if episodes else 3
            test_simple(episodes)
        elif choice == '2':
            print("\n✓ Goodbye!")
            return
        else:
            print("✗ Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()