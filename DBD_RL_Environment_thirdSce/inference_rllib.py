#!/usr/bin/env python3
"""
Inference script for trained DBD agent with D1 cost metrics.
Compatible with RLlib's new API stack (RLModule).
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from ray import tune
from ray.rllib.algorithms.ppo import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dbd_env import DBDEnv

# ========== CRITICAL: REGISTER ENVIRONMENT ==========
def create_dbd_env_for_inference(config=None):
    """Create DBD environment - must match training registration!"""
    return DBDEnv(
        sample_conditions=True,
        momentum_range=(-0.3, 0.3),
        eps_pol_range=(-0.05, 0.05)
    )

# Register the environment (REQUIRED for checkpoint loading!)
tune.register_env("DBDEnv-v0", create_dbd_env_for_inference)
# =====================================================


def run_inference(checkpoint_path: str, num_episodes: int = 1):
    """Run inference with trained agent using new RLlib API."""
    
    print(f"{'='*70}")
    print(f"  DBD AGENT INFERENCE (New API)")
    print(f"{'='*70}")
    print(f"\nLoading checkpoint from: {checkpoint_path}\n")
    
    # Load trained agent
    print("Loading trained PPO agent...")
    agent = PPO.from_checkpoint(checkpoint_path)
    print("✓ Agent loaded successfully!\n")
    
    # Get the RLModule for inference (new API)
    print("Extracting RLModule...")
    rl_module = agent.get_module()
    rl_module.eval()  # Set to evaluation mode
    print("✓ RLModule ready!\n")
    
    # Create test environment with FIXED conditions
    print("Creating test environment...")
    env = DBDEnv(
        sample_conditions=False,
        momentum_range=(0.0, 0.0),
        eps_pol_range=(0.0, 0.0)
    )
    print("✓ Environment created!\n")
    
    for episode in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"  EPISODE {episode + 1}")
        print(f"{'='*70}")
        
        obs, info = env.reset()
        
        print(f"\nInitial Conditions:")
        print(f"  Momentum (p):        {info['initial_momentum']:.4f} ℏkL")
        print(f"  Polarization (ε):    {info['eps_pol']:.4f}")
        print(f"\nInitial Populations:")
        pops = info['populations']
        print(f"  |0⟩={pops[0]:.3f}  |+2ℏk⟩={pops[1]:.3f}  |-2ℏk⟩={pops[2]:.3f}  "
              f"|+4ℏk⟩={pops[3]:.3f}  |-4ℏk⟩={pops[4]:.3f}")
        print(f"{'-'*70}")
        
        episode_reward = 0.0
        step = 0
        terminated = False
        truncated = False
        
        efficiencies = []
        costs = []
        actions_taken = []
        
        while not (terminated or truncated):
            # Convert observation to tensor batch (new API requirement)
            obs_batch = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            # Forward pass through RLModule (new API)
            with torch.no_grad():
                model_outputs = rl_module.forward_inference({"obs": obs_batch})
            
            # Extract action (new API returns distribution parameters)
            action_dist_params = model_outputs["action_dist_inputs"][0].cpu().numpy()
            
            # For Gaussian distribution, take the mean (greedy action)
            action_mean = action_dist_params[:env.action_space.shape[0]]
            action = np.clip(
                action_mean,
                a_min=env.action_space.low,
                a_max=env.action_space.high
            )
            
            actions_taken.append(action[0])
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
            episode_reward += reward
            step += 1
            
            # Track metrics
            efficiencies.append(info['beam_splitter_efficiency'])
            costs.append(info['total_cost'])
            
            # Print progress every 5 steps
            if step % 5 == 0 or terminated:
                print(f"Step {step:2d}: "
                      f"Action={action[0]:+.3f}, "
                      f"Eff={info['beam_splitter_efficiency']:.4f}, "
                      f"P+2={info['P_plus2']:.3f}, "
                      f"P-2={info['P_minus2']:.3f}, "
                      f"Rew={reward:.4f}")
        
        # Final results
        print(f"\n{'='*70}")
        print("  FINAL RESULTS")
        print(f"{'='*70}")
        
        stats = env.get_episode_statistics()
        pops = stats['final_populations']
        
        print(f"\n📊 Final Populations:")
        print(f"   |0⟩:      {pops[0]:.4f} ({pops[0]*100:.1f}%)")
        print(f"   |+2ℏkL⟩:  {pops[1]:.4f} ({pops[1]*100:.1f}%)")
        print(f"   |-2ℏkL⟩:  {pops[2]:.4f} ({pops[2]*100:.1f}%)")
        print(f"   |+4ℏkL⟩:  {pops[3]:.4f} ({pops[3]*100:.1f}%)")
        print(f"   |-4ℏkL⟩:  {pops[4]:.4f} ({pops[4]*100:.1f}%)")
        
        print(f"\n📐 D1 Cost Analysis:")
        print(f"   Term 1: {stats['cost_term1']:.4f}")
        print(f"   Term 2: {stats['cost_term2']:.4f}")
        print(f"   Term 3: {stats['cost_term3']:.4f}")
        print(f"   Total:  {stats['total_cost']:.4f}")
        print(f"   Efficiency: {stats['beam_splitter_efficiency']:.4f} "
              f"({stats['beam_splitter_efficiency']*100:.1f}%)")
        
        print(f"\n📈 Performance:")
        print(f"   Episode return:  {episode_reward:.4f}")
        print(f"   Total transfer:  {stats['total_transfer']:.4f}")
        print(f"   Peak efficiency: {max(efficiencies):.4f}")
        
        print(f"\n🎮 Actions:")
        print(f"   Mean: {np.mean(actions_taken):+.4f}")
        print(f"   Std:  {np.std(actions_taken):.4f}")
        print(f"   Range: [{min(actions_taken):+.3f}, {max(actions_taken):+.3f}]")
        
        # Grade
        eff = stats['beam_splitter_efficiency']
        if eff > 0.9:
            grade, status = "A+", "✓✓ EXCELLENT"
        elif eff > 0.8:
            grade, status = "A", "✓✓ VERY GOOD"
        elif eff > 0.7:
            grade, status = "B", "✓  GOOD"
        elif eff > 0.5:
            grade, status = "C", "~  MODERATE"
        else:
            grade, status = "F", "✗  POOR"
        
        print(f"\n💡 Assessment: {status}")
        print(f"   Grade: {grade}")
        print(f"{'='*70}\n")
    
    env.close()
    print("✓ Inference completed!\n")


if __name__ == "__main__":
    # Your checkpoint path
    checkpoint_path = "C:/Users/jbs/ray_results/DBD_D1_Cost_Training/PPO_DBDEnv-v0_27598_00000_0_lr=0.0003_2025-10-14_11-52-46/checkpoint_000000"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: Checkpoint not found!")
        print(f"   Path: {checkpoint_path}")
        sys.exit(1)
    
    # Run inference
    run_inference(checkpoint_path, num_episodes=1)