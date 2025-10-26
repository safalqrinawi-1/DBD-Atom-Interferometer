#!/usr/bin/env python3
"""
Inference script for trained DBD PPO model.
Usage: python run_inference.py [model_path] [--episodes N]
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dbd_env import DBDEnv

CURRENT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = CURRENT_DIR / "checkpoints"


def run_inference(model_path, n_episodes=10, render=False, deterministic=True):
    """
    Run inference using a trained model.
    
    Args:
        model_path: Path to the saved model (.zip file)
        n_episodes: Number of episodes to run
        render: Whether to render the environment (if supported)
        deterministic: Whether to use deterministic actions
    """
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    env = Monitor(DBDEnv())
    rewards = []
    efficiencies = []  # Track beam splitting efficiency
    
    print(f"\nRunning {n_episodes} episodes...\n")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        
        while not done:
            # Get action from the trained model
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(np.asarray(reward).sum())
            steps += 1
        
        rewards.append(ep_reward)
        efficiencies.append(info['beam_splitting_score'])  # Store final efficiency
        print(f"Episode {episode+1:3d}: Reward = {ep_reward:8.3f} | Efficiency = {info['beam_splitting_score']:.4f} | Steps = {steps:4d}")
    
    env.close()
    
    # Print statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    mean_eff = np.mean(efficiencies)
    std_eff = np.std(efficiencies)
    
    print(f"\n{'='*50}")
    print(f"Results over {n_episodes} episodes:")
    print(f"  Mean reward:     {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"  Min reward:      {min_reward:.3f}")
    print(f"  Max reward:      {max_reward:.3f}")
    print(f"  Mean efficiency: {mean_eff:.4f} ± {std_eff:.4f}")
    print(f"{'='*50}\n")
    
    return mean_reward


def list_available_checkpoints():
    """List all available model checkpoints."""
    if not CHECKPOINT_DIR.exists():
        print(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        return []
    
    checkpoints = sorted(CHECKPOINT_DIR.glob("*.zip"))
    if not checkpoints:
        print(f"No checkpoints found in {CHECKPOINT_DIR}")
        return []
    
    print("\nAvailable checkpoints:")
    for i, cp in enumerate(checkpoints, 1):
        print(f"  {i}. {cp.name}")
    print()
    return checkpoints


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained PPO model")
    parser.add_argument(
        "model_path",
        nargs="?",
        default=None,
        help="Path to model checkpoint (default: ppo_final.zip)"
    )
    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic"
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available checkpoints"
    )
    
    args = parser.parse_args()
    
    # List checkpoints if requested
    if args.list:
        list_available_checkpoints()
        sys.exit(0)
    
    # Determine model path
    if args.model_path is None:
        model_path = CHECKPOINT_DIR / "ppo_final.zip"
    else:
        model_path = Path(args.model_path)
    
    # Check if model exists
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("\nTrying to list available checkpoints...")
        checkpoints = list_available_checkpoints()
        if checkpoints:
            print("Please specify one of the above models.")
        sys.exit(1)
    
    # Run inference
    run_inference(
        model_path=model_path,
        n_episodes=args.episodes,
        deterministic=not args.stochastic
    )