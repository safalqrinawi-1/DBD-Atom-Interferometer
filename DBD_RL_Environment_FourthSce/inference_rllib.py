#!/usr/bin/env python3
"""
DBD RL Environment - Stable Baselines3 Inference Script

Run inference on a trained PPO policy for the DBDEnv.
Enhanced scenario: p0=0.0, σp=0.05ℏkL, ε_pol=30%

Examples:
  python inference_sb3.py
  python inference_sb3.py --checkpoint "./checkpoints/dbd_ppo_final.zip"
  python inference_sb3.py --episodes 10
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np

# Ensure local imports work
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dbd_env import DBDEnv

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    print("Stable Baselines3 not available. Install with: pip install stable-baselines3")
    SB3_AVAILABLE = False


def find_latest_checkpoint(base_dir: str = "./checkpoints/") -> Path:
    """Automatically find the latest checkpoint in the training directory."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {base_path}")
    
    # Look for the final model first
    final_model = base_path / "dbd_ppo_final.zip"
    if final_model.exists():
        return final_model
    
    # Look for best model from evaluation
    best_model = base_path / "best_model.zip"
    if best_model.exists():
        return best_model
    
    # Find all checkpoint files
    checkpoints = list(base_path.glob("dbd_ppo_*_steps.zip"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in: {base_path}")
    
    # Get the highest step number
    checkpoint_numbers = []
    for cp in checkpoints:
        try:
            # Extract step number from filename like "dbd_ppo_1000_steps.zip"
            parts = cp.stem.split('_')
            if len(parts) >= 3 and parts[-1] == 'steps':
                num = int(parts[-2])
                checkpoint_numbers.append((num, cp))
        except (ValueError, IndexError):
            continue
    
    if not checkpoint_numbers:
        raise FileNotFoundError(f"No valid checkpoints found in: {base_path}")
    
    latest_checkpoint = max(checkpoint_numbers, key=lambda x: x[0])[1]
    return latest_checkpoint


def create_env():
    """Create environment with same parameters as training."""
    env = DBDEnv(
        max_detuning=20.0,
        steps_per_episode=15,
        pulse_duration=1.0,
        polarization_error=0.3,
        momentum_p0=0.0,
        momentum_sigma=0.05,
    )
    return env


def run_inference(model_path: Path, num_episodes: int = 3, render: bool = True) -> dict:
    """Run inference using trained Stable Baselines3 PPO model."""
    if not SB3_AVAILABLE:
        raise ImportError("Stable Baselines3 not available")

    # Load trained model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    env = create_env()

    episode_rewards = []
    beam_scores = []
    final_populations = []

    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 60)

        obs, info = env.reset()
        total_reward = 0.0

        print(f"Starting Episode - Scenario: {info['scenario']}")
        print(f"Polarization Error: {info.get('polarization_error', 0.0):.3f}")
        print(f"Momentum: p0={info.get('momentum_p0', 0.0):.3f}, σp={info.get('momentum_sigma', 0.05):.3f} ℏkL")
        print("-" * 60)

        done = False
        step_index = 0
        while not done:
            step_index += 1

            # Get action from model (deterministic for inference)
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)

            if render and (step_index % 5 == 0 or done):
                print(
                    f"  Step {step_index:2d}: Reward = {float(reward):6.3f}, "
                    f"Beam Score = {info.get('beam_splitting_score', 0.0):.4f}"
                )

        episode_rewards.append(total_reward)
        beam_scores.append(info.get('beam_splitting_score', 0.0))
        final_populations.append(info.get('populations', np.zeros(5)))

        print(f"\nEpisode {episode + 1} completed!")
        print(f"Total reward: {total_reward:.4f}")
        print(f"Final beam splitting score: {info.get('beam_splitting_score', 0.0):.4f}")
        print(f"Final populations: {info.get('populations')}")
        print("=" * 60)

    env.close()

    # Summary statistics
    print(f"\n{'='*60}")
    print("Inference Summary")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Mean Reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    print(f"Mean Beam Score: {np.mean(beam_scores):.4f} ± {np.std(beam_scores):.4f}")
    print(f"Best Beam Score: {np.max(beam_scores):.4f}")
    print(f"Worst Beam Score: {np.min(beam_scores):.4f}")
    
    # Average populations
    avg_pops = np.mean(final_populations, axis=0)
    print(f"\nAverage Final Populations:")
    print(f"  |0⟩        = {avg_pops[0]:.4f}")
    print(f"  |+2ℏk⟩   = {avg_pops[1]:.4f}")
    print(f"  |-2ℏk⟩   = {avg_pops[2]:.4f}")
    print(f"  |+4ℏk⟩   = {avg_pops[3]:.4f}")
    print(f"  |-4ℏk⟩   = {avg_pops[4]:.4f}")
    print(f"  Total ±2ℏk = {avg_pops[1] + avg_pops[2]:.4f}")
    print("=" * 60)

    return {
        'episode_rewards': episode_rewards,
        'beam_scores': beam_scores,
        'final_populations': final_populations,
    }


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Run inference on trained DBD RL model - Stable Baselines3"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (auto-finds latest if not provided)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable episode rendering"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DBD RL Inference Script - Stable Baselines3")
    print("=" * 60)
    
    if not SB3_AVAILABLE:
        print("\n✗ Stable Baselines3 not available")
        print("Please install Stable Baselines3 to use this inference script:")
        print("pip install stable-baselines3")
        return
    
    # Auto-find checkpoint if not provided
    if args.checkpoint is None:
        print("\n🔍 No checkpoint provided - searching for latest checkpoint...")
        try:
            checkpoint_path = find_latest_checkpoint()
            print(f"✓ Found checkpoint: {checkpoint_path.name}")
        except FileNotFoundError as e:
            print(f"✗ Could not find checkpoint: {e}")
            print("\nPlease run training first or provide checkpoint path with --checkpoint")
            return
    else:
        checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("Please provide a valid checkpoint path using --checkpoint")
        return
    
    print(f"\nLoading model from: {checkpoint_path}")
    
    try:
        results = run_inference(
            checkpoint_path,
            num_episodes=args.episodes,
            render=not args.no_render
        )
        
        print("\n✓ Inference completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()