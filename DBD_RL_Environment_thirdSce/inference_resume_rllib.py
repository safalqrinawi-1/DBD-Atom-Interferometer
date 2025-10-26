#!/usr/bin/env python3
"""
Inference for PPO DBD agent (matches resume_train.py environment + new API stack).

Usage examples:
  python inference_resume.py --ckpt "C:/Users/jbs/ray_results/DBD_D1_Cost_Training/EXP_DIR" --episodes 1
  python inference_resume.py --ckpt "C:/Users/jbs/ray_results/.../checkpoint_000120" --episodes 3
  python inference_resume.py --ckpt "..." --p0 0.05 --eps 0.1      # fixed test
  python inference_resume.py --ckpt "..." --random                  # randomized like training
"""

import os
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from ray import tune
from ray.rllib.algorithms.ppo import PPO

# Allow local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dbd_env import DBDEnv  # noqa: E402

# --- Env registration (match resume_train.py!) ---
def create_dbd_env_for_inference(config=None):
    return DBDEnv(
        sample_conditions=True,
        momentum_range=(-0.3, 0.3),
        # typical polarization error 0..0.3
        eps_pol_range=(0.0, 0.3),
    )
tune.register_env("DBDEnv-v0", create_dbd_env_for_inference)


def latest_checkpoint(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    # If a file or a "checkpoint_*" path is provided directly, use it.
    if p.is_file() or p.name.startswith("checkpoint_"):
        return str(p)
    # Otherwise, pick newest checkpoint_* under the directory.
    ckpts = []
    for cand in p.glob("checkpoint_*"):
        m = re.search(r"checkpoint_(\d+)", cand.name)
        if m:
            ckpts.append((int(m.group(1)), cand))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint_* found under: {path}")
    ckpts.sort(key=lambda t: t[0], reverse=True)
    return str(ckpts[0][1])


def build_eval_env(randomize: bool, p0: float, eps: float) -> DBDEnv:
    if randomize:
        return DBDEnv(sample_conditions=True,
                      momentum_range=(-0.3, 0.3),
                      eps_pol_range=(0.0, 0.3))
    else:
        return DBDEnv(sample_conditions=False,
                      momentum_range=(p0, p0),
                      eps_pol_range=(eps, eps))


def run_episode(env: DBDEnv, rl_module, print_every=5):
    obs, info = env.reset()
    print("\nInitial Conditions:")
    print(f"  Momentum p0:       {info['initial_momentum']:.4f} (ℏk_L)")
    print(f"  Polarization ε:    {info['eps_pol']:.4f}")
    pops = info["populations"]
    print("Initial Populations:")
    print(f"  |0⟩={pops[0]:.3f}  |+2ℏk⟩={pops[1]:.3f}  |-2ℏk⟩={pops[2]:.3f}  "
          f"|+4ℏk⟩={pops[3]:.3f}  |-4ℏk⟩={pops[4]:.3f}")
    print("-" * 68)

    terminated = truncated = False
    ep_ret = 0.0
    step = 0
    efficiencies, costs, actions = [], [], []

    while not (terminated or truncated):
        obs_batch = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            model_out = rl_module.forward_inference({"obs": obs_batch})

        act_dim = env.action_space.shape[0]
        act_params = model_out["action_dist_inputs"][0].cpu().numpy()
        action_mean = act_params[:act_dim]
        action = np.clip(action_mean, env.action_space.low, env.action_space.high)

        # store scalar action for printing
        a0 = float(action[0])
        actions.append(a0)

        obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
        ep_ret += float(reward)
        step += 1
        efficiencies.append(info["beam_splitter_efficiency"])
        costs.append(info["total_cost"])

        if (step % print_every == 0) or terminated or truncated:
            print(f"Step {step:02d}: Action={a0:+.4f}  "
                  f"Eff={info['beam_splitter_efficiency']:.4f}  "
                  f"P+2={info['P_plus2']:.3f}  P-2={info['P_minus2']:.3f}  "
                  f"Rew={reward:.4f}")

    stats = env.get_episode_statistics()
    pops = stats["final_populations"]

    print("\nFINAL POPULATIONS")
    print(f"  |0⟩:      {pops[0]:.4f} ({pops[0]*100:4.1f}%)")
    print(f"  |+2ℏkL⟩:  {pops[1]:.4f} ({pops[1]*100:4.1f}%)")
    print(f"  |-2ℏkL⟩:  {pops[2]:.4f} ({pops[2]*100:4.1f}%)")
    print(f"  |+4ℏkL⟩:  {pops[3]:.4f} ({pops[3]*100:4.1f}%)")
    print(f"  |-4ℏkL⟩:  {pops[4]:.4f} ({pops[4]*100:4.1f}%)")

    print("\nD1 COST TERMS")
    print(f"  Term1: {stats['cost_term1']:.4f}")
    print(f"  Term2: {stats['cost_term2']:.4f}")
    print(f"  Term3: {stats['cost_term3']:.4f}")
    print(f"  Total: {stats['total_cost']:.4f}")
    print(f"  Efficiency: {stats['beam_splitter_efficiency']:.4f} "
          f"({stats['beam_splitter_efficiency']*100:.1f}%)")

    print("\nPERFORMANCE")
    print(f"  Episode return:  {ep_ret:.4f}")
    print(f"  Total transfer:  {stats['total_transfer']:.4f}")
    print(f"  Peak efficiency: {max(efficiencies):.4f}")

    actions = np.array(actions, dtype=np.float32)
    print("\nACTIONS")
    print(f"  Mean:  {actions.mean():+.4f}")
    print(f"  Std:   {actions.std():.4f}")
    print(f"  Range: [{actions.min():+.4f}, {actions.max():+.4f}]")

    eff = stats["beam_splitter_efficiency"]
    if eff > 0.9:
        grade, status = "A+", "EXCELLENT"
    elif eff > 0.8:
        grade, status = "A", "VERY GOOD"
    elif eff > 0.7:
        grade, status = "B", "GOOD"
    elif eff > 0.5:
        grade, status = "C", "MODERATE"
    else:
        grade, status = "F", "POOR"
    print(f"\nASSESSMENT: {status}  (Grade {grade})\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Path to checkpoint_* dir/file OR an experiment folder (auto-picks newest).")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--random", action="store_true",
                    help="Sample conditions like training. Otherwise fixed p0/eps.")
    ap.add_argument("--p0", type=float, default=0.0, help="Fixed initial momentum (ℏ k_L).")
    ap.add_argument("--eps", type=float, default=0.0, help="Fixed polarization error ε (0..0.3).")
    args = ap.parse_args()

    ckpt_path = latest_checkpoint(args.ckpt)
    print("=" * 70)
    print("   DBD PPO INFERENCE (new API)")
    print("=" * 70)
    print("Using checkpoint:", ckpt_path)

    agent = PPO.from_checkpoint(ckpt_path)
    rl_module = agent.get_module()
    rl_module.eval()

    env = build_eval_env(args.random, args.p0, args.eps)
    for ep in range(args.episodes):
        print("\n" + "=" * 70)
        print(f"EPISODE {ep + 1}")
        print("=" * 70)
        run_episode(env, rl_module)
    env.close()
    print("\n✓ Inference completed.")


if __name__ == "__main__":
    main()
