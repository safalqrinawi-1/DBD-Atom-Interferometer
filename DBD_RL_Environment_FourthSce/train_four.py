#!/usr/bin/env python3
import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dbd_env import DBDEnv


def create_env():
    env = DBDEnv(
        max_detuning=20.0,
        steps_per_episode=15,
        pulse_duration=1.0,
        polarization_error=0.3,
        momentum_p0=0.0,
        momentum_sigma=0.05,
    )
    return Monitor(env)


def make_env(rank, seed=0):
    def _init():
        env = create_env()
        env.reset(seed=seed + rank)
        return env
    return _init


def train():
    log_dir, checkpoint_dir = "./logs/", "./checkpoints/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    n_envs = 4
    use_subproc = True
    env = (
        SubprocVecEnv([make_env(i) for i in range(n_envs)])
        if use_subproc
        else DummyVecEnv([create_env for _ in range(n_envs)])
    )
    eval_env = DummyVecEnv([create_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=checkpoint_dir,
        name_prefix="dbd_ppo",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=150,
        batch_size=2000,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        device="auto",
    )

    model.learn(
        total_timesteps=100000,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    model.save(os.path.join(checkpoint_dir, "dbd_ppo_final"))
    env.close()
    eval_env.close()
    return model


def test_trained_model(model_path, n_episodes=10):
    model = PPO.load(model_path)
    env = create_env()
    
    rewards = []
    efficiencies = []  # Track beam splitting efficiency
    
    for i in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(np.asarray(reward).sum())
        
        rewards.append(ep_reward)
        efficiencies.append(info['beam_splitting_score'])  # Store final efficiency
        print(f"Episode {i+1:2d}: Reward = {ep_reward:.3f}, Efficiency = {info['beam_splitting_score']:.4f}")
    
    env.close()
    
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    mean_eff = float(np.mean(efficiencies))
    std_eff = float(np.std(efficiencies))
    
    print(f"\nMean reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"Mean efficiency: {mean_eff:.4f} ± {std_eff:.4f}")
    
    return mean_reward, mean_eff


if __name__ == "__main__":
    model = train()
    test_trained_model(os.path.join("./checkpoints/", "dbd_ppo_final"))

