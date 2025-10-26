from dbd_env import DBDEnv
import numpy as np
env = DBDEnv()
print('Testing environment rewards...\n')
obs, info = env.reset()
print(f'Initial state:')
print(f'  Populations: {info["populations"]}')
print(f'  Beam score: {info["beam_splitting_score"]}')
print(f'  Pol error: {info["initial_polarization_error"]:.3f}')
print(f'  Mom width: {info["initial_momentum_width"]:.3f}')
print(f'\nRunning 15 steps...')
total_reward = 0.0
for step in range(15):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if step % 5 == 0 or step == 14:
        print(f'\nStep {step+1}:')
        print(f'  Reward: {reward:.6f}')
        print(f'  Beam score: {info["beam_splitting_score"]:.4f}')
print(f'\nFinal total reward: {total_reward:.6f}')
if total_reward == 0.0:
    print('PROBLEM: Reward is zero!')
else:
    print('Rewards working!')