#!/usr/bin/env python3
"""
DBD RL Environment - Real-Time Adaptation

Gymnasium environment with real-time parameter adaptation.

Features:
- Real-time adaptation with randomized parameters
- Polarization error: 0-20% (randomized per episode)
- Momentum width: σp ∈ [0.03, 0.07]ℏkL (randomized per episode)
- Parameter drift during episodes (optional)
- Beam splitting efficiency optimization
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dbd_physics import DBDPhysics, MomentumDist, DBDParams


class DBDEnv(gym.Env):
    """DBD Environment with real-time parameter adaptation."""
    
    def __init__(self,
                 max_detuning: float = 20.0,
                 steps_per_episode: int = 15,
                 pulse_duration: float = 1.0,
                 polarization_error: float = 0.3,  # Default, will be randomized
                 momentum_p0: float = 0.0,
                 momentum_sigma: float = 0.05,  # Default, will be randomized
                 polarization_range: tuple = (0.0, 0.2),
                 momentum_sigma_range: tuple = (0.03, 0.07),
                 enable_parameter_drift: bool = True):
        
        super().__init__()
        
        # Environment parameters
        self.max_detuning = max_detuning
        self.steps_per_episode = steps_per_episode
        self.pulse_duration = pulse_duration
        self.momentum_p0 = momentum_p0
        
        # Randomization ranges
        self.polarization_range = polarization_range
        self.momentum_sigma_range = momentum_sigma_range
        self.enable_parameter_drift = enable_parameter_drift
        
        # Initial momentum distribution (will be updated in reset)
        self.mdist = MomentumDist(
            p0=momentum_p0,
            sigma_p=momentum_sigma,
            p_grid_pts=121  # Good balance
        )
        
        # Initialize DBD parameters
        self.params = DBDParams(
            omega_rec=1.0,
            Omega_R=2.0,
            tau_g=0.47,
            t_center=0.5 * pulse_duration,
            eps_pol=polarization_error
        )
        
        # Initialize physics engine
        self.physics = DBDPhysics(
            pulse_duration=pulse_duration,
            polarization_error=polarization_error,
            mdist=self.mdist,
            params=self.params
        )
        
        # Action space: single detuning value
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: [time, pop_0, pop_+2, pop_-2, pop_+4, pop_-4, pol_error, mom_sigma]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.previous_beam_score = 0.0
        self.episode_reward = 0.0
        
        # Current randomized parameters
        self.current_polarization_error = polarization_error
        self.current_momentum_sigma = momentum_sigma
        self.initial_polarization_error = polarization_error
        self.initial_momentum_sigma = momentum_sigma
        
        # Scenario info
        self.scenario_name = "real_time_adaptation"
        self.scenario_description = f"RT_ADAPT: pol∈{polarization_range}, σp∈{momentum_sigma_range}ℏkL"
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment with randomized parameters."""
        super().reset(seed=seed)
        
        # Randomize polarization error
        self.initial_polarization_error = self.np_random.uniform(
            self.polarization_range[0],
            self.polarization_range[1]
        )
        self.current_polarization_error = self.initial_polarization_error
        
        # Randomize momentum width
        self.initial_momentum_sigma = self.np_random.uniform(
            self.momentum_sigma_range[0],
            self.momentum_sigma_range[1]
        )
        self.current_momentum_sigma = self.initial_momentum_sigma
        
        # Create new momentum distribution with randomized width
        self.mdist = MomentumDist(
            p0=self.momentum_p0,
            sigma_p=self.current_momentum_sigma,
            p_grid_pts=121
        )
        
        # Update params with randomized polarization
        self.params.eps_pol = self.current_polarization_error
        
        # Reinitialize physics with new parameters
        self.physics = DBDPhysics(
            pulse_duration=self.pulse_duration,
            polarization_error=self.current_polarization_error,
            mdist=self.mdist,
            params=self.params
        )
        
        # Reset tracking
        self.current_step = 0
        self.previous_beam_score = 0.0
        self.episode_reward = 0.0
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'episode_reward': 0.0,
            'populations': self.physics.get_populations(),
            'beam_splitting_score': 0.0,
            'initial_polarization_error': self.initial_polarization_error,
            'current_polarization_error': self.current_polarization_error,
            'initial_momentum_sigma': self.initial_momentum_sigma,
            'current_momentum_sigma': self.current_momentum_sigma,
            'polarization_error': self.current_polarization_error,  # For compatibility
            'momentum_sigma': self.current_momentum_sigma,           # FIXED: was 'momentum_width'
            'momentum_p0': self.momentum_p0,
            'scenario': self.scenario_name
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step with real-time adaptation."""
        # Convert action to detuning
        detuning = float(action[0] * self.max_detuning)
        
        # Time segment
        time_segment = self.pulse_duration / self.steps_per_episode
        
        # Create detuning function
        def detuning_function(t):
            return detuning
        
        # Evolve physics
        populations = self.physics.evolve(detuning_function, time_segment)
        
        # Calculate beam score
        current_beam_score = self.physics.get_beam_splitting_score()
        
        # Reward: scaled for better learning
        improvement = (current_beam_score - self.previous_beam_score) * 10.0
        absolute = current_beam_score ** 2
        
        reward = 0.5 * improvement + 0.5 * absolute
        
        # Final step bonus
        if self.current_step == self.steps_per_episode - 1:
            reward += 3.0 * current_beam_score
        
        # Update tracking
        self.previous_beam_score = current_beam_score
        self.episode_reward += reward
        self.current_step += 1
        
        # Simulate parameter drift (if enabled)
        if self.enable_parameter_drift and self.current_step % 5 == 0:
            # Small drift in polarization error
            drift_pol = self.np_random.uniform(-0.01, 0.01)
            self.current_polarization_error = np.clip(
                self.current_polarization_error + drift_pol,
                self.polarization_range[0],
                self.polarization_range[1]
            )
            
            # Small drift in momentum width
            drift_mom = self.np_random.uniform(-0.005, 0.005)
            self.current_momentum_sigma = np.clip(
                self.current_momentum_sigma + drift_mom,
                self.momentum_sigma_range[0],
                self.momentum_sigma_range[1]
            )
        
        # Get observation
        observation = self._get_observation()
        
        # Check termination
        terminated = self.current_step >= self.steps_per_episode
        truncated = False
        
        # Info
        info = {
            'episode_reward': self.episode_reward,
            'populations': populations,
            'beam_splitting_score': current_beam_score,
            'symmetric_population': self.physics.get_symmetric_population(),
            'initial_polarization_error': self.initial_polarization_error,
            'current_polarization_error': self.current_polarization_error,
            'initial_momentum_sigma': self.initial_momentum_sigma,
            'current_momentum_sigma': self.current_momentum_sigma,
            'polarization_error': self.current_polarization_error,  # For compatibility
            'momentum_sigma': self.current_momentum_sigma,           # FIXED: was 'momentum_width'
            'momentum_p0': self.momentum_p0,
            'scenario': self.scenario_name,
            'step': self.current_step
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation with parameter information."""
        normalized_time = self.current_step / self.steps_per_episode
        populations = self.physics.get_populations()
        
        # Normalize polarization error to [0, 1]
        pol_range = self.polarization_range[1] - self.polarization_range[0]
        normalized_pol = (self.current_polarization_error - self.polarization_range[0]) / pol_range
        
        # Normalize momentum sigma to [0, 1]
        mom_range = self.momentum_sigma_range[1] - self.momentum_sigma_range[0]
        normalized_mom = (self.current_momentum_sigma - self.momentum_sigma_range[0]) / mom_range
        
        observation = np.array([
            normalized_time,
            populations[0],  # |0⟩
            populations[1],  # |+2ℏkL⟩
            populations[2],  # |-2ℏkL⟩
            populations[3],  # |+4ℏkL⟩
            populations[4],  # |-4ℏkL⟩
            normalized_pol,  # Current polarization error
            normalized_mom   # Current momentum sigma
        ], dtype=np.float32)
        
        return observation
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render environment."""
        if mode == 'human':
            populations = self.physics.get_populations()
            beam_score = self.physics.get_beam_splitting_score()
            
            print(f"Step: {self.current_step}/{self.steps_per_episode}")
            print(f"Populations: {populations}")
            print(f"Beam Score: {beam_score:.4f}")
            print(f"Pol Error: {self.current_polarization_error:.3f} (init: {self.initial_polarization_error:.3f})")
            print(f"Mom Sigma: {self.current_momentum_sigma:.3f} (init: {self.initial_momentum_sigma:.3f})")
            print("-" * 40)
        
        return None
    
    def close(self):
        """Close environment."""
        pass
    
    def get_physics_info(self) -> Dict[str, Any]:
        """Get physics information."""
        return self.physics.get_physics_info()
    
    def get_scenario_info(self) -> Dict[str, Any]:
        """Get scenario information."""
        return {
            'scenario_name': self.scenario_name,
            'scenario_description': self.scenario_description,
            'momentum_p0': self.momentum_p0,
            'momentum_sigma_range': self.momentum_sigma_range,
            'polarization_range': self.polarization_range,
            'momentum_grid_pts': 121,
            'steps_per_episode': self.steps_per_episode,
            'real_time_adaptation': True,
            'parameter_drift': self.enable_parameter_drift,
            'target': '50/50 beam split'
        }


# Quick test
if __name__ == "__main__":
    print("Testing DBD Environment - Real-Time Adaptation")
    print("=" * 70)
    
    env = DBDEnv()
    print(f"Configuration:")
    print(f"  Steps per episode: {env.steps_per_episode}")
    print(f"  Momentum grid: 121 points")
    print(f"  Polarization: random {env.polarization_range}")
    print(f"  Momentum sigma: random {env.momentum_sigma_range} ℏkL")
    print(f"  Real-time adaptation: YES")
    print(f"  Parameter drift: {env.enable_parameter_drift}")
    print("=" * 70)
    
    obs, info = env.reset()
    print(f"\nInitial obs shape: {obs.shape}")
    print(f"Initial polarization: {info['initial_polarization_error']:.3f}")
    print(f"Initial momentum sigma: {info['initial_momentum_sigma']:.3f}")
    print(f"\nRunning test episode...")
    
    total_reward = 0.0
    for step in range(env.steps_per_episode):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % 5 == 0 or terminated:
            print(f"  Step {step+1:2d}: Beam={info['beam_splitting_score']:.4f}, "
                  f"Pol={info['current_polarization_error']:.3f}, "
                  f"Mom={info['current_momentum_sigma']:.3f}")
    
    print(f"\nEpisode completed!")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final beam score: {info['beam_splitting_score']:.4f}")
    print(f"Parameter changes:")
    print(f"  Pol: {info['initial_polarization_error']:.3f} → {info['current_polarization_error']:.3f}")
    print(f"  Mom: {info['initial_momentum_sigma']:.3f} → {info['current_momentum_sigma']:.3f}")