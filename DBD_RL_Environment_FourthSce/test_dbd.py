#!/usr/bin/env python3
"""
DBD Environment Test Script - Enhanced Scenario (Small Iterations)

Test script for DBD environment with visualization and simple testing options.
Enhanced scenario: p0=0.0, σp=0.05ℏkL, ε_pol=30%
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ensure UTF-8 console output on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dbd_env import DBDEnv


def test_simple(episodes=3):
    """Simple console testing without visualization."""
    print(f"\n[TEST] DBD Environment - Enhanced Scenario (Small Iterations)")
    print("=" * 60)

    env = DBDEnv()
    
    print(f"Configuration:")
    print(f"  Steps per episode: {env.steps_per_episode}")
    print(f"  Momentum grid points: {env.mdist.p_grid_pts}")
    print(f"  Polarization error: {env.polarization_error:.1%}")
    print(f"  Max detuning: {env.max_detuning} ω_rec")

    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        print("-" * 40)
        
        obs, info = env.reset()
        
        print(f"Starting Episode - Scenario: {info['scenario']}")
        print(f"Polarization Error: {info['polarization_error']:.3f}")
        print(f"Momentum: p0={info.get('momentum_p0', 0.0):.3f}, σp={info.get('momentum_sigma', 0.05):.3f} ℏkL")
        print("-" * 40)
        
        total_reward = 0.0
        
        for step in range(env.steps_per_episode):
            # Random action for testing
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            beam_score = info['beam_splitting_score']
            
            if step % 3 == 0 or terminated:
                print(f"    Step {step+1:2d}/{env.steps_per_episode} | Beam Score: {beam_score:.4f} | Reward: {reward:.3f}")
        
        print(f"\nEpisode Complete! Total Reward: {total_reward:.4f}")
        print(f"Final populations: {info['populations']}")
        print(f"Beam splitting score: {info.get('beam_splitting_score', 0.0):.4f}")
        print("=" * 60)
    
    print("\n✓ Simple testing completed!")


def test_visualization(episodes=2):
    """Testing with pygame visualization."""
    try:
        import pygame
    except ImportError:
        print("✗ Pygame not installed. Install with: pip install pygame")
        return
    
    print(f"\n[TEST] DBD Environment - Enhanced Scenario (Visualization)")
    print("=" * 60)
    
    # Initialize pygame
    pygame.init()
    
    # Display settings
    WIDTH, HEIGHT = 1200, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DBD Environment - Enhanced Scenario (Small Iterations)")
    
    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (0, 100, 255)
    RED = (255, 100, 100)
    GREEN = (100, 255, 100)
    YELLOW = (255, 255, 100)
    PURPLE = (200, 100, 255)
    ORANGE = (255, 150, 100)
    
    # Fonts
    font_large = pygame.font.Font(None, 36)
    font_medium = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 18)
    
    env = DBDEnv()
    
    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        
        obs, info = env.reset()
        
        # Episode data for plotting
        step_data = []
        reward_data = []
        beam_score_data = []
        population_data = []
        
        total_reward = 0.0
        
        for step in range(env.steps_per_episode):
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Random action for testing
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Store data
            step_data.append(step)
            reward_data.append(reward)
            beam_score_data.append(info['beam_splitting_score'])
            population_data.append(info['populations'].copy())
            
            # Clear screen
            screen.fill(BLACK)
            
            # Title
            title = font_large.render("DBD Environment - Enhanced Scenario (Small Iterations)", True, WHITE)
            screen.blit(title, (20, 20))
            
            # Episode info
            episode_text = font_medium.render(f"Episode: {episode + 1}/{episodes}", True, WHITE)
            screen.blit(episode_text, (20, 60))
            
            step_text = font_medium.render(f"Step: {step + 1}/{env.steps_per_episode}", True, WHITE)
            screen.blit(step_text, (20, 85))
            
            # Progress bar
            progress_pct = (step + 1) / env.steps_per_episode
            pygame.draw.rect(screen, WHITE, (20, 110, 400, 20), 2)
            pygame.draw.rect(screen, GREEN, (22, 112, int(396 * progress_pct), 16))
            
            progress_text = font_medium.render(f"Progress: {progress_pct*100:.1f}%", True, WHITE)
            screen.blit(progress_text, (20, 140))
            
            # Reward
            reward_text = font_medium.render(f"Reward: {reward:.4f}", True, YELLOW)
            screen.blit(reward_text, (20, 170))
            
            total_reward_text = font_medium.render(f"Total Reward: {total_reward:.4f}", True, YELLOW)
            screen.blit(total_reward_text, (20, 195))
            
            # Beam splitting score
            beam_score = info['beam_splitting_score']
            beam_text = font_medium.render(f"Beam Splitting Score: {beam_score:.4f}", True, GREEN)
            screen.blit(beam_text, (20, 220))
            
            # Population bars
            populations = info['populations']
            state_names = ['|0⟩', '|+2ℏkL⟩', '|-2ℏkL⟩', '|+4ℏkL⟩', '|-4ℏkL⟩']
            colors = [WHITE, BLUE, RED, PURPLE, ORANGE]
            
            y_start = 260
            for i, (name, pop, color) in enumerate(zip(state_names, populations, colors)):
                # State name
                state_text = font_small.render(f"{name}: {pop:.3f}", True, color)
                screen.blit(state_text, (20, y_start + i * 30))
                
                # Population bar
                bar_width = int(pop * 300)
                pygame.draw.rect(screen, color, (200, y_start + i * 30, bar_width, 20))
                pygame.draw.rect(screen, WHITE, (200, y_start + i * 30, 300, 20), 2)
                
                # Percentage text
                pct_text = font_small.render(f"{pop*100:.1f}%", True, WHITE)
                screen.blit(pct_text, (210, y_start + i * 30 + 2))
            
            # Efficiency metrics
            efficiency_text = font_medium.render("Efficiency Metrics:", True, WHITE)
            screen.blit(efficiency_text, (20, y_start + 180))
            
            symmetric_pop = info.get('symmetric_population', 0.0)
            sym_text = font_small.render(f"Symmetric Population: {symmetric_pop:.4f}", True, GREEN)
            screen.blit(sym_text, (20, y_start + 205))
            
            # Scenario info
            scenario_text = font_medium.render("Scenario Info:", True, WHITE)
            screen.blit(scenario_text, (20, y_start + 240))
            
            pol_error = info.get('polarization_error', 0.0)
            pol_text = font_small.render(f"Polarization Error: {pol_error:.3f}", True, YELLOW)
            screen.blit(pol_text, (20, y_start + 265))
            
            momentum_p0 = info.get('momentum_p0', 0.0)
            momentum_sigma = info.get('momentum_sigma', 0.05)
            mom_text = font_small.render(f"Momentum: p0={momentum_p0:.3f}, σp={momentum_sigma:.3f} ℏkL", True, YELLOW)
            screen.blit(mom_text, (20, y_start + 285))
            
            steps_text = font_small.render(f"Steps per episode: {env.steps_per_episode}", True, YELLOW)
            screen.blit(steps_text, (20, y_start + 305))
            
            # Update display
            pygame.display.flip()
            
            # Control frame rate
            pygame.time.wait(100)
        
        print(f"Episode {episode + 1} completed!")
        print(f"Total reward: {total_reward:.4f}")
        print(f"Final beam splitting score: {beam_score:.4f}")
        print(f"Final populations: {populations}")
    
    # Keep window open for a moment
    pygame.time.wait(2000)
    pygame.quit()
    
    print("\n✓ Visualization testing completed!")


def main():
    """Main test function."""
    print("=" * 60)
    print("DBD Environment Test Script - Enhanced Scenario")
    print("Small Iterations Configuration")
    print("=" * 60)
    
    while True:
        print("\nSelect test mode:")
        print("1. Simple console testing (quick, shows step-by-step progress)")
        print("2. Visualization testing (pygame, graphical interface)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            episodes = input("Number of episodes to test (default 3): ").strip()
            episodes = int(episodes) if episodes else 3
            test_simple(episodes)
        elif choice == '2':
            episodes = input("Number of episodes to test (default 2): ").strip()
            episodes = int(episodes) if episodes else 2
            test_visualization(episodes)
        elif choice == '3':
            print("\n✓ Goodbye!")
            return
        else:
            print("✗ Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()