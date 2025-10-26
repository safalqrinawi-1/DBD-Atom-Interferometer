#!/usr/bin/env python3
"""
DBD Environment Test Script

Unified test script for DBD environment with visualization and simple testing options.
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dbd_env import DBDEnv

def test_simple(episodes=5, polarization_error=0.0):
    """Simple console testing without visualization."""
    print(f"\n🎯 Testing DBD Environment (Simple Mode - 30 Steps per Episode)")
    print(f"Polarization Error: {polarization_error}")
    print("=" * 60)

    env = DBDEnv(polarization_error=polarization_error)

    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        print("-" * 40)

        # Reset environment
        obs, info = env.reset()
        print(f"Initial observation: {obs}")

        # Run steps_per_episode steps
        for step in range(env.steps_per_episode):
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Print detailed info for EVERY step
            populations = info['populations']
            symmetric_pop = info['symmetric_population']
            print(f"Step {step+1:2d} | Action: {action[0]:.4f} | Reward: {reward:.4f} | "
                  f"Populations: |0⟩={populations[0]:.3f} |+2⟩={populations[1]:.3f} |-2⟩={populations[2]:.3f} "
                  f"|+4⟩={populations[3]:.3f} |-4⟩={populations[4]:.3f}")

            # Show cumulative progress every 5 steps
            if step > 0 and step % 5 == 0:
                print(f"    📊 Progress: Step {step+1}/{env.steps_per_episode} | Symmetric Pop: {symmetric_pop:.4f}")

        # Show final episode summary
        print("\n" + "="*60)
        print("🎯 Episode Summary:")
        print(f"Episode reward: {info['episode_reward']:.4f}")
        print(f"Final populations: {info['populations']}")
        print(f"Symmetric population: {info['symmetric_population']:.4f}")
        print(f"Episode completed: {terminated}")
        print("="*60)

    env.close()
    print("\n✅ Simple testing completed!")
    print("=" * 60)

def test_with_visualization(episodes=5, polarization_error=0.0):
    """Test with pygame visualization."""
    try:
        import pygame
        pygame.init()
    except ImportError:
        print("❌ Pygame not installed. Install with: pip install pygame")
        return

    print(f"\n🎮 Testing DBD Environment (Visualization Mode - 30 Steps per Episode)")
    print(f"Polarization Error: {polarization_error}")
    print("=" * 70)

    # Enhanced pygame visualizer with detailed information
    class SimpleVisualizer:
        def __init__(self, width=800, height=600):
            self.width = width
            self.height = height
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("DBD Environment - Real-time Visualization")
            self.font_small = pygame.font.Font(None, 20)
            self.font_medium = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 32)
            self.BLACK = (0, 0, 0)
            self.WHITE = (255, 255, 255)
            self.GREEN = (100, 255, 100)
            self.BLUE = (100, 100, 255)
            self.RED = (255, 100, 100)
            self.YELLOW = (255, 255, 100)
            self.GRAY = (150, 150, 150)

        def update(self, populations, reward, step_count, steps_per_episode):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return False

            self.screen.fill(self.BLACK)

            # Title
            title = self.font_large.render("DBD Environment - Detuning Optimization", True, self.WHITE)
            self.screen.blit(title, (20, 20))

            # Episode progress
            progress_pct = (step_count / steps_per_episode) * 100
            progress_text = self.font_medium.render(".1f", True, self.WHITE)
            self.screen.blit(progress_text, (20, 60))

            # Step and reward info
            step_text = self.font_medium.render(f"Step: {step_count}/{steps_per_episode}", True, self.GREEN)
            reward_text = self.font_medium.render(".4f", True, self.YELLOW)
            self.screen.blit(step_text, (20, 90))
            self.screen.blit(reward_text, (250, 90))

            # Population header
            pop_header = self.font_medium.render("Quantum State Populations:", True, self.BLUE)
            self.screen.blit(pop_header, (20, 130))

            # Individual state populations with bars
            state_names = ['|0⟩ Ground', '|+2ℏkL⟩ Sym+', '|-2ℏkL⟩ Sym-', '|+4ℏkL⟩ Higher+', '|-4ℏkL⟩ Higher-']
            colors = [self.WHITE, self.GREEN, self.GREEN, self.RED, self.RED]

            y_pos = 160
            bar_width = 300
            bar_height = 25

            for i, (name, pop, color) in enumerate(zip(state_names, populations, colors)):
                # State name and percentage
                state_text = self.font_small.render(".1f", True, color)
                self.screen.blit(state_text, (20, y_pos))

                # Population bar
                bar_x = 200
                bar_fill = int(pop * bar_width)
                pygame.draw.rect(self.screen, self.GRAY, (bar_x, y_pos, bar_width, bar_height), 2)  # Outline
                pygame.draw.rect(self.screen, color, (bar_x, y_pos, bar_fill, bar_height))  # Fill

                # Percentage on bar
                if pop > 0.1:  # Only show if visible
                    pct_text = self.font_small.render(".1f", True, self.BLACK)
                    self.screen.blit(pct_text, (bar_x + bar_fill - 30, y_pos + 5))

                y_pos += 35

            # DBD Efficiency (symmetric population)
            symmetric_pop = populations[1] + populations[2]
            efficiency_y = y_pos + 20
            efficiency_text = self.font_medium.render(".4f", True, self.YELLOW)
            self.screen.blit(efficiency_text, (20, efficiency_y))

            # Progress bar for episode
            progress_y = efficiency_y + 40
            progress_width = self.width - 40
            progress_fill = int((step_count / steps_per_episode) * progress_width)
            pygame.draw.rect(self.screen, self.GRAY, (20, progress_y, progress_width, 20), 2)
            pygame.draw.rect(self.screen, self.BLUE, (20, progress_y, progress_fill, 20))

            # Instructions
            instructions = self.font_small.render("Press ESC or close window to exit", True, self.WHITE)
            self.screen.blit(instructions, (20, self.height - 30))

            pygame.display.flip()
            return True

        def reset(self):
            pass

        def close(self):
            pygame.quit()

    visualizer = SimpleVisualizer()

    # Create environment
    env = DBDEnv(polarization_error=polarization_error)

    try:
        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}/{episodes}")
            print("-" * 50)

            # Reset environment
            obs, info = env.reset()
            visualizer.reset()

            # Run steps_per_episode steps
            for step in range(env.steps_per_episode):
                # Take random action
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                # Print detailed info for EVERY step
                populations = info['populations']
                symmetric_pop = info['symmetric_population']
                print(f"Step {step+1:2d} | Action: {action[0]:.4f} | Reward: {reward:.4f} | "
                      f"Populations: |0⟩={populations[0]:.3f} |+2⟩={populations[1]:.3f} |-2⟩={populations[2]:.3f} "
                      f"|+4⟩={populations[3]:.3f} |-4⟩={populations[4]:.3f}")

                # Show cumulative progress every 5 steps
                if step > 0 and step % 5 == 0:
                    print(f"    📊 Progress: Step {step+1}/{env.steps_per_episode} | Symmetric Pop: {symmetric_pop:.4f}")

                # Update visualization
                if not visualizer.update(
                    populations=np.array(info['populations']),
                    reward=reward,
                    step_count=info['step_count'],
                    steps_per_episode=env.steps_per_episode
                ):
                    print("Visualization closed by user")
                    return

                # Small delay for visualization
                pygame.time.delay(100)

            # Final episode summary
            print("\n" + "="*60)
            print("🎯 Episode Summary:")
            print(f"Episode reward: {info['episode_reward']:.4f}")
            print(f"Final populations: {info['populations']}")
            print(f"Symmetric population: {info['symmetric_population']:.4f}")
            print("="*60)

            # Delay between episodes
            pygame.time.delay(1000)

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        visualizer.close()
        env.close()

    print("\n✅ Visualization testing completed!")

def main():
    """Main function with user interaction."""
    print("🎯 DBD Environment Test Script")
    print("=" * 40)

    # Ask for test mode
    while True:
        print("\nSelect test mode:")
        print("1. Simple console testing (shows step-by-step rewards & populations)")
        print("2. Visualization testing (pygame graphs + detailed console info)")

        try:
            mode_choice = input("Enter choice (1 or 2): ").strip()

            if mode_choice == "1":
                test_mode = "simple"
                break
            elif mode_choice == "2":
                test_mode = "visualization"
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return

    # Ask for number of episodes
    while True:
        try:
            episodes_input = input("Enter number of episodes to test (default 5): ").strip()
            if episodes_input == "":
                episodes = 5
            else:
                episodes = int(episodes_input)
                if episodes < 1:
                    raise ValueError("Must be positive")
            break
        except ValueError:
            print("❌ Please enter a valid positive number.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return

    # Ask for polarization error
    while True:
        try:
            pol_error_input = input("Enter polarization error [0.0-0.1] (default 0.0): ").strip()
            if pol_error_input == "":
                polarization_error = 0.0
            else:
                polarization_error = float(pol_error_input)
                if not (0.0 <= polarization_error <= 0.1):
                    raise ValueError("Must be between 0.0 and 0.1")
            break
        except ValueError as e:
            print(f"❌ {e}. Please enter a value between 0.0 and 0.1.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return

    # Explain episode structure
    print("\n📊 Episode Information:")
    print("   • Each episode = 30 steps (divided pulse duration)")
    print("   • Each step: single detuning value Δ(t) for that time segment")
    print("   • Reward = DENSE (incremental improvements + final bonus)")
    print("   • IMPORTANT: Each step will show:")
    print("     - Step number, Action taken, Reward received")
    print("     - All 5 population values (|0⟩, |+2⟩, |-2⟩, |+4⟩, |-4⟩)")
    print("     - Current time progress")
    print("   • Symmetric population = (|+2ℏkL⟩ + |-2ℏkL⟩) at current step")
    print("   • Reward Calculation:")
    print("     - Each step: P_sym(current) - P_sym(previous)")
    print("     - Step 30: Above + bonus = P_sym(final)")
    print("   • Goal: Maximize cumulative reward (incremental + final bonus)")
    print("   • Learning Challenge: Improve symmetric population at each step")

    # Run selected test
    if test_mode == "simple":
        test_simple(episodes, polarization_error)
    else:
        test_with_visualization(episodes, polarization_error)

if __name__ == "__main__":
    main()
