import os
import sys
import os.path
# Get the absolute path to the project root directory (parent of the script's directory)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from main import BlackjackEnv, Action
from agent.blackjack_agent import DeepQLearningAgent
import torch

# Create res directory if it doesn't exist
res_dir = os.path.join(project_root, 'agent', 'res')
os.makedirs(res_dir, exist_ok=True)

def train_agent(num_episodes=50000, parallel_games=1000):
    print("torch.cuda.is_available(): ", torch.cuda.is_available())
    print("torch.backends.mps.is_available(): ", torch.backends.mps.is_available())
    agent = DeepQLearningAgent()
    
    # Initialize metrics tracking
    all_rewards = []
    win_rates = []
    epsilons = []
    episodes_list = []
    
    # Create training log file with utf-8 encoding
    train_log = open(os.path.join(res_dir, 'training_logs.txt'), 'w', encoding='utf-8')
    train_log.write("Training Results\n")
    train_log.write("================\n\n")
    
    # Initialize vectorized environments
    envs = [BlackjackEnv() for _ in range(parallel_games)]
    states = [env.reset()[0] for env in envs]
    dones = [False] * parallel_games
    transitions = []
    
    progress_bar = tqdm(range(0, num_episodes, parallel_games), desc="Training")
    
    for episode_batch in progress_bar:
        batch_rewards = []
        
        while not all(dones):
            actions = [agent.choose_action(state) for state in states]
            for i, env in enumerate(envs):
                if dones[i]:
                    continue
                next_state, reward, done = env.step(actions[i])
                transitions.append((states[i], actions[i], reward, next_state, done))
                states[i] = next_state
                dones[i] = done
                batch_rewards.append(reward)
        
        # Store and learn from all transitions
        agent.store_parallel_transitions(transitions)
        
        # Reset for next batch
        states = [env.reset()[0] for env in envs]
        dones = [False] * parallel_games
        transitions.clear()
        
        # Update metrics
        if batch_rewards:
            current_avg_reward = np.mean(batch_rewards)
            current_win_rate = len([r for r in batch_rewards if r > 0]) / len(batch_rewards)
            
            all_rewards.append(current_avg_reward)
            win_rates.append(current_win_rate)
            epsilons.append(agent.epsilon)
            episodes_list.append(episode_batch + parallel_games)
            
            # Log episode details
            train_log.write(f"\nEpisode Batch {episode_batch}-{episode_batch + parallel_games}\n")
            train_log.write(f"Average Reward: {current_avg_reward:.3f}\n")
            train_log.write(f"Win Rate: {current_win_rate:.3f}\n")
            train_log.write(f"Epsilon: {agent.epsilon:.3f}\n")
            train_log.write("-" * 50 + "\n")
            
            progress_bar.set_postfix({
                'Avg Reward': f'{current_avg_reward:.3f}',
                'Win Rate': f'{current_win_rate:.3f}',
                'Epsilon': f'{agent.epsilon:.3f}'
            })
    
    train_log.close()
    
    # Plot training metrics
    plt.figure(figsize=(15, 5))
    
    # Plot rewards
    plt.subplot(131)
    plt.plot(episodes_list, all_rewards)
    plt.title('Average Reward over Time')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    
    # Plot win rates
    plt.subplot(132)
    plt.plot(episodes_list, win_rates)
    plt.title('Win Rate over Time')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate')
    
    # Plot epsilon decay
    plt.subplot(133)
    plt.plot(episodes_list, epsilons)
    plt.title('Epsilon Decay')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, 'training_metrics.png'))
    plt.close()
    
    print("\nSaving trained agent...")
    agent.save()
    
    return agent

def evaluate_agent(num_episodes=10000, parallel_games=100):
    """
    Evaluates the performance of a trained Deep Q-Learning agent in parallel.
    
    Parameters:
        num_episodes (int): Total number of episodes to evaluate
        parallel_games (int): Number of games to run in parallel
    """
    agent = DeepQLearningAgent()
    
    try:
        agent.load()
    except FileNotFoundError:
        print("No trained agent found!")
        return
    
    # Initialize vectorized environments
    envs = [BlackjackEnv() for _ in range(parallel_games)]
    states = [env.reset()[0] for env in envs]
    dones = [False] * parallel_games
    
    # Create evaluation log file with utf-8 encoding
    eval_log = open(os.path.join(res_dir, 'evaluation_logs.txt'), 'w', encoding='utf-8')
    eval_log.write("Evaluation Results\n")
    eval_log.write("==================\n\n")
    
    wins = 0
    total_games = 0
    all_rewards = []
    
    progress_bar = tqdm(range(0, num_episodes, parallel_games), desc="Evaluating")
    
    for _ in progress_bar:
        episode_rewards = [0] * parallel_games
        episode_actions = [[] for _ in range(parallel_games)]
        episode_states = [[] for _ in range(parallel_games)]
        
        # Run episodes until all are done
        while not all(dones):
            # Get actions for all non-done states
            actions = [
                agent.choose_action(state, training=False) 
                if not done else None 
                for state, done in zip(states, dones)
            ]
            
            # Step through all environments
            for i, (env, action) in enumerate(zip(envs, actions)):
                if dones[i]:
                    continue
                    
                episode_states[i].append(states[i])
                episode_actions[i].append(action)
                
                next_state, reward, done = env.step(action)
                episode_rewards[i] += reward
                states[i] = next_state
                dones[i] = done
        
        # Process results for all completed episodes
        for i in range(parallel_games):
            all_rewards.append(episode_rewards[i])
            
            # Count wins with double weight for double downs
            if episode_actions[i] and episode_actions[i][-1] == Action.DOUBLE:
                total_games += 2
                if episode_rewards[i] > 0:
                    wins += 2
            else:
                total_games += 1
                if episode_rewards[i] > 0:
                    wins += 1
            
            # Log each episode
            eval_log.write(f"\nEpisode {i}\n")
            eval_log.write(f"States: {episode_states[i]}\n")
            eval_log.write(f"Actions: {episode_actions[i]}\n")
            eval_log.write(f"Final Reward: {episode_rewards[i]}\n")
            eval_log.write("-" * 50 + "\n")
        
        # Reset environments for next batch
        states = [env.reset()[0] for env in envs]
        dones = [False] * parallel_games
        
        # Update progress bar with current metrics
        win_rate = wins / total_games if total_games > 0 else 0
        avg_reward = np.mean(all_rewards) if all_rewards else 0
        progress_bar.set_postfix({
            'Avg Reward': f'{avg_reward:.3f}',
            'Win Rate': f'{win_rate:.3f}'
        })
    
    # Log final statistics
    win_rate = wins / total_games
    avg_reward = np.mean(all_rewards)
    
    eval_log.write("\nSummary Statistics\n")
    eval_log.write(f"Average Reward: {avg_reward:.3f}\n")
    eval_log.write(f"Win Rate: {win_rate:.3f}\n")
    eval_log.write(f"Total Games (including double downs): {total_games}\n")
    
    eval_log.close()
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Win Rate: {win_rate:.3f}")

if __name__ == "__main__":
    # Get number of episodes from command line argument, default to 5000 if not provided
    num_episodes = 50000
    parallel_games = 1000
    if len(sys.argv) > 1:
        try:
            num_episodes = int(sys.argv[1])
            try:
                parallel_games = int(sys.argv[2])
            except ValueError:
                print(f"No specified number of parallel games, using default value of 1000.")
        except ValueError:
            print(f"No specified number of episodes, using default value of 50000.")
    
    print(f"Starting training for {num_episodes} episodes... and {parallel_games} parallel games")
    train_agent(num_episodes=num_episodes, parallel_games=parallel_games)
    print("\nEvaluating trained agent...")
    evaluate_agent()
