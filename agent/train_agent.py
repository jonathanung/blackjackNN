import os
import sys
import os.path
# Get the absolute path to the project root directory (parent of the script's directory)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from main import BlackjackEnv
from agent.blackjack_agent import DeepQLearningAgent

# Create res directory if it doesn't exist
res_dir = os.path.join(project_root, 'agent', 'res')
os.makedirs(res_dir, exist_ok=True)

def train_agent(num_episodes=5000, eval_interval=1000):
    """
        Trains a Deep Q-Learning agent on the Blackjack environment for a specified number of episodes.

        Parameters:
        num_episodes (int): The number of episodes to train the agent. Default is 500,000.
        eval_interval (int): The frequency at which to log training details. Default is 1,000.

        Variables:
        env (BlackjackEnv): Instance of the Blackjack environment
        agent (DeepQLearningAgent): Instance of the Deep Q-Learning agent
        all_rewards (list): List to keep track of rewards for each episode
        cumulative_rewards (list): List to keep track of cumulative average rewards over episodes
        win_rates (list): List to keep track of win rates over episodes
        log_file (file object): File object for logging training details

        Process:
        - Initializes the environment and agent
        - Initializes metrics lists for rewards, cumulative rewards, and win rates
        - Opens a log file to write training details
        - Runs the training loop over the specified number of episodes
        - In each episode, resets the environment and tracks states, actions, and rewards
        - Tracks cumulative average rewards and win rates for logging and progress display
        - Logs details every 100 episodes
        - Updates a progress bar with the current average reward, win rate, and agent's epsilon value
        - Closes the log file after training is complete
        - Plots and saves training metrics as a PNG image
    """
    env = BlackjackEnv()
    agent = DeepQLearningAgent()
    
    # Training metrics
    all_rewards = []
    cumulative_rewards = []
    win_rates = []
    
    # Create log file
    log_file = open(os.path.join(res_dir, 'training_logs.txt'), 'w')
    log_file.write("Training Log\n")
    log_file.write("============\n\n")
    
    # Training loop
    progress_bar = tqdm(range(num_episodes), desc="Training")
    for episode in progress_bar:
        state, _, _ = env.reset()
        episode_reward = 0
        episode_actions = []
        episode_states = []
        done = False
        
        while not done:
            action = agent.choose_action(state)
            episode_states.append(state)
            episode_actions.append(action)
            
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            episode_reward += reward
        
        all_rewards.append(episode_reward)
        
        # Calculate cumulative metrics
        current_avg_reward = np.mean(all_rewards)
        cumulative_rewards.append(current_avg_reward)
        current_win_rate = len([r for r in all_rewards if r > 0]) / len(all_rewards)
        win_rates.append(current_win_rate)
        
        # Log episode details
        log_file.write(f"\nEpisode {episode}\n")
        log_file.write(f"States: {episode_states}\n")
        log_file.write(f"Actions: {episode_actions}\n")
        log_file.write(f"Reward: {episode_reward}\n")
        log_file.write(f"Cumulative Avg Reward: {current_avg_reward:.3f}\n")
        log_file.write(f"Cumulative Win Rate: {current_win_rate:.3f}\n")
        log_file.write("-" * 50 + "\n")
        
        # Update progress bar
        progress_bar.set_postfix({
            'Avg Reward': f'{current_avg_reward:.3f}',
            'Win Rate': f'{current_win_rate:.3f}',
            'Epsilon': f'{agent.epsilon:.3f}'
        })
    
    log_file.close()
    
    # Plot training metrics
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(num_episodes), cumulative_rewards)
    plt.title('Cumulative Average Reward over Training')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.ylim(-2, 2)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(num_episodes), win_rates)
    plt.title('Cumulative Win Rate over Training')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, 'training_metrics.png'))
    plt.close()
    
    print("\nSaving trained agent...")
    agent.save()
    
    return agent

def evaluate_agent(num_episodes=1000):
    """
    Evaluates the performance of a trained Deep Q-Learning agent in a Blackjack environment over a specified number of episodes.

    Parameters:
        num_episodes (int): The number of episodes to run for evaluation (default is 1000).

    Returns:
        None

    Creates an evaluation log file 'evaluation_log.txt' and writes the following information:
        - Detailed state-action-reward sequence for each episode.
        - Summary statistics including:
          - Average reward per episode.
          - Win rate (proportion of positive rewards).

    Notes:
        - The agent is loaded from a pre-trained state. If no such state is found, the function exits with a message.
        - Utilizes tqdm for a progress bar representation during evaluation.
    """
    env = BlackjackEnv()
    agent = DeepQLearningAgent()
    
    try:
        agent.load()
    except FileNotFoundError:
        print("No trained agent found!")
        return
    
    # Create evaluation log file
    eval_log = open(os.path.join(res_dir, 'evaluation_logs.txt'), 'w')
    eval_log.write("Evaluation Results\n")
    eval_log.write("==================\n\n")
    
    rewards = []
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        state, _, _ = env.reset()
        done = False
        episode_reward = 0
        episode_actions = []
        episode_states = []
        
        while not done:
            action = agent.choose_action(state, training=False)
            episode_states.append(state)
            episode_actions.append(action)
            
            state, reward, done = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
        
        # Log each episode
        eval_log.write(f"\nEpisode {episode}\n")
        eval_log.write(f"States: {episode_states}\n")
        eval_log.write(f"Actions: {episode_actions}\n")
        eval_log.write(f"Final Reward: {episode_reward}\n")
        eval_log.write("-" * 50 + "\n")
    
    win_rate = len([r for r in rewards if r > 0]) / num_episodes
    avg_reward = np.mean(rewards)
    
    # Log summary statistics
    eval_log.write("\nSummary Statistics\n")
    eval_log.write(f"Average Reward: {avg_reward:.3f}\n")
    eval_log.write(f"Win Rate: {win_rate:.3f}\n")
    
    eval_log.close()
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Win Rate: {win_rate:.3f}")

if __name__ == "__main__":
    # Get number of episodes from command line argument, default to 5000 if not provided
    num_episodes = 5000
    if len(sys.argv) > 1:
        try:
            num_episodes = int(sys.argv[1])
        except ValueError:
            print(f"No specified number of episodes, using default value of 5000.")
    
    print(f"Starting training for {num_episodes} episodes...")
    train_agent(num_episodes=num_episodes)
    print("\nEvaluating trained agent...")
    evaluate_agent()
