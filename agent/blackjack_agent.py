import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from main import Action
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pickle

class QNetwork(nn.Module):
    """
        QNetwork

        A neural network model class inheriting from nn.Module for Q-learning applications.

        Methods
        -------
        __init__(input_size, hidden_size, output_size):
            Initializes the neural network layers and their respective weights and biases.

        forward(x):
            Defines the forward pass of the network.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        Applies the forward pass of the neural network.

        Args:
            x: Input tensor to the neural network.

        Returns:
            The output tensor after passing through the neural network.
        """
        return self.network(x)

class DeepQLearningAgent:
    """
    DeepQLearningAgent class implements a Deep Q-Learning agent for Blackjack game.

    __init__(self, input_size=6, hidden_size=128, learning_rate=0.0003, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995):
        Initializes the DeepQLearningAgent.
        - input_size: Input size for the neural network.
        - hidden_size: Number of hidden units in the neural network.
        - learning_rate: Learning rate for the optimizer.
        - gamma: Discount factor for future rewards.
        - epsilon_start: Initial value for epsilon in epsilon-greedy policy.
        - epsilon_end: Minimum value for epsilon.
        - epsilon_decay: Decay rate for epsilon.

    state_to_tensor(self, state):
        Converts the game state dictionary to a normalized tensor suitable for input into the neural network.
        - state: Dictionary containing the current game state.

    choose_action(self, state, training=True):
        Chooses an action based on the current state using epsilon-greedy policy.
        - state: The current state of the environment.
        - training: Boolean indicating whether the agent is in training mode.

    store_transition(self, state, action, reward, next_state, done):
        Stores a transition in the experience replay memory.
        - state: The current state.
        - action: The action taken.
        - reward: The reward received.
        - next_state: The next state observed.
        - done: Boolean indicating whether the episode has ended.

    train(self):
        Samples a batch from experience replay memory and performs a training step to update the Q-network.

    save(self, filename='blackjack_agent.pkl'):
        Saves the state of the agent, including neural network parameters and optimizer state, to a file.
        - filename: The path to the file where the agent's state will be saved.

    load(self, filename='blackjack_agent.pkl'):
        Loads the agent's state from a file, restoring neural network parameters and optimizer state.
        - filename: The path to the file from which the agent's state will be loaded.
    """
    def __init__(self, input_size=6, hidden_size=128, learning_rate=0.0003, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural Network
        self.q_network = QNetwork(input_size, hidden_size, len(Action)).to(self.device)
        self.target_network = QNetwork(input_size, hidden_size, len(Action)).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience Replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

    def state_to_tensor(self, state):
        """
        Converts the game state into a tensor representation, incorporating original and newly computed features.

        Parameters:
            state (dict): A dictionary representing the current game state with keys:
                          - 'player_value': Current total value of the player's hand.
                          - 'dealer_visible': Dealer's visible card.
                          - 'player_hand': A list of cards in the player's hand.

        Returns:
            torch.FloatTensor: A tensor representation of the state, including normalized values and additional computed features.
        """
        # Enhanced state representation
        player_value = state['player_value']
        dealer_card_value = state['dealer_visible'].get_value()
        num_cards = len(state['player_hand'])
        has_usable_ace = 1 if any(card.value == 'A' for card in state['player_hand']) else 0
        
        # New features
        bust_probability = max(0, (21 - player_value) / 21)  # Probability of busting
        dealer_bust_probability = max(0, (21 - dealer_card_value) / 21)  # Dealer bust probability
        
        state_array = np.array([
            player_value / 21,  # Normalize values
            dealer_card_value / 13,
            num_cards / 10,
            has_usable_ace,
            bust_probability,
            dealer_bust_probability
        ], dtype=np.float32)
        
        return torch.FloatTensor(state_array).to(self.device)

    def choose_action(self, state, training=True):
        """
        Selects an action based on the given state.

        If in training mode and a random number is less than epsilon, a random action is selected.
        Otherwise, the action with the highest Q-value as predicted by the Q-network is chosen.

        Parameters:
            state: The current state from which to choose an action.
            training: A boolean indicating whether the agent is in training mode (default is True).

        Returns:
            An action, either random or the one with the highest predicted Q-value.
        """
        if training and random.random() < self.epsilon:
            return random.choice(list(Action))
        
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
            return Action(action_idx)

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition in the memory buffer.

        Args:
            state: The current state of the environment.
            action: The action taken by the agent.
            reward: The reward received after taking the action.
            next_state: The subsequent state of the environment after taking the action.
            done: A boolean flag indicating whether the episode has terminated.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """
        Trains the DQN model by sampling a batch of experiences from memory,
        computing the loss between the predicted Q values and the target Q values,
        and updating the model weights.

        Steps:
        1. Return if there are not enough experiences in memory to form a batch.
        2. Sample a random batch of experiences from memory.
        3. Convert the batch data into tensors for states, actions, rewards, next states, and done flags.
        4. Compute the current Q values for the batch states and actions using the Q network.
        5. Compute the next Q values for the batch next states using the target network, then derive the target Q values.
        6. Compute the loss between the current Q values and target Q values.
        7. Backpropagate the loss and update the Q network's weights.
        8. Update the exploration epsilon value.
        9. Occasionally update the target network with the Q network's weights.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare tensors - extract state from the tuple (state, reward, done)
        state_tensors = torch.stack([self.state_to_tensor(state) for state, _, _, _, _ in batch])
        action_tensors = torch.tensor([action.value for _, action, _, _, _ in batch], device=self.device)
        reward_tensors = torch.tensor([reward for _, _, reward, _, _ in batch], dtype=torch.float32, device=self.device)
        next_state_tensors = torch.stack([self.state_to_tensor(next_state) for _, _, _, next_state, _ in batch])
        done_tensors = torch.tensor([done for _, _, _, _, done in batch], dtype=torch.float32, device=self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(state_tensors)
        current_q_values = current_q_values.gather(1, action_tensors.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensors)
            max_next_q = next_q_values.max(1)[0]
            target_q_values = reward_tensors + (1 - done_tensors) * self.gamma * max_next_q
        
        # Compute loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Periodically update target network
        if random.random() < 0.01:  # 1% chance each training step
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filename='blackjack_agent.pkl'):
        """
        Saves the current state of the BlackjackAgent to a file.

        Parameters:
        filename (str): The name of the file where the state will be saved. Default is 'blackjack_agent.pkl'.

        The saved state includes:
        - q_network state dictionary
        - target_network state dictionary
        - optimizer state dictionary
        - epsilon value
        """
        state_dict = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        with open(filename, 'wb') as f:
            pickle.dump(state_dict, f)

    def load(self, filename='blackjack_agent.pkl'):
        """

        Loads the state of the agent from a file.

        Parameters:
        filename (str): The name of the file from which to load the agent's state. Default is 'blackjack_agent.pkl'.

        The file should contain a dictionary with the following keys:
        - 'q_network': State dictionary for the Q-network.
        - 'target_network': State dictionary for the target network.
        - 'optimizer': State dictionary for the optimizer.
        - 'epsilon': The value of epsilon for the epsilon-greedy policy.

        Raises:
        FileNotFoundError: If the specified file does not exist.
        """
        with open(filename, 'rb') as f:
            state_dict = pickle.load(f)
        
        self.q_network.load_state_dict(state_dict['q_network'])
        self.target_network.load_state_dict(state_dict['target_network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.epsilon = state_dict['epsilon']
