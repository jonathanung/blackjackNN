
# BlackjackNN - Deep Q-Learning Blackjack Agent

A neural network-based Blackjack AI that learns optimal playing strategies through deep Q-learning. The project includes both training capabilities and interactive GUI interfaces for playing against or getting advice from the trained agent.

## Features

- Deep Q-Learning implementation for Blackjack strategy learning
- Interactive Blackjack GUI for playing games
- Decision Helper GUI for getting real-time advice
- Detailed training metrics and logging
- Basic and advanced reward shaping based on optimal Blackjack strategy

## Project Structure

- `main.py` - Core Blackjack game environment and logic
- `blackjack_agent.py` - Deep Q-Learning agent implementation
- `train_agent.py` - Training script with metrics logging
- `blackjack_gui.py` - Interactive Blackjack game interface
- `decision_helper_gui.py` - Real-time strategy advice interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jonathanung/blackjackNN.git
cd blackjackNN
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Agent

```bash
python ./agent/train_agent.py [num_episodes]
```

This will:
- Train the agent for the specified number of episodes (default is 5000)
- Generate training metrics plots
- Save the trained model as 'blackjack_agent.pkl'
- Create detailed training logs

If you want to train the agent on a GPU or MPS, you can uncomment the relevant lines in `blackjack_agent.py` under the `__init__` method.

### Playing Blackjack

```bash
python ./gui/blackjack_gui.py
```

Features:
- Play Blackjack with a graphical interface
- Get real-time suggestions from the trained agent
- View agent's confidence in its suggestions
- Color-coded advice based on confidence levels

### Getting Strategy Advice

```bash
python ./gui/decision_helper_gui.py
```

This will open a GUI interface for getting real-time strategy advice from the trained agent.


Input your:
- Current hand value
- Number of cards
- Dealer's visible card

Get:
- Recommended action
- Confidence levels
- Q-values for all possible actions

## Training Metrics

The training process generates:
- Win rate over time
- Average reward progression
- Detailed state-action logs
- Agent's exploration rate (epsilon) decay


## Acknowledgments

- Built using PyTorch for deep learning
- Implements standard Blackjack rules and optimal strategy rewards
- Uses epsilon-greedy exploration strategy

### Future Features
- More sophisticated reward shaping
- Train with multiple agents in parallel to simulate multiplayer Blackjack
- Implement agents in multiplayer Blackjack game, allowing players to choose their position on the table


### Data over 10M episodes in 750 size batches
![Training Metrics](agent/res/training_metrics.png)