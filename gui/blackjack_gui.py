import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import tkinter as tk
from tkinter import messagebox
from main import BlackjackEnv, Action
from agent.blackjack_agent import DeepQLearningAgent
import torch

class BlackjackGUI:
    """
    BlackjackGUI

    A graphical user interface for playing Blackjack, integrated with a deep Q-learning agent for suggesting optimal moves. Utilizes Tkinter for the GUI components and provides options for playing a new game, hitting, and standing. The GUI updates to show the dealer's and player's hands, and displays the agent's suggestions along with its confidence level.

    Methods:

    __init__(self, root)
        Initialize the BlackjackGUI with the given root window.
        Load a trained Deep Q-learning agent, if available.
        Set up the GUI frames, labels, and buttons.
        Start a new game.

    update_display(self, show_all=False)
        Update the display of the dealer's and player's hands.
        Optionally reveal all dealer cards if the game is over.
        Update the agent's suggestion if the game is ongoing.

    update_suggestion(self)
        Update the agent's suggestion based on the current game state.
        Display the suggested action and the confidence of the agent.
        Color code the suggestion text based on confidence level.

    hit(self)
        Execute the hit action and update the display.
        Check if the game is over and handle the game over scenario.

    stand(self)
        Execute the stand action, reveal all dealer cards, and update the display.
        Handle the game over scenario.

    game_over(self, reward)
        Disable all action buttons and display a game over message.
        Show a popup message indicating the game result (win, loss, tie).

    new_game(self)
        Start a new game by resetting the environment.
        Enable all action buttons.
        Update the display with the initial state.

    play_action(self, action)
        Play the specified action and disable the double button after the first action.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Blackjack")
        self.env = BlackjackEnv()
        
        # Load the trained agent
        self.agent = DeepQLearningAgent()
        try:
            agent_path = os.path.join(project_root, 'agent', 'res', 'blackjack_agent.pkl')
            self.agent.load(agent_path)
            print("Loaded trained agent successfully")
        except FileNotFoundError:
            print("No trained agent found, using untrained agent")
        
        # Create frames
        self.dealer_frame = tk.Frame(root)
        self.player_frame = tk.Frame(root)
        self.button_frame = tk.Frame(root)
        
        self.dealer_frame.pack(pady=20)
        self.player_frame.pack(pady=20)
        self.button_frame.pack(pady=20)
        
        # Labels
        self.dealer_label = tk.Label(self.dealer_frame, text="Dealer's Hand:", font=('Arial', 12))
        self.dealer_cards = tk.Label(self.dealer_frame, text="", font=('Arial', 12))
        self.dealer_label.pack()
        self.dealer_cards.pack()
        
        self.player_label = tk.Label(self.player_frame, text="Your Hand:", font=('Arial', 12))
        self.player_cards = tk.Label(self.player_frame, text="", font=('Arial', 12))
        self.player_value = tk.Label(self.player_frame, text="", font=('Arial', 12))
        self.player_label.pack()
        self.player_cards.pack()
        self.player_value.pack()
        
        # Add suggestion label and confidence display
        self.suggestion_frame = tk.Frame(self.button_frame)
        self.suggestion_frame.pack(pady=10)
        
        self.suggestion_label = tk.Label(self.suggestion_frame, 
                                       text="Agent suggests: ", 
                                       font=('Arial', 12))
        self.suggestion_label.pack(side=tk.LEFT)
        
        self.confidence_label = tk.Label(self.suggestion_frame, 
                                       text="Confidence: ", 
                                       font=('Arial', 12))
        self.confidence_label.pack(side=tk.LEFT)
        
        # Modify button creation to store references
        self.buttons = {
            Action.HIT: tk.Button(self.button_frame, text="Hit", command=self.hit),
            Action.STAND: tk.Button(self.button_frame, text="Stand", command=self.stand),
            Action.DOUBLE: tk.Button(self.button_frame, text="Double", command=self.double)
        }
        
        self.buttons[Action.HIT].pack(side=tk.LEFT, padx=5)
        self.buttons[Action.STAND].pack(side=tk.LEFT, padx=5)
        self.buttons[Action.DOUBLE].pack(side=tk.LEFT, padx=5)
        self.new_game_button = tk.Button(self.button_frame, text="New Game", command=self.new_game)
        self.new_game_button.pack(side=tk.LEFT, padx=5)
        
        self.new_game()
    
    def update_display(self, show_all=False):
        """
        Updates the display of the game interface, including dealer's cards, player's cards,
        and agent's suggestion. The display varies based on the state of the game.

        Parameters:
        show_all (bool): If True, show all of the dealer's cards. Otherwise, hide the
                         second dealer card if the game is ongoing.
        """
        # Show dealer's cards (hide second card if game is ongoing)
        dealer_cards = self.env.dealer_hand.cards
        if show_all:
            dealer_text = " ".join(str(card) for card in dealer_cards)
            dealer_value = f"Value: {self.env.dealer_hand.get_value()}"
        else:
            dealer_text = f"{dealer_cards[0]} ??"
            dealer_value = ""
        self.dealer_cards.config(text=f"{dealer_text}\n{dealer_value}")
        
        # Show player's cards
        player_cards = self.env.player_hand.cards
        player_text = " ".join(str(card) for card in player_cards)
        player_value = f"Value: {self.env.player_hand.get_value()}"
        self.player_cards.config(text=player_text)
        self.player_value.config(text=player_value)
        
        # Update agent suggestion
        if not show_all:  # Only show suggestion during active game
            self.update_suggestion()
    
    def update_suggestion(self):
        """
        Update the suggestion label with the agent's recommended action and its confidence.

        Retrieves the current state from the environment and converts it into a tensor representation.
        Fetches the Q-values of possible actions using the agent's Q-network.
        Determines the best action based on the highest Q-value and computes the softmax probabilities.
        Sets the suggestion text and displays the confidence level for the recommended action.
        Applies color coding to the text label based on the confidence level:
            - Green for confidence > 80%
            - Orange for confidence > 60%
            - Red for confidence <= 60%
        """
        state = self.env.get_state()
        
        # Get Q-values and action from agent
        with torch.no_grad():
            state_tensor = self.agent.state_to_tensor(state)
            q_values = self.agent.q_network(state_tensor)
            
            # Get best action and its probability
            action_probs = torch.softmax(q_values, dim=0)
            best_action_idx = q_values.argmax().item()
            confidence = action_probs[best_action_idx].item()
            
            action = Action(best_action_idx)
        
        # Update suggestion labels
        suggestion_text = f"Agent suggests: {action.name}"
        confidence_text = f"Confidence: {confidence:.2%}"
        
        self.suggestion_label.config(text=suggestion_text)
        self.confidence_label.config(text=confidence_text)
        
        # Color code based on confidence
        if confidence > 0.8:
            self.suggestion_label.config(fg="green")
        elif confidence > 0.6:
            self.suggestion_label.config(fg="orange")
        else:
            self.suggestion_label.config(fg="red")
    
    def hit(self):
        """
        Executes the "Hit" action within the game's environment.

        This method triggers the HIT action, updates the display, and checks if the game
        has ended. If the game is over, it handles the end-of-game scenario.

        state - The current state of the game after taking the HIT action.
        reward - The reward received after executing the HIT action.
        done - A boolean indicating if the game has ended after executing the HIT action.

        self.env.step(Action.HIT)
            Performs the HIT action within the current game environment and returns the new state, reward, and done flag.

        self.update_display()
            Updates the game's display to reflect the current state after executing the HIT action.

        self.game_over(reward)
            Handles the end-of-game logic, if the game has ended after executing the HIT action.
        """
        state, reward, done = self.env.step(Action.HIT)
        self.update_display()
        
        if done:
            self.game_over(reward)
    
    def stand(self):
        """
        Executes the 'STAND' action in the environment

        Performs the following steps:
        - Executes the 'STAND' action in the current environment state and receives the resulting state, reward, and done flag.
        - Updates the display with the new state, showing all relevant information.
        - Checks if the game is over by analyzing the reward.

        Args:
            None

        Returns:
            None
        """
        state, reward, done = self.env.step(Action.STAND)
        self.update_display(show_all=True)
        self.game_over(reward)
    
    def game_over(self, reward):
        """
        Handles the game over logic by resetting button states and displaying results.

        Args:
            reward (int): The reward value indicating the game's outcome.
                          Positive for a win, negative for a loss, and zero for a tie.

        Actions:
            - Resets all button colors to their default state.
            - Disables all buttons to prevent further interaction.
            - Updates the suggestion label to indicate that the game is over.
            - Displays a message box indicating whether the user has won, lost, or tied.
        """
        # Reset button colors
        for button in self.buttons.values():
            button.config(bg='SystemButtonFace')
            button.config(state=tk.DISABLED)
        
        self.suggestion_label.config(text="Game Over")
        
        if reward > 0:
            messagebox.showinfo("Game Over", "You Win!")
        elif reward < 0:
            messagebox.showinfo("Game Over", "You Lose!")
        else:
            messagebox.showinfo("Game Over", "It's a Tie!")
    
    def new_game(self):
        """

        Starts a new game by resetting the environment, enabling all buttons,
        and updating the display.

        - Resets the game environment to its initial state.
        - Enables all buttons to their normal state, allowing user interaction.
        - Updates the display to reflect the new game state.
        """
        self.env.reset()
        for button in self.buttons.values():
            button.config(state=tk.NORMAL)
        self.update_display()
    
    def play_action(self, action):
        """
        play_action(action)

        Disables the 'double' button if the player has more than two cards in their hand after the first action.

        Parameters:
        - action: The action performed that may affect the player's hand.

        Behavior:
        - Checks if the player's hand has more than two cards. If true, disables the double button configuration to prevent further 'double' actions.
        """
        # Disable double button after first action
        if len(self.env.player_hand.cards) > 2:
            self.buttons[Action.DOUBLE].config(state='disabled')
    
    def double(self):
        """
        Executes the "Double Down" action within the game's environment.
        This allows the player to double their bet, receive exactly one more card, and automatically stand.
        """
        state, reward, done = self.env.step(Action.DOUBLE)
        self.update_display(show_all=True)
        self.game_over(reward)

if __name__ == "__main__":
    root = tk.Tk()
    game = BlackjackGUI(root)
    root.mainloop() 