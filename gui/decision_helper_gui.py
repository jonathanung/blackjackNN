import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import tkinter as tk
from tkinter import ttk, messagebox
from main import Action, Card
from agent.blackjack_agent import DeepQLearningAgent
import torch

class DecisionHelperGUI:
    """
        class DecisionHelperGUI:
            A GUI application that helps in making decisions for Blackjack using a trained Deep Q-Learning agent.

        def __init__(self, root):
            Initialize the DecisionHelperGUI with the given root window.

            Args:
                root: The root window of the Tkinter application.

        def get_suggestion(self):
            Get agent's suggestion based on current inputs.

            The method reads the player's hand value, number of cards, and the dealer's visible card from
            the user inputs. It then creates a mock state and passes it to the Deep Q-Learning agent to get
            Q-values and action probabilities. The best action and agent's confidence are displayed in the GUI.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Blackjack Decision Helper")
        
        # Load the trained agent
        self.agent = DeepQLearningAgent()
        try:
            agent_path = os.path.join(project_root, 'agent', 'res', 'blackjack_agent.pkl')
            self.agent.load(agent_path)
            print("Loaded trained agent successfully")
        except FileNotFoundError:
            print("No trained agent found")
            messagebox.showerror("Error", "No trained agent model found!")
            return
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add after creating the main frame
        self.style = ttk.Style()
        self.style.configure("Green.TLabel", foreground="green")
        self.style.configure("Orange.TLabel", foreground="orange")
        self.style.configure("Red.TLabel", foreground="red")
        self.style.configure("Default.TLabel", foreground="black")
        
        # Player hand section
        ttk.Label(self.main_frame, text="Your Hand Value:", font=('Arial', 12)).grid(row=0, column=0, pady=5)
        self.player_value = ttk.Entry(
            self.main_frame, 
            width=10, 
            validate='key', 
            validatecommand=(self.root.register(self.validate_number), '%P')
        )
        self.player_value.grid(row=0, column=1, pady=5)
        self.player_value.bind('<KeyRelease>', self.on_input_change)
        
        ttk.Label(self.main_frame, text="Number of Cards:", font=('Arial', 12)).grid(row=1, column=0, pady=5)
        self.num_cards = ttk.Entry(
            self.main_frame, 
            width=10, 
            validate='key', 
            validatecommand=(self.root.register(self.validate_cards), '%P')
        )
        self.num_cards.grid(row=1, column=1, pady=5)
        self.num_cards.bind('<KeyRelease>', self.on_input_change)
        
        # Dealer card section
        ttk.Label(self.main_frame, text="Dealer's Visible Card:", font=('Arial', 12)).grid(row=2, column=0, pady=5)
        
        # Dropdown for dealer's card
        self.dealer_cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.dealer_card_var = tk.StringVar()
        self.dealer_card_dropdown = ttk.Combobox(
            self.main_frame, 
            textvariable=self.dealer_card_var,
            values=self.dealer_cards,
            width=7
        )
        self.dealer_card_dropdown.grid(row=2, column=1, pady=5)
        self.dealer_card_dropdown.set(self.dealer_cards[0])
        self.dealer_card_var.trace_add('write', self.on_dealer_card_change)
        
        # Get suggestion button
        ttk.Button(
            self.main_frame, 
            text="Get Suggestion", 
            command=self.get_suggestion
        ).grid(row=3, column=0, columnspan=2, pady=20)
        
        # Suggestion display
        self.suggestion_frame = ttk.Frame(self.main_frame)
        self.suggestion_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        self.suggestion_label = ttk.Label(
            self.suggestion_frame, 
            text="Suggestion will appear here",
            font=('Arial', 14, 'bold'),
            style="Default.TLabel"
        )
        self.suggestion_label.pack()
        
        # Add detailed Q-values display first (around row 5)
        self.q_values_frame = ttk.LabelFrame(self.main_frame, text="Action Values", 
                                           padding="10")
        self.q_values_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        self.q_value_labels = {}
        for i, action in enumerate(Action):
            ttk.Label(self.q_values_frame, 
                     text=f"{action.name}:", 
                     font=('Arial', 12)).grid(row=i, column=0, padx=5)
            self.q_value_labels[action] = ttk.Label(self.q_values_frame, 
                                                  text="0.00", 
                                                  font=('Arial', 12))
            self.q_value_labels[action].grid(row=i, column=1, padx=5)
        
        # Move confidence display below Q-values frame (now row 6)
        ttk.Label(self.main_frame, 
            text="Agent Confidence:", 
            font=('Arial', 12)
        ).grid(row=6, column=0, pady=5)
        
        self.confidence_label = ttk.Label(self.main_frame, 
            text="", 
            font=('Arial', 12)
        )
        self.confidence_label.grid(row=6, column=1, pady=5)
        
        # Add some padding to all widgets
        for child in self.main_frame.winfo_children():
            child.grid_configure(padx=5)
    
    def get_suggestion(self):
        """
        get_suggestion(self)

        Updates the UI to suggest an action for the player using the trained agent's Q-values and action probabilities.

        Steps:
        1. Retrieve player value, number of cards, and dealer's visible card from the UI inputs.
        2. Convert face card values (J, Q, K, A) to numerical values suitable for Q-value calculations.
        3. Create a mock state dictionary that includes player and dealer hand representations.
        4. Use the trained agent's network to calculate Q-values from the state tensor.
        5. Update the Q-value labels in the UI with the calculated values and the associated action probabilities.
        6. Identify the best action based on Q-values, determine its corresponding action suggestion, and confidence level.
        7. Update the suggestion label in the UI with the suggested action and confidence.
        8. Change text color of suggestion label based on confidence level to indicate certainty (green, orange, red).

        Handles ValueError exceptions and shows an error message if the input values are invalid.
        """
        try:
            # Create state dictionary from inputs
            player_value = int(self.player_value.get())
            num_cards = int(self.num_cards.get())
            dealer_card = self.dealer_card_var.get()
            
            # Convert face cards to values
            if dealer_card in ['J', 'Q', 'K']:
                dealer_value = 10
            elif dealer_card == 'A':
                dealer_value = 11
            else:
                dealer_value = int(dealer_card)
            
            # Create mock state
            state = {
                'player_value': player_value,
                'player_hand': [Card('♠', 'X')] * num_cards,  # Dummy cards
                'dealer_visible': Card('♠', dealer_card),
                'dealer_hand': [Card('♠', dealer_card)]
            }
            
            # Get Q-values and action from agent
            with torch.no_grad():
                state_tensor = self.agent.state_to_tensor(state)
                q_values = self.agent.q_network(state_tensor)
                action_probs = torch.softmax(q_values, dim=0)
                
                # Update Q-value displays
                for action in Action:
                    q_value = q_values[action.value].item()
                    prob = action_probs[action.value].item()
                    self.q_value_labels[action].config(
                        text=f"Q: {q_value:.2f}\n(P: {prob:.2%})")
                
                best_action_idx = q_values.argmax().item()
                confidence = action_probs[best_action_idx].item()
                
                # Update suggestion and confidence
                suggestion = Action(best_action_idx).name
                self.suggestion_label.config(text=f"Suggested Action:\n{suggestion}")
                self.confidence_label.config(
                    text=f"{confidence:.1%}"
                )
                
                # Color code based on confidence
                color_style = "Green.TLabel" if confidence > 0.8 else "Orange.TLabel" if confidence > 0.6 else "Red.TLabel"
                self.suggestion_label.configure(style=color_style)
                
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
            return

    def validate_number(self, value):
        """Validate player hand value input"""
        if value == "":
            return True
        try:
            num = int(value)
            return 1 <= num <= 21
        except ValueError:
            return False

    def validate_cards(self, value):
        """Validate number of cards input"""
        if value == "":
            return True
        try:
            num = int(value)
            return 1 <= num <= 5  # Maximum 5 cards possible
        except ValueError:
            return False

    def on_input_change(self, event=None):
        """Handle any input change"""
        if self.is_input_valid():
            self.update_suggestion()
        else:
            self.clear_suggestion()

    def on_dealer_card_change(self, *args):
        """Handle dealer card selection change"""
        if self.is_input_valid():
            self.update_suggestion()

    def is_input_valid(self):
        """Check if all inputs are valid"""
        try:
            player_value = int(self.player_value.get())
            num_cards = int(self.num_cards.get())
            return (1 <= player_value <= 21 and 
                   1 <= num_cards <= 5 and 
                   self.dealer_card_var.get() in self.dealer_cards)
        except ValueError:
            return False

    def clear_suggestion(self):
        """Clear all suggestion displays"""
        self.suggestion_label.configure(
            text="Enter valid values", 
            style="Default.TLabel"
        )
        self.confidence_label.config(text="")
        for action in Action:
            self.q_value_labels[action].config(text="")

    def update_suggestion(self):
        """Update the suggestion based on current inputs"""
        try:
            # Create state dictionary from inputs
            player_value = int(self.player_value.get())
            num_cards = int(self.num_cards.get())
            dealer_card = self.dealer_card_var.get()
            
            # Convert face cards to values
            if dealer_card in ['J', 'Q', 'K']:
                dealer_value = 10
            elif dealer_card == 'A':
                dealer_value = 11
            else:
                dealer_value = int(dealer_card)
            
            # Create mock state
            state = {
                'player_value': player_value,
                'player_hand': [Card('♠', 'X')] * num_cards,  # Dummy cards
                'dealer_visible': Card('♠', dealer_card),
                'dealer_hand': [Card('♠', dealer_card)]
            }
            
            # Get Q-values and action from agent
            with torch.no_grad():
                state_tensor = self.agent.state_to_tensor(state)
                q_values = self.agent.q_network(state_tensor)
                action_probs = torch.softmax(q_values, dim=0)
                
                # Update Q-value displays
                for action in Action:
                    q_value = q_values[action.value].item()
                    prob = action_probs[action.value].item()
                    self.q_value_labels[action].config(
                        text=f"Q: {q_value:.2f}\n(P: {prob:.2%})")
                
                best_action_idx = q_values.argmax().item()
                confidence = action_probs[best_action_idx].item()
                
                # Update suggestion and confidence
                suggestion = Action(best_action_idx).name
                self.suggestion_label.config(text=f"Suggested Action:\n{suggestion}")
                self.confidence_label.config(
                    text=f"{confidence:.1%}"
                )
                
                # Color code based on confidence
                color_style = (
                    "Green.TLabel" if confidence > 0.8 
                    else "Orange.TLabel" if confidence > 0.6 
                    else "Red.TLabel"
                )
                self.suggestion_label.configure(style=color_style)
                
        except ValueError:
            self.clear_suggestion()

if __name__ == "__main__":
    root = tk.Tk()
    app = DecisionHelperGUI(root)
    root.mainloop() 