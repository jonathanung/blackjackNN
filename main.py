from enum import Enum
import random

class Action(Enum):
    """
    Enum class representing possible actions in a game.

    Attributes:
        HIT: Represents the action to hit (value 0).
        STAND: Represents the action to stand (value 1).
        DOUBLE: Represents the action to double (value 2).
    """
    HIT = 0
    STAND = 1
    DOUBLE = 2

class Card:
    """
     class Card:

     def __init__(self, suit, value):
    """
    
    def __init__(self, suit, value):
        """
        @brief Initialize a new card
        @param suit: String representing the card's suit
        @param value: String representing the card's value
        """
        self.suit = suit
        self.value = value
        
    def get_value(self):
        """
        @brief Get the numerical value of the card
        
        @details Converts face cards to their numerical values:
            - J, Q, K = 10
            - A = 11 (Ace value is handled separately in Hand class)
            - Number cards = their face value
        
        @return int: The numerical value of the card
        """
        if self.value in ['J', 'Q', 'K']:
            return 10
        elif self.value == 'A':
            return 11
        return int(self.value)
    
    def __str__(self):
        """Returns a string representation of the card (e.g., 'A♠' or '10♥')"""
        return f"{self.value}{self.suit}"
    
    def __repr__(self):
        """Returns the same as __str__ for cleaner list printing"""
        return self.__str__()

class Hand:
    """
    class Hand:

    def __init__(self):
        """
    
    def __init__(self):
        """@brief Initialize an empty hand"""
        self.cards = []
        
    def add_card(self, card):
        """
        @brief Add a card to the hand
        @param card: Card object to add
        """
        self.cards.append(card)
        
    def get_value(self):
        """
        @brief Calculate the optimal value of the hand
        
        @details Handles ace values intelligently:
            - Counts non-ace cards first
            - For each ace, decides whether to use 1 or 11
            - Maximizes hand value while avoiding bust
        
        @return int: The optimal total value of the hand
        """
        value = 0
        aces = 0
        
        for card in self.cards:
            if card.value == 'A':
                aces += 1
            else:
                value += card.get_value()
                
        # Add aces
        for _ in range(aces):
            if value + 11 <= 21:
                value += 11
            else:
                value += 1
                
        return value
    
    def is_bust(self):
        return self.get_value() > 21
    
    def is_blackjack(self):
        return len(self.cards) == 2 and self.get_value() == 21

class BlackjackEnv:
    """
    class BlackjackEnv:
        A class representing the environment of a Blackjack game, enabling interactions,
        state management, and reward calculations for different actions taken by the player.

        Methods:
        __init__:
            Initializes the BlackjackEnv environment, creates a deck of cards, and sets the player
            and dealer hands to None. Call create_deck method to initialize the deck.

        create_deck:
            Initializes and shuffles the deck of cards. Cards consist of four suits (♠, ♥, ♦, ♣)
            and values ranging from '2' to 'A'. Reconstructs and shuffles the deck.

        reset:
            Resets the game by reshuffling the deck if it has less than 20 cards left, and dealing
            initial hands to the player and dealer. Checks for initial blackjacks and returns the
            game state, reward, and whether the game is done.

        get_state:
            Returns the current state as a dictionary containing the player's hand, player's value,
            dealer's visible card, and dealer's full hand.

        step:
            Performs the given action (HIT, STAND, DOUBLE) and updates the state and rewards based
            on that action. Computes rewards based on general game rules and optimal strategies.
            Returns the updated state, reward, and whether the game is done.
    """
    def __init__(self):
        self.deck = []
        self.player_hand = None
        self.dealer_hand = None
        self.create_deck()
        
    def create_deck(self):
        """
        Generates a standard 52-card deck and shuffles it.

        The deck is composed of cards from four suits (Spades, Hearts, Diamonds, Clubs)
        and thirteen values ('2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A').

        Deck is stored in self.deck.

        The deck is shuffled to ensure randomness.
        """
        suits = ['♠', '♥', '♦', '♣']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.deck = [Card(suit, value) for suit in suits for value in values]
        random.shuffle(self.deck)
        
    def reset(self):
        """
        Resets the game state for a new game round.

        If the deck has fewer than 20 cards, the deck is reshuffled by calling `create_deck`.

        Resets the player's hand and dealer's hand by initializing them to new `Hand` objects.

        Deals two cards each to the player and dealer from the deck.

        Checks for blackjack conditions for both player and dealer.
        - If both have blackjacks, it returns the game state, a reward of 0, and True indicating the game is done (push).
        - If only the player has a blackjack, it returns the game state, a reward of 1.5, and True indicating the game is done.
        - If only the dealer has a blackjack, it returns the game state, a reward of -1, and True indicating the game is done.

        If neither have blackjacks, it returns the game state, a reward of 0, and False indicating the game is not done.
        """
        if len(self.deck) < 20:  # Reshuffle if deck is low
            self.create_deck()
        
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        
        # Initial deal
        self.player_hand.add_card(self.deck.pop())
        self.dealer_hand.add_card(self.deck.pop())
        self.player_hand.add_card(self.deck.pop())
        self.dealer_hand.add_card(self.deck.pop())
        
        # Check for blackjacks
        player_blackjack = self.player_hand.is_blackjack()
        dealer_blackjack = self.dealer_hand.is_blackjack()
        
        if player_blackjack or dealer_blackjack:
            if player_blackjack and dealer_blackjack:
                return self.get_state(), 0, True  # Push
            elif player_blackjack:
                return self.get_state(), 1.5, True  # Player blackjack
            else:
                return self.get_state(), -1, True  # Dealer blackjack
            
        return self.get_state(), 0, False  # Return state, reward=0, done=False for normal start
    
    def get_state(self):
        """Returns the current state as a dictionary with all necessary information"""
        return {
            'player_hand': self.player_hand.cards,
            'player_value': self.player_hand.get_value(),
            'dealer_visible': self.dealer_hand.cards[0],  # Only the first card is visible
            'dealer_hand': self.dealer_hand.cards,  # Full hand, might be useful for training
        }
    
    def step(self, action):
        """
        Perform one step in the game environment based on the given action.

        Args:
            action (Action): The action to be taken by the player. This can be HIT, STAND, or DOUBLE.

        Returns:
            tuple: A tuple containing the current state of the game, the reward obtained from the action, and a boolean indicating if the game is done.

        If the action is HIT:
            - A card is added to the player's hand.
            - If the player's hand value exceeds 21, the player busts, resulting in a base reward of -1 and ending the game.
            - If the player's hand value is exactly 21, a reward of 0.5 is given and the game ends.

        If the action is STAND:
            - Standing incurs a small penalty if the player's hand value is less than 12.
            - The dealer plays their hand until the value is at least 17.
            - If the dealer busts, a reward of 1.2 is given to the player.
            - If the player's hand value is greater than the dealer's, a reward of 1 is given.
            - If the player's hand value is less than the dealer's, a reward of -1 is given.
            - If the hand values are equal, no reward is given (push).
            - The game ends after the dealer's turn.

        If the action is DOUBLE:
            - The action is only valid if the player has exactly two cards in their hand.
            - If invalid, a penalty of -2 is applied and the game ends.
            - A card is added to the player's hand, and the game plays out similarly to a HIT with doubled rewards and penalties.

        Additional reward shaping:
            - If the game is not done, rewards are adjusted based on optimal strategic decisions in the game.
            - Correct decisions (e.g., hitting on certain values, standing on others) provide small additional rewards to encourage optimal play.
        """
        base_reward = 0
        done = False
        
        if action == Action.HIT:
            self.player_hand.add_card(self.deck.pop())
            if self.player_hand.is_bust():
                base_reward = -1
                done = True
            elif self.player_hand.get_value() == 21:
                # Reward getting exactly 21
                base_reward = 0.5
                done = True
                
        elif action == Action.STAND:
            # Small penalty for standing on very low values
            if self.player_hand.get_value() < 12:
                base_reward = -0.5
            
            # Dealer's turn
            while self.dealer_hand.get_value() < 17:
                self.dealer_hand.add_card(self.deck.pop())
                
            player_value = self.player_hand.get_value()
            dealer_value = self.dealer_hand.get_value()
            
            if dealer_value > 21:
                base_reward = 1.2  # Extra reward for dealer bust
            elif player_value > dealer_value:
                base_reward = 1
            elif player_value < dealer_value:
                base_reward = -1
            else:
                base_reward = 0  # Push
            done = True
            
        elif action == Action.DOUBLE:
            # Can only double on first action when you have exactly 2 cards
            if len(self.player_hand.cards) != 2:
                return self.get_state(), -2, True  # Bigger penalty for invalid double
            
            # Hit exactly once and double the stakes
            self.player_hand.add_card(self.deck.pop())
            if self.player_hand.get_value() > 21:
                return self.get_state(), -2, True  # Double loss on bust
            
            # Play out dealer's hand
            while self.dealer_hand.get_value() < 17:
                self.dealer_hand.add_card(self.deck.pop())
            
            dealer_value = self.dealer_hand.get_value()
            player_value = self.player_hand.get_value()
            
            if dealer_value > 21:
                return self.get_state(), 2.4, True  # Extra reward for dealer bust on double
            elif player_value > dealer_value:
                return self.get_state(), 2.2, True  # Extra reward for winning on double
            elif player_value < dealer_value:
                return self.get_state(), -2, True
            else:
                return self.get_state(), 0, True  # Push
        
        # Additional reward shaping based on optimal strategy
        if not done:
            player_value = self.player_hand.get_value()
            dealer_up_card = self.dealer_hand.cards[0].get_value()
            
            # Reward for making strategically sound decisions
            if action == Action.HIT:
                if player_value <= 11:
                    base_reward += 0.1  # Always correct to hit
                elif player_value == 12 and dealer_up_card in [2, 3]:
                    base_reward += 0.1  # Correct to hit against 2 or 3
                elif 12 <= player_value <= 16 and dealer_up_card >= 7:
                    base_reward += 0.1  # Correct to hit against high cards
                    
            elif action == Action.STAND:
                if player_value >= 17:
                    base_reward += 0.1  # Always correct to stand
                elif 13 <= player_value <= 16 and dealer_up_card <= 6:
                    base_reward += 0.1  # Correct to stand against low cards
                    
            elif action == Action.DOUBLE:
                if len(self.player_hand.cards) == 2:
                    if player_value == 11:
                        base_reward += 0.2  # Always correct to double on 11
                    elif player_value == 10 and dealer_up_card <= 9:
                        base_reward += 0.2  # Correct to double on 10 vs low cards
                    elif player_value == 9 and 3 <= dealer_up_card <= 6:
                        base_reward += 0.2  # Correct to double on 9 vs middle cards
        
        return self.get_state(), base_reward, done
