def format_hand(cards):
    """
    Formats a list of card objects into a string representation.

    :param cards: List of card objects, where each card has 'value' and 'suit' attributes.
    :return: A single string with each card's value and suit concatenated, separated by spaces.
    """
    return ' '.join([f"{card.value}{card.suit}" for card in cards])

def get_card_color(suit):
    """
    Determines the color of a playing card based on its suit.

    Args:
        suit (str): The suit of the card, which should be one of the following strings: '♥', '♦', '♠', '♣'.

    Returns:
        str: The color of the card, either "red" for hearts and diamonds, or "black" for spades and clubs.
    """
    return "red" if suit in ['♥', '♦'] else "black" 