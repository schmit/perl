from collections import namedtuple
import random

from .core import MDP

Hand = namedtuple("Hand", "sum_cards aces")


def compute_score(hand):
    possible_scores = (hand.sum_cards + hand.aces + i * 10 for i in range(hand.aces+1))
    valid_scores = (score for score in possible_scores if score <= 21)
    best_score = max(valid_scores, default=22)
    return best_score

def update_hand(state, card):
    if card == "A":
        new_state = Hand(state.sum_cards, 1)
    else:
        new_state = Hand(state.sum_cards + card, state.aces)

    if compute_score(new_state) <= 21:
        return new_state
    return None

def iswin(player_hand, dealer_hand):
    if dealer_hand is None:
        return True
    return compute_score(player_hand) > compute_score(dealer_hand)

def Blackjack(cards=["A", 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    """
    Variant of Blackjack game with fixed threshold

    Game is played with replacement of card,
    player can "Hit": draw another card, and lose if score is over 21, or "Stand",
    Actions:
        - Hit: draw another card, lose if score is over 21
        - Stand: draw one more card, if over 21 then win else lose

    Further notes:
    The ace ("A") is worth either 1 or 11.
    There can only be 1 ace (to reduce state space)
    """
    ncards = len(cards)

    def initial_states():
        return [(1, Hand(0, 0))]

    def actions(state):
        return ["Hit", "Stand"]

    def transitions(state, action):
        if action == "Hit":
            updated_states = (update_hand(state, card) for card in cards)
            return [(1/ncards, (updated_state, 0 if updated_state else -1))
                    for updated_state in updated_states]

        if action == "Stand":
            dealer_hand = Hand(max(10, state.sum_cards), 0)
            updated_states = (update_hand(dealer_hand, card) for card in cards)
            return [(1/ncards, (None, 1 if iswin(state, dealer_state) else -1))
                    for dealer_state in updated_states]

    return MDP(initial_states, actions, transitions, 0.99)



