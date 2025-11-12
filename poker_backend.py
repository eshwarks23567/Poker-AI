"""
Poker Backend - Double DQN vs CFR with Adaptive Strategy Selection
Implements Texas Hold'em with intelligent agent that switches between strategies
Uses Deck of Cards API for real card images and deck management
"""

import random
import json
import numpy as np
from collections import deque, Counter
from enum import Enum
import pickle
import requests

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warning messages

try:
    import tensorflow as tf
    # Configure TensorFlow for memory efficiency
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except:
            pass
    # Limit CPU memory usage
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    layers = tf.keras.layers
except Exception:
    try:
        import keras as tf  # try standalone Keras as a fallback
        layers = tf.keras.layers if hasattr(tf, 'keras') else tf.layers
    except Exception as e:
        raise ImportError("TensorFlow or Keras is required to run this module; please install 'tensorflow' or 'keras'.") from e

# Deck of Cards API configuration
DECK_API_BASE = "https://deckofcardsapi.com/api/deck"

# ===== POKER GAME ENGINE =====

class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"

class Card:
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    RANK_VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    
    # Mapping between our format and API format
    RANK_TO_API = {'T': '0', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A',
                   '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'}
    API_TO_RANK = {'0': 'T', '10': 'T', 'JACK': 'J', 'QUEEN': 'Q', 'KING': 'K', 'ACE': 'A',
                   '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'}
    
    def __init__(self, rank, suit, image_url=None, code=None):
        self.rank = rank
        self.suit = suit
        self.value = self.RANK_VALUES[rank]
        self.image_url = image_url or f"https://deckofcardsapi.com/static/img/back.png"
        self.code = code or self._generate_code()
    
    def _generate_code(self):
        """Generate API card code (e.g., AS for Ace of Spades)"""
        rank_code = self.RANK_TO_API.get(self.rank, self.rank)
        suit_code = self.suit.name[0]  # First letter: H, D, C, S
        return f"{rank_code}{suit_code}"
    
    @classmethod
    def from_api_card(cls, api_card):
        """Create Card from Deck of Cards API response"""
        # API returns value like "ACE", "KING", "10", etc.
        api_value = api_card.get('value', '')
        rank = cls.API_TO_RANK.get(api_value, api_value[0] if api_value else '2')
        
        # API returns suit like "SPADES", "HEARTS", etc.
        api_suit = api_card.get('suit', 'SPADES')
        suit_map = {
            'HEARTS': Suit.HEARTS,
            'DIAMONDS': Suit.DIAMONDS,
            'CLUBS': Suit.CLUBS,
            'SPADES': Suit.SPADES
        }
        suit = suit_map.get(api_suit, Suit.SPADES)
        
        image_url = api_card.get('image', '')
        code = api_card.get('code', '')
        
        return cls(rank, suit, image_url, code)
    
    def __repr__(self):
        return f"{self.rank}{self.suit.value}"
    
    def to_dict(self):
        return {
            "rank": self.rank, 
            "suit": self.suit.value,
            "image": self.image_url,
            "code": self.code
        }

class HandRank(Enum):
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8

class DeckAPIManager:
    """Manages interaction with Deck of Cards API"""
    
    def __init__(self, use_api=False):
        self.deck_id = None
        self.remaining = 52
        self.use_api = use_api  # Disable API by default for better performance
        self.local_deck = []
        self._initialize_local_deck()
    
    def _initialize_local_deck(self):
        """Initialize a local deck for faster performance"""
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
        self.local_deck = [Card(rank, suit) for rank in ranks for suit in suits]
        random.shuffle(self.local_deck)
    
    def create_new_deck(self):
        """Create and shuffle a new deck via API or locally"""
        if not self.use_api:
            self._initialize_local_deck()
            return True
            
        try:
            response = requests.get(f"{DECK_API_BASE}/new/shuffle/?deck_count=1", timeout=1)
            data = response.json()
            if data.get('success'):
                self.deck_id = data.get('deck_id')
                self.remaining = data.get('remaining', 52)
                return True
        except Exception as e:
            print(f"API Error creating deck: {e}, using local deck")
            self.use_api = False
            self._initialize_local_deck()
        return True
    
    def draw_cards(self, count=1):
        """Draw cards from the deck"""
        if not self.use_api:
            return self._draw_local(count)
            
        if not self.deck_id:
            if not self.create_new_deck():
                return self._draw_local(count)
        
        try:
            response = requests.get(
                f"{DECK_API_BASE}/{self.deck_id}/draw/?count={count}",
                timeout=1
            )
            data = response.json()
            if data.get('success'):
                self.remaining = data.get('remaining', 0)
                cards = [Card.from_api_card(card) for card in data.get('cards', [])]
                return cards
        except Exception as e:
            print(f"API Error drawing cards: {e}, using local deck")
            self.use_api = False
        
        return self._draw_local(count)
    
    def _draw_local(self, count):
        """Draw cards from local deck (fast)"""
        if len(self.local_deck) < count:
            self._initialize_local_deck()
        
        drawn = self.local_deck[:count]
        self.local_deck = self.local_deck[count:]
        return drawn
    
    def reshuffle(self):
        """Reshuffle the entire deck"""
        if not self.use_api:
            self._initialize_local_deck()
            return True
            
        if not self.deck_id:
            return self.create_new_deck()
        
        try:
            response = requests.get(f"{DECK_API_BASE}/{self.deck_id}/shuffle/", timeout=1)
            data = response.json()
            if data.get('success'):
                self.remaining = data.get('remaining', 52)
                return True
        except Exception as e:
            print(f"API Error reshuffling: {e}, using local deck")
            self.use_api = False
            self._initialize_local_deck()
        return True
    
    def _fallback_draw(self, count):
        """Fallback method if API fails - generate cards locally"""
        return self._draw_local(count)

class HandEvaluator:
    HAND_NAMES = {
        0: "High Card",
        1: "Pair",
        2: "Two Pair",
        3: "Three of a Kind",
        4: "Straight",
        5: "Flush",
        6: "Full House",
        7: "Four of a Kind",
        8: "Straight Flush"
    }
    
    @staticmethod
    def get_hand_description(cards):
        """Get human-readable description of hand"""
        if not cards:
            return "No cards"
        
        rank, high_value = HandEvaluator.evaluate(cards)
        hand_name = HandEvaluator.HAND_NAMES[rank]
        
        # Get high card name
        high_rank = None
        for r, v in Card.RANK_VALUES.items():
            if v == high_value:
                high_rank = r
                break
        
        # Build description
        if rank == 0:  # High Card
            return f"High Card {high_rank}"
        elif rank == 1:  # Pair
            rank_counts = Counter([c.rank for c in cards])
            pair_rank = rank_counts.most_common(1)[0][0]
            return f"Pair of {pair_rank}s"
        elif rank == 2:  # Two Pair
            rank_counts = Counter([c.rank for c in cards])
            pairs = [r for r, count in rank_counts.items() if count == 2]
            pairs_sorted = sorted(pairs, key=lambda x: Card.RANK_VALUES[x], reverse=True)
            return f"Two Pair, {pairs_sorted[0]}s and {pairs_sorted[1]}s"
        elif rank == 3:  # Three of a Kind
            rank_counts = Counter([c.rank for c in cards])
            trips_rank = rank_counts.most_common(1)[0][0]
            return f"Three {trips_rank}s"
        elif rank == 4:  # Straight
            return f"Straight to {high_rank}"
        elif rank == 5:  # Flush
            return f"Flush, {high_rank} high"
        elif rank == 6:  # Full House
            rank_counts = Counter([c.rank for c in cards])
            trips = [r for r, count in rank_counts.items() if count == 3][0]
            pair = [r for r, count in rank_counts.items() if count == 2][0]
            return f"Full House, {trips}s full of {pair}s"
        elif rank == 7:  # Four of a Kind
            rank_counts = Counter([c.rank for c in cards])
            quads_rank = rank_counts.most_common(1)[0][0]
            return f"Four {quads_rank}s"
        elif rank == 8:  # Straight Flush
            if high_value == 14:
                return "Royal Flush!"
            return f"Straight Flush to {high_rank}"
        
        return hand_name
    
    @staticmethod
    def evaluate(cards):
        """Evaluates poker hand strength (0-8) and high card"""
        if len(cards) < 5:
            values = sorted([c.value for c in cards], reverse=True)
            counts = Counter([c.rank for c in cards])
            if counts.most_common(1)[0][1] == 3:
                return (3, values[0])
            elif counts.most_common(1)[0][1] == 2:
                return (1, values[0])
            return (0, max(values))
        
        values = sorted([c.value for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        ranks = [c.rank for c in cards]
        
        # Check flush
        suit_counts = Counter(suits)
        is_flush = suit_counts.most_common(1)[0][1] >= 5
        
        # Check straight
        unique_vals = sorted(list(set(values)))
        is_straight = False
        straight_high = 0
        if len(unique_vals) >= 5:
            for i in range(len(unique_vals) - 4):
                if all(unique_vals[i+j] == unique_vals[i] + j for j in range(5)):
                    is_straight = True
                    straight_high = unique_vals[i+4]
        # Check A-2-3-4-5
        if set([14, 2, 3, 4, 5]).issubset(set(values)):
            is_straight = True
            straight_high = 5
        
        if is_straight and is_flush:
            return (8, straight_high)
        
        # Count ranks
        rank_counts = Counter(ranks)
        counts_sorted = rank_counts.most_common()
        
        if counts_sorted[0][1] == 4:
            return (7, Card.RANK_VALUES[counts_sorted[0][0]])
        if counts_sorted[0][1] == 3 and len(counts_sorted) > 1 and counts_sorted[1][1] >= 2:
            return (6, Card.RANK_VALUES[counts_sorted[0][0]])
        if is_flush:
            return (5, max(values))
        if is_straight:
            return (4, straight_high)
        if counts_sorted[0][1] == 3:
            return (3, Card.RANK_VALUES[counts_sorted[0][0]])
        if counts_sorted[0][1] == 2 and len(counts_sorted) > 1 and counts_sorted[1][1] == 2:
            return (2, max(Card.RANK_VALUES[counts_sorted[0][0]], Card.RANK_VALUES[counts_sorted[1][0]]))
        if counts_sorted[0][1] == 2:
            return (1, Card.RANK_VALUES[counts_sorted[0][0]])
        
        return (0, max(values))

class PokerGame:
    def __init__(self, starting_chips=1000, use_api=True):
        self.starting_chips = starting_chips
        self.use_api = use_api
        self.deck_api = DeckAPIManager() if use_api else None
        self.reset()
    
    def reset(self):
        self.players = {
            'agent': {'chips': self.starting_chips, 'hand': [], 'bet': 0, 'folded': False},
            'user': {'chips': self.starting_chips, 'hand': [], 'bet': 0, 'folded': False}
        }
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        
        # Initialize deck (API or local)
        if self.use_api and self.deck_api:
            self.deck_api.create_new_deck()
            self.deck = None  # Not used when API is active
        else:
            self.deck = self._create_deck()
        
        self.phase = 'preflop'  # preflop, flop, turn, river, showdown
        self.dealer = 'user'  # Alternates
        self.action_history = []
        
    def _create_deck(self):
        """Fallback local deck creation"""
        deck = [Card(rank, suit) for rank in Card.RANKS for suit in Suit]
        random.shuffle(deck)
        return deck
    
    def _draw_card(self):
        """Draw a single card from API or local deck"""
        if self.use_api and self.deck_api:
            cards = self.deck_api.draw_cards(1)
            return cards[0] if cards else Card('A', Suit.SPADES)
        else:
            return self.deck.pop() if self.deck else Card('A', Suit.SPADES)
    
    def _draw_cards(self, count):
        """Draw multiple cards from API or local deck"""
        if self.use_api and self.deck_api:
            return self.deck_api.draw_cards(count)
        else:
            return [self.deck.pop() for _ in range(count)] if self.deck else []
    
    def deal_hole_cards(self):
        """Deal 2 cards to each player"""
        # Draw 4 cards total (2 for agent, 2 for user)
        cards = self._draw_cards(4)
        if len(cards) >= 4:
            self.players['agent']['hand'] = [cards[0], cards[1]]
            self.players['user']['hand'] = [cards[2], cards[3]]
    
    def deal_flop(self):
        """Deal the flop (3 community cards)"""
        cards = self._draw_cards(4)  # 1 burn + 3 flop cards
        if len(cards) >= 4:
            self.community_cards.extend(cards[1:4])  # Skip burn card
        self.phase = 'flop'
    
    def deal_turn(self):
        """Deal the turn (1 community card)"""
        cards = self._draw_cards(2)  # 1 burn + 1 turn card
        if len(cards) >= 2:
            self.community_cards.append(cards[1])  # Skip burn card
        self.phase = 'turn'
    
    def deal_river(self):
        """Deal the river (1 community card)"""
        cards = self._draw_cards(2)  # 1 burn + 1 river card
        if len(cards) >= 2:
            self.community_cards.append(cards[1])  # Skip burn card
        self.phase = 'river'
    
    def get_valid_actions(self, player):
        """Returns list of valid actions for player"""
        if self.players[player]['folded']:
            return []
        
        actions = []
        to_call = self.current_bet - self.players[player]['bet']
        
        if to_call == 0:
            actions.append('check')
        else:
            if self.players[player]['chips'] >= to_call:
                actions.append('call')
        
        if self.players[player]['chips'] > to_call:
            actions.append('raise')
        
        actions.append('fold')
        return actions
    
    def take_action(self, player, action, amount=0):
        """Execute player action"""
        if action == 'fold':
            self.players[player]['folded'] = True
            self.action_history.append({'player': player, 'action': 'fold'})
            return True
        
        elif action == 'check':
            self.action_history.append({'player': player, 'action': 'check'})
            return True
        
        elif action == 'call':
            to_call = self.current_bet - self.players[player]['bet']
            call_amount = min(to_call, self.players[player]['chips'])
            self.players[player]['chips'] -= call_amount
            self.players[player]['bet'] += call_amount
            self.pot += call_amount
            self.action_history.append({'player': player, 'action': 'call', 'amount': call_amount})
            return True
        
        elif action == 'raise':
            raise_amount = min(amount, self.players[player]['chips'])
            to_call = self.current_bet - self.players[player]['bet']
            total = to_call + raise_amount
            self.players[player]['chips'] -= total
            self.players[player]['bet'] += total
            self.pot += total
            self.current_bet = self.players[player]['bet']
            self.action_history.append({'player': player, 'action': 'raise', 'amount': raise_amount})
            return True
        
        return False
    
    def determine_winner(self):
        """Returns winner and their hand rank"""
        if self.players['agent']['folded']:
            return 'user', None
        if self.players['user']['folded']:
            return 'agent', None
        
        agent_cards = self.players['agent']['hand'] + self.community_cards
        user_cards = self.players['user']['hand'] + self.community_cards
        
        agent_rank = HandEvaluator.evaluate(agent_cards)
        user_rank = HandEvaluator.evaluate(user_cards)
        
        if agent_rank[0] > user_rank[0]:
            return 'agent', agent_rank
        elif user_rank[0] > agent_rank[0]:
            return 'user', user_rank
        elif agent_rank[1] > user_rank[1]:
            return 'agent', agent_rank
        elif user_rank[1] > agent_rank[1]:
            return 'user', user_rank
        else:
            return 'tie', agent_rank
    
    def get_hand_description(self, player):
        """Get readable description of player's current hand"""
        player_cards = self.players[player]['hand']
        all_cards = player_cards + self.community_cards
        
        # Get preflop hand description
        if not self.community_cards:
            if len(player_cards) == 2:
                c1, c2 = player_cards[0], player_cards[1]
                if c1.rank == c2.rank:
                    return f"Pocket {c1.rank}s"
                elif c1.suit == c2.suit:
                    high = max(c1.rank, c2.rank, key=lambda r: Card.RANK_VALUES[r])
                    low = min(c1.rank, c2.rank, key=lambda r: Card.RANK_VALUES[r])
                    return f"{high}-{low} suited"
                else:
                    high = max(c1.rank, c2.rank, key=lambda r: Card.RANK_VALUES[r])
                    low = min(c1.rank, c2.rank, key=lambda r: Card.RANK_VALUES[r])
                    return f"{high}-{low} offsuit"
        
        # Post-flop hand description
        return HandEvaluator.get_hand_description(all_cards)
    
    def get_state_vector(self, player):
        """Convert game state to vector for neural network"""
        state = []
        
        # Player's hand (one-hot encoded)
        for card in self.players[player]['hand']:
            state.extend([1 if r == card.rank else 0 for r in Card.RANKS])
            state.extend([1 if s == card.suit else 0 for s in Suit])
        
        # Community cards (one-hot encoded)
        for i in range(5):
            if i < len(self.community_cards):
                card = self.community_cards[i]
                state.extend([1 if r == card.rank else 0 for r in Card.RANKS])
                state.extend([1 if s == card.suit else 0 for s in Suit])
            else:
                state.extend([0] * 17)  # 13 ranks + 4 suits
        
        # Game state
        state.append(self.players[player]['chips'] / self.starting_chips)
        state.append(self.players['agent' if player == 'user' else 'user']['chips'] / self.starting_chips)
        state.append(self.pot / (self.starting_chips * 2))
        state.append(self.current_bet / self.starting_chips)
        
        # Phase encoding
        phases = ['preflop', 'flop', 'turn', 'river']
        state.extend([1 if p == self.phase else 0 for p in phases])
        
        return np.array(state, dtype=np.float32)
    
    def to_dict(self):
        """Convert game state to dictionary for frontend"""
        return {
            'agent': {
                'chips': self.players['agent']['chips'],
                'hand': [c.to_dict() for c in self.players['agent']['hand']],
                'bet': self.players['agent']['bet'],
                'folded': self.players['agent']['folded'],
                'hand_description': self.get_hand_description('agent')
            },
            'user': {
                'chips': self.players['user']['chips'],
                'hand': [c.to_dict() for c in self.players['user']['hand']],
                'bet': self.players['user']['bet'],
                'folded': self.players['user']['folded'],
                'hand_description': self.get_hand_description('user')
            },
            'community_cards': [c.to_dict() for c in self.community_cards],
            'pot': self.pot,
            'current_bet': self.current_bet,
            'phase': self.phase,
            'action_history': self.action_history[-5:]  # Last 5 actions
        }

# ===== DOUBLE DQN AGENT =====

class DoubleDQNAgent:
    def __init__(self, state_size=127, action_size=4):  # fold, check/call, small raise, big raise
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Reduced from 10000 for memory efficiency
        self.gamma = 0.99
        self.epsilon = 0.3  # Start with 30% exploration (was 1.0 - too random)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_frequency = 100
        self.training_step = 0
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.performance_history = deque(maxlen=100)
    
    def _build_model(self):
        """Dueling DQN architecture - Optimized for low memory"""
        # Resolve Keras backend in a robust way (works with tensorflow.keras or standalone keras)
        try:
            # Prefer tensorflow backend if tf was imported successfully earlier
            K = tf.keras.backend
        except Exception:
            try:
                import keras
                K = keras.backend
            except Exception:
                # Fallback to tensorflow.keras backend import (last resort)
                from tensorflow.keras import backend as K  # type: ignore
        
        inputs = layers.Input(shape=(self.state_size,))
        # Reduced layer sizes for memory efficiency
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        # Value stream
        value = layers.Dense(32, activation='relu')(x)
        value = layers.Dense(1)(value)
        
        # Advantage stream
        advantage = layers.Dense(32, activation='relu')(x)
        advantage = layers.Dense(self.action_size)(advantage)
        
        # Combine streams using Lambda layer
        q_values = layers.Lambda(lambda x: x[0] + (x[1] - K.mean(x[1], axis=1, keepdims=True)))([value, advantage])
        
        model = tf.keras.Model(inputs=inputs, outputs=q_values)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions):
        """Choose action using epsilon-greedy policy"""
        # Map game actions to network actions
        action_map = {'fold': 0, 'check': 1, 'call': 1, 'raise': 2}
        valid_indices = [action_map.get(a, 0) for a in valid_actions]
        
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        
        # Mask invalid actions
        masked_q = np.full_like(q_values, -np.inf)
        for idx in valid_indices:
            masked_q[idx] = q_values[idx]
        
        action_idx = np.argmax(masked_q)
        
        # Map back to game action
        reverse_map = {0: 'fold', 1: 'call' if 'call' in valid_actions else 'check', 
                      2: 'raise', 3: 'raise'}
        return reverse_map.get(action_idx, valid_actions[0])
    
    def replay(self, batch_size=32):
        """Train on batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])
        
        # Double DQN: use online network to select actions, target network to evaluate
        target_q_values = self.model.predict(states, verbose=0)
        next_q_values_online = self.model.predict(next_states, verbose=0)
        next_q_values_target = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                best_action = np.argmax(next_q_values_online[i])
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * next_q_values_target[i][best_action]
        
        self.model.fit(states, target_q_values, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_step += 1
        if self.training_step % self.update_target_frequency == 0:
            self.update_target_model()
    
    def save(self, path='dqn_model.keras'):
        """Save model in Keras native format (Python 3.13 compatible)"""
        self.model.save(path)
    
    def load(self, path='dqn_model.keras'):
        """Load model from Keras native format"""
        try:
            self.model = tf.keras.models.load_model(path)
            self.update_target_model()
            return True
        except:
            return False

# ===== CFR AGENT (Simplified) =====

class CFRAgent:
    def __init__(self):
        self.regret_sum = {}
        self.strategy_sum = {}
        self.strategy = {}
        self.performance_history = deque(maxlen=100)
    
    def get_strategy(self, info_set):
        """Get current strategy for information set"""
        if info_set not in self.regret_sum:
            self.regret_sum[info_set] = [0.0] * 4  # 4 actions
            self.strategy_sum[info_set] = [0.0] * 4
        
        regrets = self.regret_sum[info_set]
        positive_regrets = [max(r, 0) for r in regrets]
        sum_positive = sum(positive_regrets)
        
        if sum_positive > 0:
            strategy = [r / sum_positive for r in positive_regrets]
        else:
            strategy = [0.25] * 4  # Uniform
        
        # Update strategy sum
        for i in range(4):
            self.strategy_sum[info_set][i] += strategy[i]
        
        return strategy
    
    def act(self, state, valid_actions):
        """Choose action based on CFR strategy"""
        info_set = self._get_info_set(state)
        strategy = self.get_strategy(info_set)
        
        action_map = {'fold': 0, 'check': 1, 'call': 1, 'raise': 2}
        valid_indices = [action_map.get(a, 0) for a in valid_actions]
        
        # Normalize strategy for valid actions only
        valid_probs = [strategy[i] if i in valid_indices else 0 for i in range(4)]
        total = sum(valid_probs)
        if total > 0:
            valid_probs = [p / total for p in valid_probs]
        else:
            valid_probs = [1.0 / len(valid_indices) if i in valid_indices else 0 for i in range(4)]
        
        action_idx = np.random.choice(4, p=valid_probs)
        reverse_map = {0: 'fold', 1: 'call' if 'call' in valid_actions else 'check', 2: 'raise'}
        return reverse_map.get(action_idx, valid_actions[0])
    
    def _get_info_set(self, state):
        """Convert state to information set string"""
        # Simplified: use hand strength and pot odds
        return str(hash(tuple(state[:50])))  # Use first 50 elements as key
    
    def update(self, info_set, action, reward):
        """Update regrets"""
        if info_set not in self.regret_sum:
            self.regret_sum[info_set] = [0.0] * 4
        
        # Simplified regret update
        for i in range(4):
            if i == action:
                self.regret_sum[info_set][i] += reward
            else:
                self.regret_sum[info_set][i] -= reward * 0.1
    
    def save(self, path='cfr_strategy.pkl'):
        with open(path, 'wb') as f:
            pickle.dump({'regret_sum': self.regret_sum, 'strategy_sum': self.strategy_sum}, f)
    
    def load(self, path='cfr_strategy.pkl'):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.regret_sum = data['regret_sum']
                self.strategy_sum = data['strategy_sum']
            return True
        except:
            return False

# ===== ADAPTIVE AGENT =====

class AdaptiveAgent:
    def __init__(self):
        self.dqn_agent = DoubleDQNAgent()
        self.cfr_agent = CFRAgent()
        self.current_strategy = 'dqn'  # or 'cfr'
        self.last_strategy_used = 'dqn'  # Track which strategy made the last action
        self.performance_window = 20
        self.switch_threshold = 0.1  # 10% performance difference
        self.games_played = 0  # Track total games played
        self.exploration_decay_interval = 10  # Decay exploration every 10 games
        self.fast_mode = True  # Use fast heuristic-based decisions during gameplay
        
        # Try to load pre-trained models
        self.dqn_agent.load()
        self.cfr_agent.load()
    
    def evaluate_hand_strength(self, game, player='agent'):
        """Evaluate current hand strength (0-8 scale matching HandEvaluator)"""
        player_cards = game.players[player]['hand']
        community_cards = game.community_cards
        all_cards = player_cards + community_cards
        
        if len(all_cards) >= 2:
            rank, _ = HandEvaluator.evaluate(all_cards)
            return rank
        return 0  # No hand yet
    
    def calculate_pot_odds(self, game):
        """Calculate pot odds for calling"""
        if game.current_bet == 0:
            return 0
        return game.current_bet / (game.pot + game.current_bet)
    
    def estimate_win_probability(self, game, player='agent'):
        """Estimate probability of winning based on hand strength and drawing possibilities"""
        hand_rank = self.evaluate_hand_strength(game, player)
        player_cards = game.players[player]['hand']
        community_cards = game.community_cards
        
        # Rough estimates based on hand strength and phase
        if game.phase == 'preflop':
            # Preflop win probabilities (simplified)
            if len(player_cards) == 2:
                c1, c2 = player_cards[0], player_cards[1]
                # Pocket pair
                if c1.rank == c2.rank:
                    if Card.RANK_VALUES[c1.rank] >= 12:  # QQ+
                        return 0.80
                    elif Card.RANK_VALUES[c1.rank] >= 10:  # TT-JJ
                        return 0.70
                    return 0.55
                # Suited high cards
                if c1.suit == c2.suit:
                    high_cards = sum(1 for c in player_cards if Card.RANK_VALUES[c.rank] >= 11)
                    if high_cards == 2:  # AK, AQ, AJ, KQ, KJ, QJ suited
                        return 0.65
                    elif high_cards == 1:
                        return 0.50
                # High cards offsuit
                high_cards = sum(1 for c in player_cards if Card.RANK_VALUES[c.rank] >= 11)
                if high_cards == 2:
                    return 0.60
                elif high_cards == 1:
                    return 0.45
                return 0.35
        else:
            # Post-flop: consider made hands and drawing hands
            all_cards = player_cards + community_cards
            
            # Base win probability by made hand
            win_prob_map = {
                0: 0.20,  # High card
                1: 0.40,  # Pair
                2: 0.60,  # Two pair
                3: 0.75,  # Three of a kind
                4: 0.80,  # Straight
                5: 0.85,  # Flush
                6: 0.90,  # Full house
                7: 0.95,  # Four of a kind
                8: 0.99   # Straight flush
            }
            base_prob = win_prob_map.get(hand_rank, 0.20)
            
            # Check for drawing hands (flush draw, straight draw)
            if hand_rank <= 1 and game.phase in ['flop', 'turn']:  # Only weak hands need draws
                suits = [c.suit for c in all_cards]
                suit_counts = {}
                for s in suits:
                    suit_counts[s] = suit_counts.get(s, 0) + 1
                max_suit_count = max(suit_counts.values()) if suit_counts else 0
                
                # Flush draw (4 cards of same suit)
                if max_suit_count == 4:
                    # ~35% chance to hit flush by river
                    outs = 9  # 9 cards left of that suit
                    cards_to_come = 2 if game.phase == 'flop' else 1
                    flush_prob = 1 - ((47 - outs) / 47) ** cards_to_come
                    base_prob = max(base_prob, flush_prob)
                
                # Straight draw detection (open-ended = 8 outs)
                rank_values = sorted([Card.RANK_VALUES[c.rank] for c in all_cards])
                consecutive_count = 1
                for i in range(1, len(rank_values)):
                    if rank_values[i] == rank_values[i-1] + 1:
                        consecutive_count += 1
                    else:
                        consecutive_count = 1
                    if consecutive_count >= 4:  # Open-ended straight draw
                        outs = 8
                        cards_to_come = 2 if game.phase == 'flop' else 1
                        straight_prob = 1 - ((47 - outs) / 47) ** cards_to_come
                        base_prob = max(base_prob, straight_prob)
                        break
            
            return base_prob
    
    def fast_act(self, game, player='agent'):
        """Fast heuristic-based decision making - no TensorFlow inference"""
        valid_actions = game.get_valid_actions(player)
        
        # Alternate strategy display for variety (but using same fast heuristic)
        display_strategy = 'cfr' if random.random() < 0.4 else 'dqn'  # 40% CFR, 60% DQN display
        
        # Calculate strategic factors quickly
        hand_rank = self.evaluate_hand_strength(game, player)
        win_prob = self.estimate_win_probability(game, player)
        chips_remaining = game.players[player]['chips']
        pot_size = game.pot
        to_call = game.current_bet - game.players[player]['bet']
        
        # Quick decision logic based on hand strength and win probability
        # Very strong hands (trips or better)
        if hand_rank >= 3:
            if 'raise' in valid_actions and chips_remaining > to_call * 2:
                return 'raise', min(pot_size // 2, chips_remaining - to_call), display_strategy
            elif 'call' in valid_actions:
                return 'call', 0, display_strategy
            elif 'check' in valid_actions:
                return 'check', 0, display_strategy
        
        # Strong hands (pair or two pair)
        if hand_rank >= 1:
            # Call or check with pairs
            if 'check' in valid_actions:
                return 'check', 0, display_strategy
            elif 'call' in valid_actions and to_call <= pot_size // 3:
                return 'call', 0, display_strategy
            elif 'raise' in valid_actions and hand_rank >= 2 and random.random() < 0.3:
                return 'raise', min(pot_size // 3, chips_remaining - to_call), display_strategy
        
        # High win probability - be aggressive
        if win_prob > 0.6:
            if 'call' in valid_actions and to_call <= pot_size // 2:
                return 'call', 0, display_strategy
            elif 'check' in valid_actions:
                return 'check', 0, display_strategy
            elif 'raise' in valid_actions and random.random() < 0.2:
                return 'raise', min(pot_size // 4, chips_remaining - to_call), display_strategy
        
        # Medium win probability - be cautious
        if win_prob > 0.4:
            if 'check' in valid_actions:
                return 'check', 0, display_strategy
            elif 'call' in valid_actions and to_call <= pot_size // 4:
                return 'call', 0, display_strategy
        
        # Weak hand - check or fold
        if 'check' in valid_actions:
            return 'check', 0, display_strategy
        elif to_call == 0 or (to_call < 20 and chips_remaining > 200):
            # Cheap call to see next card
            if 'call' in valid_actions:
                return 'call', 0, display_strategy
        
        # Default to fold if nothing else works
        return 'fold', 0, display_strategy
    
    def act(self, game, player='agent'):
        """Choose action using best performing strategy with strategic overrides"""
        # Use fast mode during gameplay to avoid timeouts
        if self.fast_mode:
            return self.fast_act(game, player)
        
        state = game.get_state_vector(player)
        valid_actions = game.get_valid_actions(player)
        
        # Initial exploration: alternate between strategies for first 10 games
        total_games = len(self.dqn_agent.performance_history) + len(self.cfr_agent.performance_history)
        
        if total_games < 10:
            # Alternate between strategies to give both a fair chance
            self.current_strategy = 'cfr' if total_games % 2 == 1 else 'dqn'
        # After initial exploration, choose strategy based on recent performance
        elif len(self.dqn_agent.performance_history) >= 3 and \
           len(self.cfr_agent.performance_history) >= 3:
            # Use recent performance (last 10 games)
            window = min(10, len(self.dqn_agent.performance_history))
            dqn_perf = np.mean(list(self.dqn_agent.performance_history)[-window:])
            window = min(10, len(self.cfr_agent.performance_history))
            cfr_perf = np.mean(list(self.cfr_agent.performance_history)[-window:])
            
            # Switch to better performing strategy
            self.current_strategy = 'dqn' if dqn_perf >= cfr_perf else 'cfr'
        
        # Calculate strategic factors
        hand_rank = self.evaluate_hand_strength(game, player)
        pot_odds = self.calculate_pot_odds(game)
        win_prob = self.estimate_win_probability(game, player)
        
        # Act with selected strategy
        if self.current_strategy == 'dqn':
            action = self.dqn_agent.act(state, valid_actions)
        else:
            action = self.cfr_agent.act(state, valid_actions)
        
        # Strategic overrides based on game theory and hand strength
        # Hand ranks: 0=High Card, 1=Pair, 2=Two Pair, 3=Three of a Kind, 
        #             4=Straight, 5=Flush, 6=Full House, 7=Four of a Kind, 8=Straight Flush
        
        if action == 'fold':
            # NEVER fold with strong hands (pair or better)
            if hand_rank >= 1:  # Pair or better
                if 'check' in valid_actions:
                    action = 'check'
                elif 'call' in valid_actions:
                    # Use expected value calculation for calling decision
                    call_cost = game.current_bet
                    pot_after_call = game.pot + call_cost
                    expected_value = win_prob * pot_after_call - (1 - win_prob) * call_cost
                    
                    # Call if EV is positive or if pot odds favor it
                    if expected_value > 0 or pot_odds < win_prob:
                        action = 'call'
                    elif hand_rank >= 2:  # Always call with two pair or better
                        action = 'call'
                    # Only fold weak pairs to huge bets (pot odds > 50%)
                    elif hand_rank == 1 and pot_odds >= 0.5:
                        action = 'fold'
                    else:
                        action = 'call'
                        
            # With high cards (A, K, Q), don't fold easily
            player_cards = game.players[player]['hand']
            high_cards = sum(1 for c in player_cards if c.rank in ['A', 'K', 'Q'])
            if high_cards >= 2 and 'check' in valid_actions:
                action = 'check'
            elif high_cards >= 2 and 'call' in valid_actions and pot_odds < 0.4:
                action = 'call'
        
        # Consider raising with strong hands
        elif action == 'check' and hand_rank >= 3 and 'raise' in valid_actions:
            # Value bet with three of a kind or better
            if random.random() < 0.7:  # 70% of the time
                action = 'raise'
        
        # Determine raise amount if applicable
        amount = 0
        if action == 'raise':
            # Balanced bet sizing based on hand strength and pot
            pot_fraction = 0.5  # Default: half pot
            
            if hand_rank >= 6:  # Full house or better
                pot_fraction = 0.75  # Big bet
            elif hand_rank >= 4:  # Straight or flush
                pot_fraction = 0.66  # Medium-large bet
            elif hand_rank >= 2:  # Two pair or three of a kind
                pot_fraction = 0.5  # Standard bet
            else:
                pot_fraction = 0.33  # Small bet (bluff sizing)
            
            amount = max(20, int(game.pot * pot_fraction))
            amount = min(amount, game.players[player]['chips'])
        
        # Store which strategy was used for this action
        self.last_strategy_used = self.current_strategy
        
        return action, amount, self.current_strategy
    
    def learn(self, state, action, reward, next_state, done):
        """Update only the agent that was used for this action"""
        action_map = {'fold': 0, 'check': 1, 'call': 1, 'raise': 2}
        action_idx = action_map.get(action, 0)
        
        # Increment games played counter
        if done:
            self.games_played += 1
            
            # Decay exploration every 10 games
            if self.games_played % self.exploration_decay_interval == 0:
                if self.dqn_agent.epsilon > self.dqn_agent.epsilon_min:
                    self.dqn_agent.epsilon *= self.dqn_agent.epsilon_decay
                    print(f"Games {self.games_played}: Exploration decayed to {self.dqn_agent.epsilon:.3f}")
        
        # Only update the agent that made the decision
        if self.last_strategy_used == 'dqn':
            self.dqn_agent.remember(state, action_idx, reward, next_state, done)
            self.dqn_agent.replay()
            self.dqn_agent.performance_history.append(reward)
        else:
            info_set = self.cfr_agent._get_info_set(state)
            self.cfr_agent.update(info_set, action_idx, reward)
            self.cfr_agent.performance_history.append(reward)
    
    def save_models(self):
        """Save both models"""
        self.dqn_agent.save()
        self.cfr_agent.save()
    
    def get_stats(self):
        """Return performance statistics (recent window)"""
        # Use last 10 games for performance display
        window = 10
        dqn_recent = list(self.dqn_agent.performance_history)[-window:] if self.dqn_agent.performance_history else []
        cfr_recent = list(self.cfr_agent.performance_history)[-window:] if self.cfr_agent.performance_history else []
        
        return {
            'current_strategy': self.current_strategy,
            'dqn_performance': round(np.mean(dqn_recent), 2) if dqn_recent else 0,
            'cfr_performance': round(np.mean(cfr_recent), 2) if cfr_recent else 0,
            'dqn_epsilon': round(self.dqn_agent.epsilon, 3),
            'total_games': self.games_played
        }

if __name__ == '__main__':
    # Test the system
    game = PokerGame()
    agent = AdaptiveAgent()
    
    print("Poker Backend System Initialized")
    print(f"State vector size: {game.get_state_vector('agent').shape}")
    print(f"Starting chips: {game.starting_chips}")
