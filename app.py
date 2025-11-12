from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from poker_backend import PokerGame, AdaptiveAgent
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# Enable CORS for all routes with all origins
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

game = PokerGame(starting_chips=1000)
agent = AdaptiveAgent()
game_active = False

# Handle OPTIONS requests for CORS preflight
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/new_game', methods=['POST'])
def new_game():
    global game, game_active
    game.reset()
    game.deal_hole_cards()
    game_active = True
    game.players['user']['chips'] -= 10
    game.players['user']['bet'] = 10
    game.players['agent']['chips'] -= 20
    game.players['agent']['bet'] = 20
    game.pot = 30
    game.current_bet = 20
    return jsonify({'success': True, 'game_state': game.to_dict(), 'message': 'New game started!'})

@app.route('/api/game_state', methods=['GET'])
def get_game_state():
    return jsonify({'success': True, 'game_state': game.to_dict(), 'game_active': game_active, 'agent_stats': agent.get_stats()})

@app.route('/api/user_action', methods=['POST'])
def user_action():
    global game_active
    data = request.json
    action = data.get('action')
    amount = data.get('amount', 0)
    
    if not game_active:
        return jsonify({'success': False, 'error': 'No active game'})
    
    valid_actions = game.get_valid_actions('user')
    if action not in valid_actions:
        return jsonify({'success': False, 'error': f'Invalid action. Valid: {valid_actions}'})
    
    game.take_action('user', action, amount)
    
    if game.players['user']['folded']:
        winner, hand_rank = game.determine_winner()
        game.players[winner]['chips'] += game.pot
        game_active = False
        # Learn from winning when user folds
        reward = game.pot  # Positive reward for winning
        state = game.get_state_vector('agent')
        agent.learn(state, 'call', reward, state, True)  # Use 'call' as generic action
        return jsonify({'success': True, 'game_state': game.to_dict(), 'game_over': True, 'winner': winner, 'message': f'{winner.capitalize()} wins ${game.pot}!'})
    
    agent_action, agent_amount, strategy = agent.act(game, 'agent')
    game.take_action('agent', agent_action, agent_amount)
    
    response = {'success': True, 'game_state': game.to_dict(), 'agent_action': agent_action, 'agent_amount': agent_amount, 'agent_strategy': strategy, 'message': f'Agent {agent_action}s using {strategy.upper()}'}
    
    if game.players['agent']['folded']:
        winner, hand_rank = game.determine_winner()
        game.players[winner]['chips'] += game.pot
        game_active = False
        response['game_over'] = True
        response['winner'] = winner
        response['message'] = f'{winner.capitalize()} wins ${game.pot}!'
        # Learn from folding decision
        reward = -game.pot  # Negative reward for folding (lost the pot)
        state = game.get_state_vector('agent')
        agent.learn(state, agent_action, reward, state, True)
        return jsonify(response)
    
    if game.players['user']['bet'] == game.players['agent']['bet']:
        if game.phase == 'preflop':
            game.deal_flop()
            response['message'] += ' - Flop dealt!'
        elif game.phase == 'flop':
            game.deal_turn()
            response['message'] += ' - Turn dealt!'
        elif game.phase == 'turn':
            game.deal_river()
            response['message'] += ' - River dealt!'
        elif game.phase == 'river':
            winner, hand_rank = game.determine_winner()
            # Handle ties
            if winner == 'tie':
                game.players['agent']['chips'] += game.pot // 2
                game.players['user']['chips'] += game.pot // 2
                response['message'] = f'Tie! Pot split ${game.pot // 2} each'
                reward = 0  # Neutral reward for tie
            else:
                game.players[winner]['chips'] += game.pot
                response['message'] = f'Showdown! {winner.capitalize()} wins ${game.pot}!'
                reward = game.pot if winner == 'agent' else -game.pot
            
            game_active = False
            response['game_over'] = True
            response['winner'] = winner
            response['hand_rank'] = hand_rank[0] if hand_rank else None
            state = game.get_state_vector('agent')
            agent.learn(state, agent_action, reward, state, True)
        
        game.players['user']['bet'] = 0
        game.players['agent']['bet'] = 0
        game.current_bet = 0
        response['game_state'] = game.to_dict()
    
    return jsonify(response)

@app.route('/api/skip_to_showdown', methods=['POST'])
def skip_to_showdown():
    global game_active
    if not game_active:
        return jsonify({'success': False, 'error': 'No active game'})
    
    # Deal all remaining community cards
    while len(game.community_cards) < 5:
        if len(game.community_cards) == 0:
            game.deal_flop()
        elif len(game.community_cards) == 3:
            game.deal_turn()
        elif len(game.community_cards) == 4:
            game.deal_river()
    
    # Determine winner
    winner, hand_rank = game.determine_winner()
    
    # Handle winner or tie
    if winner == 'tie':
        # Split pot
        split_amount = game.pot // 2
        game.players['agent']['chips'] += split_amount
        game.players['user']['chips'] += split_amount
        message = f'Tie game! Pot split: ${split_amount} each'
    else:
        game.players[winner]['chips'] += game.pot
        message = f'Showdown! {winner.capitalize()} wins ${game.pot}!'
    
    game_active = False
    return jsonify({
        'success': True, 
        'game_state': game.to_dict(), 
        'game_over': True, 
        'winner': winner, 
        'hand_rank': hand_rank[0] if hand_rank else None, 
        'message': message
    })

@app.route('/api/agent_stats', methods=['GET'])
def get_agent_stats():
    return jsonify({'success': True, 'stats': agent.get_stats()})

@app.route('/api/train', methods=['POST'])
def train():
    import random
    import traceback
    
    try:
        data = request.json if request.json else {}
        episodes = data.get('episodes', 100)
        
        # Save current game state
        global game_active
        saved_game_active = game_active
        
        # Disable fast mode for training - use full AI
        agent.fast_mode = False
        
        results = []
        wins = 0
        losses = 0
        
        for episode in range(episodes):
            try:
                # Reset game for new episode
                game.reset()
                game.deal_hole_cards()
                
                # Track episode state
                episode_states = []
                episode_actions = []
                
                # Play through the game
                game_done = False
                while not game_done:
                    # Get agent state and action
                    state = game.get_state_vector('agent')
                    action, amount, strategy = agent.act(game, 'agent')
                    
                    episode_states.append(state)
                    episode_actions.append(action)
                    
                    # Take agent action
                    game.take_action('agent', action, amount)
                    
                    # Check if agent folded
                    if game.players['agent']['folded']:
                        game_done = True
                        break
                    
                    # Simulate user action
                    user_actions = game.get_valid_actions('user')
                    if user_actions:
                        user_action = random.choice(user_actions)
                        user_amount = 20 if user_action == 'raise' else 0
                        game.take_action('user', user_action, user_amount)
                    
                    # Check if user folded
                    if game.players['user']['folded']:
                        game_done = True
                        break
                    
                    # Progress through phases if bets are matched
                    if game.players['user']['bet'] == game.players['agent']['bet']:
                        if game.phase == 'preflop':
                            game.deal_flop()
                            game.players['user']['bet'] = 0
                            game.players['agent']['bet'] = 0
                            game.current_bet = 0
                        elif game.phase == 'flop':
                            game.deal_turn()
                            game.players['user']['bet'] = 0
                            game.players['agent']['bet'] = 0
                            game.current_bet = 0
                        elif game.phase == 'turn':
                            game.deal_river()
                            game.players['user']['bet'] = 0
                            game.players['agent']['bet'] = 0
                            game.current_bet = 0
                        elif game.phase == 'river':
                            game_done = True
                            break
                
                # Determine winner and reward
                winner, _ = game.determine_winner()
                if winner == 'agent':
                    reward = game.pot
                    wins += 1
                elif winner == 'tie':
                    reward = 0
                else:
                    reward = -game.pot
                    losses += 1
                
                # Learn from the episode
                if episode_states and episode_actions:
                    # Use the last state and action for learning
                    last_state = episode_states[-1]
                    last_action = episode_actions[-1]
                    next_state = game.get_state_vector('agent')
                    agent.learn(last_state, last_action, reward, next_state, True)
                
                results.append({'winner': winner, 'pot': game.pot, 'reward': reward})
                
            except Exception as episode_error:
                print(f"Error in episode {episode}: {str(episode_error)}")
                traceback.print_exc()
                continue
        
        # Save models after training
        agent.save_models()
        
        # Re-enable fast mode after training
        agent.fast_mode = True
        
        # Restore game state
        game_active = saved_game_active
        if not game_active:
            game.reset()
        
        return jsonify({
            'success': True, 
            'episodes': episodes, 
            'wins': wins,
            'losses': losses,
            'results': results[-10:], 
            'stats': agent.get_stats()
        })
        
    except Exception as e:
        # Re-enable fast mode on error
        agent.fast_mode = True
        error_msg = str(e)
        print(f"Training error: {error_msg}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': error_msg}), 500

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    print('=' * 50)
    print('POKER AI SERVER STARTING')
    print('=' * 50)
    print('  ✓ Double DQN Agent')
    print('  ✓ CFR Agent')
    print('  ✓ Adaptive Strategy Selection')
    print('  ✓ Texas Hold\'em Engine')
    print('=' * 50)
    app.run(debug=True, port=5000)
