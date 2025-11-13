// Poker AI Frontend JavaScript

const API_BASE = '/api';  // Use relative URL to avoid CORS issues

// Card symbols mapping
const CARD_SYMBOLS = {
    '♥': { color: 'red', symbol: '♥' },
    '♦': { color: 'red', symbol: '♦' },
    '♣': { color: 'black', symbol: '♣' },
    '♠': { color: 'black', symbol: '♠' }
};

// Game state
let gameActive = false;

// DOM Elements
const btnNewGame = document.getElementById('btn-new-game');
const btnFold = document.getElementById('btn-fold');
const btnCheck = document.getElementById('btn-check');
const btnCall = document.getElementById('btn-call');
const btnRaise = document.getElementById('btn-raise');
const btnConfirmRaise = document.getElementById('btn-confirm-raise');
const btnSkip = document.getElementById('btn-skip');
const btnTrain = document.getElementById('btn-train');
const raiseControls = document.getElementById('raise-controls');
const raiseAmountInput = document.getElementById('raise-amount');
const gameMessage = document.getElementById('game-message');
const loadingOverlay = document.getElementById('loading-overlay');

// Loading functions
function showLoading() {
    if (loadingOverlay) loadingOverlay.style.display = 'flex';
    // Disable all buttons during loading
    disableAllButtons();
}

function hideLoading() {
    if (loadingOverlay) loadingOverlay.style.display = 'none';
}

function disableAllButtons() {
    btnNewGame.disabled = true;
    btnFold.disabled = true;
    btnCheck.disabled = true;
    btnCall.disabled = true;
    btnRaise.disabled = true;
    // Don't disable confirm raise if raise controls are showing
    if (raiseControls.style.display !== 'flex') {
        btnConfirmRaise.disabled = true;
    }
    btnSkip.disabled = true;
    btnTrain.disabled = true;
}

function enableNewGameButton() {
    btnNewGame.disabled = false;
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updateAgentStats();
    setInterval(updateAgentStats, 5000); // Update stats every 5 seconds
    enableNewGameButton(); // Enable new game button on load
});

// Event Listeners
function setupEventListeners() {
    btnNewGame.addEventListener('click', startNewGame);
    btnFold.addEventListener('click', () => takeAction('fold'));
    btnCheck.addEventListener('click', () => takeAction('check'));
    btnCall.addEventListener('click', () => takeAction('call'));
    btnRaise.addEventListener('click', showRaiseControls);
    btnConfirmRaise.addEventListener('click', confirmRaise);
    btnSkip.addEventListener('click', skipToShowdown);
    btnTrain.addEventListener('click', trainAgent);
}

// API Calls
async function startNewGame() {
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/new_game`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            gameActive = true;
            updateUI(data.game_state);
            updateMessage(data.message);
            enableActionButtons(data.game_state);
            btnSkip.disabled = false;
        }
    } catch (error) {
        console.error('Error starting new game:', error);
        updateMessage('Error starting game. Please try again.');
        enableNewGameButton();
    } finally {
        hideLoading();
        enableNewGameButton();
        btnTrain.disabled = false;
    }
}

async function takeAction(action, amount = 0) {
    if (!gameActive) {
        console.log('Game not active, ignoring action');
        return;
    }
    
    console.log(`Taking action: ${action}, amount: ${amount}`);
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/user_action`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, amount })
        });
        const data = await response.json();
        console.log('Response from user_action:', data);
        
        if (data.success) {
            updateUI(data.game_state, data.game_over);
            updateMessage(data.message);
            
            if (data.game_over) {
                console.log('Game over detected');
                gameActive = false;
                disableActionButtons();
                showWinner(data.winner, data.message);
                // Re-enable New Game button after game ends
                enableNewGameButton();
                btnTrain.disabled = false;
            } else {
                console.log('Enabling action buttons with game state:', data.game_state);
                enableActionButtons(data.game_state);
            }
            
            // Update agent stats
            updateAgentStats();
        } else {
            console.error('Action failed:', data.error);
            updateMessage(data.error || 'Invalid action');
            // Re-enable action buttons on error
            if (gameActive && data.game_state) {
                enableActionButtons(data.game_state);
            } else {
                // Fallback: enable all action buttons
                btnFold.disabled = false;
                btnCheck.disabled = false;
                btnCall.disabled = false;
                btnRaise.disabled = false;
            }
        }
    } catch (error) {
        console.error('Error taking action:', error);
        updateMessage('Error processing action. Please try again.');
        // Enable all action buttons on error so user can try again
        btnFold.disabled = false;
        btnCheck.disabled = false;
        btnCall.disabled = false;
        btnRaise.disabled = false;
    } finally {
        hideLoading();
        console.log('Button states after action:', {
            fold: btnFold.disabled,
            check: btnCheck.disabled,
            call: btnCall.disabled,
            raise: btnRaise.disabled
        });
    }
}

async function skipToShowdown() {
    if (!gameActive) return;
    
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/skip_to_showdown`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            gameActive = false;
            updateUI(data.game_state, true);  // Reveal agent cards
            disableActionButtons();
            showWinner(data.winner, data.message);
            // Re-enable New Game button after skip to showdown
            enableNewGameButton();
            btnTrain.disabled = false;
        }
    } catch (error) {
        console.error('Error skipping to showdown:', error);
    } finally {
        hideLoading();
    }
}

async function trainAgent() {
    btnTrain.disabled = true;
    showLoading();
    updateMessage('Training agent for 5 episodes... Please wait.');
    
    try {
        const response = await fetch(`${API_BASE}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ episodes: 5 })
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            updateMessage(`Training complete! ${data.wins} wins, ${data.losses} losses out of ${data.episodes} episodes.`);
            updateAgentStats();
            // Enable new game button after training
            enableNewGameButton();
        } else {
            updateMessage(`Training failed: ${data.error || 'Unknown error'}`);
            enableNewGameButton();
        }
    } catch (error) {
        console.error('Error training agent:', error);
        updateMessage(`Training failed: ${error.message}`);
        enableNewGameButton();
    } finally {
        hideLoading();
        btnTrain.disabled = false;
    }
}

async function updateAgentStats() {
    try {
        const response = await fetch(`${API_BASE}/agent_stats`);
        const data = await response.json();
        
        if (data.success) {
            const stats = data.stats;
            document.getElementById('current-strategy').textContent = stats.current_strategy.toUpperCase();
            document.getElementById('dqn-perf').textContent = stats.dqn_performance.toFixed(2);
            document.getElementById('cfr-perf').textContent = stats.cfr_performance.toFixed(2);
            document.getElementById('epsilon').textContent = stats.dqn_epsilon.toFixed(3);
        }
    } catch (error) {
        console.error('Error fetching agent stats:', error);
    }
}

// UI Updates
function updateUI(gameState, revealAgentCards = false) {
    // Update chips
    document.getElementById('agent-chips').textContent = gameState.agent.chips;
    document.getElementById('user-chips').textContent = gameState.user.chips;
    
    // Update bets
    document.getElementById('agent-bet').textContent = gameState.agent.bet;
    document.getElementById('user-bet').textContent = gameState.user.bet;
    
    // Update pot
    document.getElementById('pot-amount').textContent = gameState.pot;
    
    // Update cards - only show agent cards at showdown
    updateCards('agent', gameState.agent.hand, revealAgentCards);
    updateCards('user', gameState.user.hand, true);
    updateCommunityCards(gameState.community_cards);
    
    // Update hand descriptions
    updateHandDescription('user', gameState.user.hand_description);
    if (revealAgentCards) {
        updateHandDescription('agent', gameState.agent.hand_description);
    } else {
        updateHandDescription('agent', '');
    }
    
    // Update player status
    updatePlayerStatus('agent', gameState.agent.folded);
    updatePlayerStatus('user', gameState.user.folded);
}

function updateCards(player, cards, showCards) {
    const container = document.getElementById(`${player}-cards`);
    container.innerHTML = '';
    
    if (showCards && cards.length > 0) {
        cards.forEach(card => {
            const cardDiv = createCard(card);
            container.appendChild(cardDiv);
        });
    } else {
        // Show card backs using API image
        for (let i = 0; i < 2; i++) {
            const cardDiv = document.createElement('div');
            cardDiv.className = 'card card-back dealing';
            const img = document.createElement('img');
            img.src = 'https://deckofcardsapi.com/static/img/back.png';
            img.alt = 'Card back';
            img.style.width = '100%';
            img.style.height = '100%';
            img.style.objectFit = 'contain';
            cardDiv.appendChild(img);
            container.appendChild(cardDiv);
        }
    }
}

function updateCommunityCards(cards) {
    const container = document.getElementById('community-cards-container');
    container.innerHTML = '';
    
    for (let i = 0; i < 5; i++) {
        if (i < cards.length) {
            const cardDiv = createCard(cards[i]);
            cardDiv.classList.add('dealing');
            container.appendChild(cardDiv);
        } else {
            const cardDiv = document.createElement('div');
            cardDiv.className = 'card card-back';
            const img = document.createElement('img');
            img.src = 'https://deckofcardsapi.com/static/img/back.png';
            img.alt = 'Card back';
            img.style.width = '100%';
            img.style.height = '100%';
            img.style.objectFit = 'contain';
            cardDiv.appendChild(img);
            container.appendChild(cardDiv);
        }
    }
}

function createCard(card) {
    const cardDiv = document.createElement('div');
    
    // Check if card has an image URL from the API
    if (card.image && card.image !== 'https://deckofcardsapi.com/static/img/back.png') {
        // Use API image
        cardDiv.className = 'card card-image';
        const img = document.createElement('img');
        img.src = card.image;
        img.alt = `${card.rank}${card.suit}`;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'contain';
        cardDiv.appendChild(img);
    } else {
        // Fallback to text display
        const colorClass = CARD_SYMBOLS[card.suit].color === 'red' ? 'card-red' : 'card-black';
        cardDiv.className = `card ${colorClass}`;
        cardDiv.textContent = `${card.rank}${card.suit}`;
    }
    
    return cardDiv;
}

function updatePlayerStatus(player, folded) {
    const statusDiv = document.getElementById(`${player}-status`);
    if (folded) {
        statusDiv.textContent = 'FOLDED';
        statusDiv.className = 'player-status folded';
    } else {
        statusDiv.textContent = '';
        statusDiv.className = 'player-status';
    }
}

function updateHandDescription(player, description) {
    const descDiv = document.getElementById(`${player}-hand-description`);
    if (description && description.trim()) {
        descDiv.textContent = description;
        descDiv.style.display = 'block';
    } else {
        descDiv.textContent = '';
        descDiv.style.display = 'none';
    }
}

function updateMessage(message) {
    gameMessage.textContent = message;
}

function showWinner(winner, message) {
    updateMessage(message);
    
    // Highlight winner (unless it's a tie)
    if (winner !== 'tie') {
        const winnerStatus = document.getElementById(`${winner}-status`);
        if (winnerStatus) {
            winnerStatus.textContent = 'WINNER! 🏆';
            winnerStatus.className = 'player-status winner';
        }
    }
    // Cards are already revealed by updateUI with revealAgentCards=true
}

// Action Button Controls
function enableActionButtons(gameState) {
    console.log('enableActionButtons called with gameState:', gameState);
    const validActions = getValidActions(gameState);
    console.log('Valid actions:', validActions);
    
    btnFold.disabled = !validActions.includes('fold');
    btnCheck.disabled = !validActions.includes('check');
    btnCall.disabled = !validActions.includes('call');
    btnRaise.disabled = !validActions.includes('raise');
    
    // Update call button text with amount
    if (validActions.includes('call')) {
        const toCall = gameState.current_bet - gameState.user.bet;
        btnCall.textContent = `CALL $${toCall}`;
    }
    
    console.log('Buttons after enable:', {
        fold: btnFold.disabled,
        check: btnCheck.disabled,
        call: btnCall.disabled,
        raise: btnRaise.disabled
    });
}

function disableActionButtons() {
    btnFold.disabled = true;
    btnCheck.disabled = true;
    btnCall.disabled = true;
    btnRaise.disabled = true;
}

function getValidActions(gameState) {
    const actions = [];
    const toCall = gameState.current_bet - gameState.user.bet;
    
    if (toCall === 0) {
        actions.push('check');
    } else if (gameState.user.chips >= toCall) {
        actions.push('call');
    }
    
    if (gameState.user.chips > toCall) {
        actions.push('raise');
    }
    
    actions.push('fold');
    return actions;
}

function showRaiseControls() {
    raiseControls.style.display = 'flex';
    btnRaise.style.display = 'none';
    // Don't disable the confirm raise button
    btnConfirmRaise.disabled = false;
}

function confirmRaise() {
    const amount = parseInt(raiseAmountInput.value);
    if (isNaN(amount) || amount < 20) {
        alert('Please enter a valid raise amount (minimum $20)');
        return;
    }
    raiseControls.style.display = 'none';
    btnRaise.style.display = 'inline-block';
    takeAction('raise', amount);
}
