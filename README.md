# 🎰 Poker AI Challenge# 🎰 Poker AI - Adaptive Strategy System



A Texas Hold'em poker game featuring an adaptive AI agent that switches between Double DQN and CFR (Counterfactual Regret Minimization) strategies. Built with Flask backend and vanilla JavaScript frontend with a retro gaming aesthetic.A sophisticated Texas Hold'em poker game featuring an AI agent that adaptively switches between **Double DQN** and **CFR** strategies based on performance.



![Python](https://img.shields.io/badge/Python-3.8+-blue)## 🎯 Features

![Flask](https://img.shields.io/badge/Flask-3.0-green)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)### Backend AI Strategies

- **Double DQN Agent**: Deep reinforcement learning with dueling architecture

## 🎮 Features  - Experience replay buffer

  - Target network for stable training

- **Adaptive AI Agent**: Automatically switches between DQN and CFR strategies based on performance  - Epsilon-greedy exploration

- **Double DQN**: Deep Q-Network with experience replay and target network  

- **CFR Algorithm**: Counterfactual Regret Minimization for Nash equilibrium strategy- **CFR (Counterfactual Regret Minimization)**: Game theory optimal strategy

- **Full Texas Hold'em**: Complete poker game with all phases (preflop, flop, turn, river)  - Regret-based learning

- **Real-time Training**: Train the AI for 100 episodes and watch it improve  - Nash equilibrium convergence

- **Retro Gaming UI**: Pixel-perfect buttons with casino atmosphere and smooth animations  

- **Exploration Decay**: AI exploration rate decreases every 10 games for better exploitation- **Adaptive Selection**: Automatically switches to best-performing strategy

- **Performance Tracking**: Real-time stats showing AI performance and strategy selection  - Performance tracking over 20-hand windows

  - 10% threshold for strategy switching

## 🚀 Quick Start  - Real-time strategy display



### Prerequisites### Game Engine

- Complete Texas Hold'em implementation

- Python 3.8 or higher- Hand evaluation (High Card → Straight Flush)

- pip (Python package manager)- Betting rounds (Preflop, Flop, Turn, River)

- Pot management

### Installation- Chip tracking



1. **Clone or download this repository**### Frontend UI

- Visual poker table with green felt design

```bash- Animated card dealing

cd AceAI- Real-time chip and bet updates

```- Dealer indicator

- Community card display

2. **Create a virtual environment (recommended)**- Action buttons (Fold, Check/Call, Raise)

- Performance statistics dashboard

```bash

# Windows## 📁 Project Structure

python -m venv venv

venv\Scripts\activate```

AceAI/

# macOS/Linux├── poker_backend.py      # Core game engine + AI agents

python3 -m venv venv├── app.py               # Flask API server

source venv/bin/activate├── templates/

```│   └── index.html       # Frontend HTML

├── static/

3. **Install dependencies**│   ├── style.css        # Poker table styling

│   └── app.js          # Frontend JavaScript

```bash├── requirements_new.txt # Python dependencies

pip install -r requirements.txt└── README.md           # This file

``````



### Running Locally## 🚀 Installation



1. **Start the Flask server**### Prerequisites

- Python 3.11 or 3.12

```bash- pip package manager

python app.py

```### Steps



You should see:1. **Install dependencies**:

``````powershell

==================================================pip install -r requirements_new.txt

POKER AI SERVER STARTING```

==================================================

  ✓ Double DQN AgentThis installs:

  ✓ CFR Agent- Flask 3.0.0 (Web framework)

  ✓ Adaptive Strategy Selection- Flask-CORS 4.0.0 (Cross-origin support)

  ✓ Texas Hold'em Engine- TensorFlow 2.20.0 (Deep learning)

==================================================- NumPy 1.24+ (Numerical computing)

 * Running on http://127.0.0.1:5000

```2. **Verify backend works**:

```powershell

2. **Open your browser**python poker_backend.py

```

Navigate to: `http://localhost:5000`Should output:

```

3. **Play poker!**Poker Backend System Initialized

   - Click "NEW GAME" to startState vector size: (219,)

   - Use FOLD, CHECK, CALL, RAISE buttons to playStarting chips: 1000

   - Click "TRAIN 100 EPISODES" to improve the AI```



## 🌐 Deploying Online3. **Start the server**:

```powershell

### Option 1: Render (Recommended - Free Tier Available)python app.py

```

1. **Add gunicorn to requirements.txt**You should see:

```

```bash==================================================

echo "gunicorn==21.2.0" >> requirements.txtPOKER AI SERVER STARTING

```==================================================

Features:

2. **Push to GitHub**  ✓ Double DQN Agent

  ✓ CFR Agent

```bash  ✓ Adaptive Strategy Selection

git init  ✓ Texas Hold'em Engine

git add .==================================================

git commit -m "Initial commit" * Running on http://127.0.0.1:5000

git remote add origin <your-github-repo-url>```

git push -u origin main

```4. **Open browser**:

Navigate to: http://localhost:5000

3. **Deploy on Render**

   - Go to [render.com](https://render.com)## 🎮 How to Play

   - Click "New +" → "Web Service"

   - Connect your GitHub repository1. **Start Game**: Click "🎮 New Game" button

   - Render will auto-detect Flask settings:2. **View Cards**: You'll see your hole cards (agent's are hidden)

     - **Build Command**: `pip install -r requirements.txt`3. **Take Actions**:

     - **Start Command**: `gunicorn app:app`   - **Fold**: Give up the hand

   - **Check**: Pass (when no bet to match)

4. **Your site will be live at**: `https://your-app-name.onrender.com`   - **Call**: Match the current bet

   - **Raise**: Increase the bet (enter amount)

### Option 2: Railway4. **Watch AI**: Agent responds using best strategy (DQN or CFR)

5. **Progress**: Game advances through Flop → Turn → River

1. **Push to GitHub** (same as above)6. **Showdown**: Best hand wins the pot!



2. **Deploy on Railway**## 📊 Dashboard Stats

   - Go to [railway.app](https://railway.app)

   - Click "New Project" → "Deploy from GitHub repo"The top panel shows:

   - Select your repository- **Current Strategy**: Which AI strategy is active (DQN/CFR)

   - Railway auto-detects Python and Flask- **DQN Performance**: Average reward per hand

   - Your site will be live in minutes!- **CFR Performance**: Average reward per hand

- **Exploration**: DQN epsilon value (decreases as it learns)

### Option 3: Heroku

## 🎓 Training the AI

1. **Create a `Procfile`**

### In-Browser Training

```bashClick "🎓 Train 100 Episodes" button to:

echo "web: gunicorn app:app" > Procfile- Run 100 simulated games

```- Update both DQN and CFR strategies

- Save trained models automatically

2. **Add gunicorn**

### Training Indicator

```bashWatch the stats panel - as training progresses:

echo "gunicorn==21.2.0" >> requirements.txt- Performance values increase

```- Exploration (epsilon) decreases

- Strategy may switch to better performer

3. **Deploy**

## 🔧 Advanced Features

```bash

heroku login### Developer Controls

heroku create your-poker-ai-app- **⏩ Skip to Showdown**: Instantly deal all community cards

git push heroku main- **🎓 Train**: Run training episodes

heroku open

```### API Endpoints



### Option 4: PythonAnywhere```

POST /api/new_game          # Start new game

1. Go to [pythonanywhere.com](https://www.pythonanywhere.com)POST /api/user_action       # Take action (fold/check/call/raise)

2. Upload your files via "Files" tabGET  /api/game_state        # Get current state

3. Create a new web app (Flask)GET  /api/agent_stats       # Get AI performance stats

4. Set working directory to your project folderPOST /api/skip_to_showdown  # Skip to end

5. Configure WSGI file to import from `app.py`POST /api/train             # Train AI agents

POST /api/save_models       # Save trained models

## 📁 Project Structure```



```### State Vector (Size: 219)

AceAI/- Player hole cards (one-hot): 34 features

├── app.py                  # Flask backend API server- Community cards (one-hot): 85 features (5 × 17)

├── poker_backend.py        # Game engine, AI agents, card logic- Chip stacks: 2 features

├── requirements.txt        # Python dependencies- Pot size: 1 feature

├── .gitignore             # Git ignore rules- Current bet: 1 feature

├── README.md              # This file- Game phase: 4 features (one-hot)

├── templates/

│   └── index.html         # Main game page (retro UI)## 🧠 AI Architecture Details

├── static/

│   ├── app.js            # Frontend JavaScript### Double DQN

│   └── style.css         # Casino-themed styles```

└── models/ (auto-generated)Input (219) → Dense(256) → Dropout(0.2) → Dense(128) → Dropout(0.2)

    ├── dqn_model.keras   # Trained DQN neural network           ↓

    └── cfr_strategy.pkl  # CFR regret/strategy tables    Value Stream: Dense(64) → Dense(1)

```    Advantage Stream: Dense(64) → Dense(4)

           ↓

## 🎯 How It Works    Q-values = Value + (Advantage - mean(Advantage))

```

### AI Strategy Selection

**Hyperparameters**:

The adaptive agent tracks performance over a 20-game window and automatically switches between:- Learning rate: 0.001

- Discount (γ): 0.99

- **DQN (Deep Q-Network)**: Neural network that learns optimal actions through experience- Replay buffer: 10,000 experiences

- **CFR (Counterfactual Regret)**: Game theory approach that converges to Nash equilibrium- Batch size: 32

- Target update: Every 100 steps

When one strategy consistently outperforms the other by 10%, the agent switches.- Epsilon decay: 0.995



### Training Process### CFR Strategy

- Regret minimization on information sets

1. Agent plays 100 simulated games against a random opponent- Action selection via regret matching

2. Learns from wins/losses using reinforcement learning- Strategy averaging for Nash equilibrium

3. Updates neural network weights (DQN) or regret values (CFR)

4. Exploration rate decays every 10 games (30% → 1% over time)## 📈 Performance Tracking

5. Models are saved automatically after training

Models are saved automatically:

### Game Flow- `dqn_model.h5` - TensorFlow model

- `cfr_strategy.pkl` - Pickle file with regret/strategy sums

```

New Game → Deal Hole Cards → Small/Big Blinds Posted## 🐛 Troubleshooting

    ↓

Preflop Betting Round### Port Already in Use

    ↓```powershell

Flop (3 community cards) → Betting Round# Kill process on port 5000

    ↓netstat -ano | findstr :5000

Turn (4th card) → Betting Roundtaskkill /PID <PID> /F

    ↓```

River (5th card) → Final Betting Round

    ↓### TensorFlow Warnings

Showdown → Best 5-Card Hand WinsThese are normal:

```- "Could not load dynamic library cudart64_*.dll" (no GPU needed)

- Onednn optimization messages

## 🛠️ Configuration

### Cards Not Showing

### Environment Variables (for deployment)- Hard refresh: Ctrl + Shift + R

- Check browser console (F12) for errors

Create a `.env` file (optional):

## 🎯 Strategy Tips

```bash

FLASK_ENV=production- **Early Position**: Agent tends to be more conservative

PORT=5000- **Good Hands**: Agent raises more with strong starting hands

```- **DQN Strategy**: More exploratory, tries different plays

- **CFR Strategy**: More game-theory optimal, balanced play

### Customization

## 📝 Future Enhancements

Edit these values in `poker_backend.py`:

- [ ] Multi-player support (3+ players)

```python- [ ] Tournament mode

# Starting chips per player- [ ] Hand history replay

starting_chips = 1000- [ ] Advanced statistics (VPIP, PFR, etc.)

- [ ] Monte Carlo Tree Search (MCTS) strategy

# DQN exploration parameters- [ ] Opponent modeling

epsilon = 0.3           # Initial exploration rate (30%)- [ ] Bluff detection

epsilon_min = 0.01      # Minimum exploration (1%)

epsilon_decay = 0.995   # Decay multiplier## 🤝 Credits



# Training settings- **Game Engine**: Custom Texas Hold'em implementation

exploration_decay_interval = 10  # Decay every N games- **AI Algorithms**: Double DQN (Hasselt et al.), CFR (Zinkevich et al.)

performance_window = 20          # Games to track for strategy switch- **Framework**: Flask + TensorFlow + Vanilla JavaScript

switch_threshold = 0.1           # 10% performance difference to switch

```## 📄 License



## 📊 Performance StatsMIT License - Free to use and modify



The UI displays real-time statistics:---



- **Current Strategy**: Active AI strategy (DQN or CFR)**Enjoy playing against the adaptive AI!** 🃏🤖

- **DQN Performance**: Average reward over last 10 games
- **CFR Performance**: Average reward over last 10 games  
- **Exploration**: Current exploration rate (epsilon value)

Watch these values change as you play and train the AI!

## 🐛 Troubleshooting

### "Module not found" errors

```bash
pip install -r requirements.txt --upgrade
```

### Port already in use

Change the port in `app.py`:

```python
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change to any available port
```

### TensorFlow warnings

CPU warnings like "Could not load dynamic library" are normal and can be ignored. The game works perfectly with CPU.

### Training button not responding

1. Check browser console (F12) for errors
2. Ensure Flask server is running
3. Try refreshing the page
4. Check that fetch URLs are correct

### Cards not displaying

- Hard refresh: `Ctrl + Shift + R` (Windows) or `Cmd + Shift + R` (Mac)
- Clear browser cache
- Check browser console for errors

## 🎨 UI Features

- ✅ Retro gaming fonts (Press Start 2P, Bebas Neue)
- ✅ 3D button effects with press animation
- ✅ Casino-themed dark background with glows
- ✅ Smooth card dealing animations
- ✅ Noise texture overlay for authenticity
- ✅ Vignette lighting effect
- ✅ Responsive poker table layout
- ✅ Real-time chip and pot updates
- ✅ Hand strength indicators

## 🧠 AI Technical Details

### Double DQN Architecture

```
Input Layer (219 features)
    ↓
Dense(256, relu) + Dropout(0.2)
    ↓
Dense(128, relu) + Dropout(0.2)
    ↓
Split into two streams:
    ├─ Value Stream: Dense(64) → Dense(1)
    └─ Advantage Stream: Dense(64) → Dense(4)
    ↓
Q(s,a) = V(s) + [A(s,a) - mean(A(s,·))]
```

**Hyperparameters**:
- Learning rate: 0.001 (Adam optimizer)
- Discount factor (γ): 0.99
- Replay buffer: 10,000 experiences
- Batch size: 32
- Target network update: Every 100 steps
- Epsilon decay: 0.995 per game

### CFR Strategy

- Information set abstraction based on hand strength
- Regret matching for action selection
- Cumulative regret and strategy tracking
- Nash equilibrium convergence

### State Vector (219 dimensions)

- Hole cards (one-hot encoded): 34 features
- Community cards (one-hot encoded): 85 features (5 cards × 17 values)
- Player chips: 2 features
- Pot size: 1 feature
- Current bet: 1 feature
- Game phase: 4 features (preflop/flop/turn/river)
- Hand evaluation: 8 features
- Pot odds: 1 feature
- Win probability: 1 feature

## 📝 API Endpoints

```
POST /api/new_game          # Start a new game
POST /api/user_action       # Player action (fold/check/call/raise)
GET  /api/game_state        # Get current game state
GET  /api/agent_stats       # Get AI performance statistics
POST /api/skip_to_showdown  # Skip to final cards (dev mode)
POST /api/train             # Train AI for N episodes
```

## 🔮 Future Enhancements

- [ ] Multiplayer support (3-6 players)
- [ ] Tournament mode with blind increases
- [ ] Hand history and replay system
- [ ] Advanced statistics dashboard (VPIP, PFR, etc.)
- [ ] Mobile responsive design
- [ ] Different AI difficulty levels
- [ ] Monte Carlo Tree Search (MCTS) agent
- [ ] Opponent modeling and exploitation
- [ ] Range analysis visualization

## 📚 Learning Resources

### Poker Theory
- [Poker Hand Rankings](https://www.pokerstars.com/poker/games/rules/hand-rankings/)
- [Texas Hold'em Rules](https://www.pokernews.com/poker-rules/texas-holdem.htm)

### AI/ML Concepts
- [Deep Q-Learning](https://arxiv.org/abs/1312.5602)
- [Double DQN](https://arxiv.org/abs/1509.06461)
- [CFR Algorithm](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf)

## 💡 Credits

Built with:
- **Flask** - Python web framework
- **TensorFlow/Keras** - Deep learning
- **NumPy** - Numerical computing
- **Vanilla JavaScript** - Frontend interactivity
- **Google Fonts** - Press Start 2P & Bebas Neue

## 📞 Support

Having issues? Try these steps:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review browser console (F12) for errors
3. Ensure all dependencies are installed
4. Check Flask server logs
5. Open an issue on GitHub with details

## 📄 License

MIT License - Free to use, modify, and distribute.

---

**Ready to play? Start the server and challenge the AI! 🎰♠️♥️♣️♦️**

Good luck at the tables! 🃏🤖
