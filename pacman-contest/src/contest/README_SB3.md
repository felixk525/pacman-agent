# Pacman Capture-the-Flag with OpenAI Gym & Stable-Baselines3

This directory contains implementations for training Pacman agents using modern RL libraries.

## Files Overview

1. **`pacman_gym_env.py`** - OpenAI Gym wrapper for Pacman environment
2. **`my_team_sb3.py`** - Agent using Stable-Baselines3 (PPO/A2C/DQN)
3. **`train_sb3.py`** - Standalone training script
4. **`my_team_RL.py`** - Custom Q-Learning implementation

## Installation

```bash
# Install dependencies
pip install gymnasium
pip install stable-baselines3
pip install numpy

# Or install all at once
pip install gymnasium stable-baselines3 numpy torch
```

## Usage

### Option 1: Train with Stable-Baselines3 (Recommended)

Train using PPO (best for continuous learning):
```bash
python capture.py -r my_team_sb3 -b baseline_team -n 100 -x 100 -q
```

Train using A2C (faster, less stable):
```bash
# Modify my_team_sb3.py line 34: algorithm='A2C'
python capture.py -r my_team_sb3 -b baseline_team -n 100 -x 100 -q
```

Train using DQN (discrete actions, good for Pacman):
```bash
# Modify my_team_sb3.py line 34: algorithm='DQN'
python capture.py -r my_team_sb3 -b baseline_team -n 100 -x 100 -q
```

### Option 2: Custom Q-Learning

```bash
python capture.py -r my_team_RL -b baseline_team -n 100 -x 100 -q
```

### Option 3: Evaluate Trained Agent

```bash
# Run without training (-x flag)
python capture.py -r my_team_sb3 -b baseline_team -n 10
```

## Command Line Arguments

- `-r my_team_sb3` - Red team (your agents)
- `-b baseline_team` - Blue team (opponents)
- `-n 100` - Number of games to play
- `-x 100` - Number of training games (enables learning)
- `-q` - Quiet mode (no graphics, faster training)

## Features

### Gym Environment (`pacman_gym_env.py`)

**Observation Space (8 features):**
1. Bias (always 1.0)
2. Distance to nearest food
3. Food remaining count
4. Distance to nearest ghost
5. Food being carried
6. Distance to home
7. Distance to nearest capsule
8. Current score

**Action Space:**
- 0: North
- 1: South
- 2: East
- 3: West
- 4: Stop

**Reward Structure:**
- +100 per food eaten
- +200 per food returned
- -150 for dying with food
- -50 for dying without food
- +2 per step closer to food
- -(6-dist)² penalty for being near ghosts
- +5 for entering enemy territory
- +3×carrying for moving toward home when loaded

### SB3 Agent (`my_team_sb3.py`)

**Algorithms Available:**
- **PPO** (Proximal Policy Optimization) - Most stable, recommended
- **A2C** (Advantage Actor-Critic) - Faster training
- **DQN** (Deep Q-Network) - Good for discrete actions

**Features:**
- Automatic model saving/loading
- Training progress tracking
- Epsilon-greedy exploration (for DQN)
- Policy gradient learning (for PPO/A2C)

## Training Tips

1. **Start with PPO**: Most stable and robust
2. **Train for 100+ episodes**: Agent needs time to learn
3. **Monitor rewards**: Check if they're increasing over time
4. **Adjust hyperparameters**: In `my_team_sb3.py` if needed
5. **Use quiet mode**: `-q` flag for faster training

## Hyperparameter Tuning

Edit `my_team_sb3.py` to adjust:

**PPO:**
```python
PPO(
    'MlpPolicy',
    self.env,
    learning_rate=3e-4,    # Try 1e-4 to 1e-3
    n_steps=2048,          # Try 1024 to 4096
    batch_size=64,         # Try 32 to 128
    gamma=0.99,            # Discount factor
    clip_range=0.2,        # PPO clip range
)
```

**A2C:**
```python
A2C(
    'MlpPolicy',
    self.env,
    learning_rate=7e-4,    # Try 1e-4 to 1e-3
    n_steps=5,             # Try 5 to 20
    gamma=0.99,
)
```

**DQN:**
```python
DQN(
    'MlpPolicy',
    self.env,
    learning_rate=1e-4,    # Try 1e-5 to 1e-3
    buffer_size=10000,     # Try 5000 to 50000
    batch_size=32,         # Try 32 to 128
    exploration_fraction=0.1,
)
```

## Monitoring Training

The agent prints progress every 10 episodes:
```
Agent 0 - Episode 10/100
Score: 5
Model saved to sb3_model_PPO_agent_0.zip

Agent 0 - Episode 20/100
Score: 8
Model saved to sb3_model_PPO_agent_0.zip
```

## File Outputs

Training generates:
- `sb3_model_PPO_agent_0.zip` - Trained PPO model
- `sb3_model_A2C_agent_0.zip` - Trained A2C model
- `sb3_model_DQN_agent_0.zip` - Trained DQN model
- `q_weights_agent_0.pkl` - Q-Learning weights (if using my_team_RL)

## Advantages of SB3 over Custom Q-Learning

✅ **Better algorithms**: PPO, A2C are state-of-the-art  
✅ **Neural networks**: More expressive than linear functions  
✅ **Tested & optimized**: Industry-standard implementations  
✅ **Less code**: No need to implement learning algorithm  
✅ **TensorBoard support**: Built-in visualization  
✅ **Faster convergence**: Better exploration strategies  

## Troubleshooting

**Import Error: stable_baselines3 not found**
```bash
pip install stable-baselines3
```

**CUDA/GPU errors:**
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
```

**Agent not learning:**
- Increase number of episodes (`-x 200` or more)
- Adjust learning rate (lower if unstable, higher if slow)
- Check reward signal (should vary, not always 0)

**Training too slow:**
- Use `-q` flag for quiet mode
- Reduce batch size
- Use A2C instead of PPO

## Example Training Session

```bash
# 1. Delete old models
rm sb3_model_*.zip

# 2. Train for 200 episodes with PPO
python capture.py -r my_team_sb3 -b baseline_team -n 200 -x 200 -q

# 3. Evaluate performance (10 games with graphics)
python capture.py -r my_team_sb3 -b baseline_team -n 10

# 4. Continue training for 100 more episodes
python capture.py -r my_team_sb3 -b baseline_team -n 100 -x 100 -q
```

## Next Steps

1. ✅ Install stable-baselines3
2. ✅ Run initial training (100 episodes)
3. ✅ Monitor performance
4. ✅ Tune hyperparameters
5. ✅ Train longer (500+ episodes)
6. ✅ Test against different opponents

Good luck training your agent! 🎮🤖
