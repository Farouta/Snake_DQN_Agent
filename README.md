# Snake DQN Agent

## Overview
This project is a reinforcement learning agent that learns to play Snake using Deep Q-Networks (DQN). It started with a simple MLP and evolved into a CNN-based approach to better capture the spatial structure of the game board. The project is still actively being developed — I'm continuously experimenting with different architectures and training strategies.

## Training Results

### MLP Agent — 10,000 Games on a 5×5 Board
![MLP Score Graph](assets/score_graph_mlpp.png)

### CNN Agent — 10,000 Games on a 5×5 Board
![CNN Score Graph](assets/score_graph_cnn.png)

### CNN Training Dashboard
![CNN Training Dashboard](assets/graphs_cnn_no_gradient.png)

The dashboard tracks average score, loss, death reason ratio (wall vs body collision), and average game length across training batches of 500 games.

## Architecture Evolution

### 1. Baseline (MLP)
I started with a 3-layer fully connected network. The board state is flattened into a 1D vector and passed through hidden layers with ReLU activations. This got the agent to an average score of ~9.5 on a 5×5 board after 10,000 games.

### 2. Spatial Awareness (CNN)
To better capture the 2D spatial relationships on the board (where the snake head is relative to the fruit, where the body is, etc.), I switched to a 5-layer CNN. This preserves the grid structure instead of flattening it, and reached a slightly higher average score of ~10 with more stable learning.

The CNN doesn't flatten the state until after the convolutional layers:
```
Input (W×H×4) → 5× [Conv2d → ReLU] → Flatten → Linear → ReLU → Linear → Q-values (3)
```

## State Representation
The board is encoded as a `W × H × 4` tensor using one-hot encoding:

| Channel | Meaning    |
|---------|------------|
| 0       | Empty cell |
| 1       | Snake head |
| 2       | Fruit      |
| 3       | Snake body |

## Action Space
The agent chooses from 3 relative actions: go straight, turn right, or turn left. Using relative rather than absolute directions avoids the agent learning to move backwards into itself.

## Reward Structure
| Event                          | Reward |
|--------------------------------|--------|
| Eating fruit                   | +2.0   |
| Moving closer to fruit         | +0.01  |
| Moving away from fruit         | −0.01  |
| Dying (wall / body / timeout)  | −1.0   |
| Filling the entire board       | +10.0  |

The small Manhattan distance rewards (+0.01 / −0.01) were added because pure sparse rewards (only on eating or dying) made early training extremely slow — the agent had no gradient signal to learn directional movement.

## Training Setup
| Parameter                | Value         |
|--------------------------|---------------|
| Optimizer                | Adam          |
| Learning Rate            | 0.001         |
| Loss Function            | MSE           |
| Discount Factor (γ)      | 0.99          |
| Epsilon (start → min)    | 1.0 → 0.0001 |
| Epsilon Decay            | 0.9995        |
| Replay Memory            | 10,000        |
| Batch Size               | 256           |
| Target Network Update    | Every 512 games |
| Training Frequency (CNN) | Every 4 steps |
| Timeout                  | W × H × 2 steps |

Key techniques:
* **Experience Replay** — sampling random mini-batches from a replay buffer to break temporal correlations.
* **Target Network** — a separate, periodically-synced copy of the Q-network for stable target values.
* **ε-greedy Exploration** — starts fully random and decays toward greedy action selection.

## Observations
* **Death ratio as a learning signal**: Early in training, the agent almost exclusively dies by hitting walls. As it learns, the death ratio shifts toward body collisions — it's surviving long enough to grow and run into itself. This is visible in the dashboard's stacked bar chart.
* **Training frequency matters**: Running `replay()` every 4 steps instead of every step made CNN training significantly faster without hurting performance.
* **MLP vs CNN on small boards**: On a 5×5 board the difference is modest (~9.5 vs ~10 average score), but the CNN should generalize better to larger boards.

## Project Structure
```
Snake_DQN_Agent/
├── environment.py         # Snake game environment
├── agent.py               # AgentMLP and AgentCNN classes
├── q_network_model.py     # QNetworkMLP and QNetworkCNN (PyTorch)
├── train.ipynb            # Training notebook with logging and plots
├── models/                # Saved model weights (.pth)
└── assets/                # Training graphs
```

## Tech Stack
* Python
* PyTorch
* DQN (Deep Q-Network)
* Jupyter Notebook

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install torch matplotlib jupyter`
3. Open `train.ipynb` to train an agent or load a pre-trained model.

## Roadmap
- [ ] Gradient clipping for training stability
- [ ] Larger board sizes (10×10, 15×15)
- [ ] Dueling DQN architecture
- [ ] Prioritized experience replay
- [ ] Hyperparameter sweep / tuning