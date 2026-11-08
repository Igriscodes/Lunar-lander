# Lunar lander

 A Jupyter notebook implementing a Deep Q-Network (DQN) for reinforcement learning:

* **Setup**: Initializes PyTorch, and loads the Gymnasium `LunarLander-v3` environment.


* **Neural Network (`QNetwork`)**: Implements a feedforward neural network with hidden layers and ReLU activations to map environment states to action values.


* **Experience Replay (`ReplayBuffer`)**: Utilizes a deque-based memory buffer to store and randomly sample transitions, stabilizing training by decorrelating consecutive experiences.


* **DQN Agent (`DQNAgent`)**:

* Manages separate policy and target networks.


* Implements an epsilon-greedy action selection strategy with a decay schedule.


* Optimizes the network using Smooth L1 Loss (Huber loss) and gradient clipping.


* Performs soft updates (`tau`) to gradually update the target network.

* **Training & Evaluation**: Trains the agent across episodes until it hits the solve threshold (a 100-episode average reward of 200 or higher), saves the trained model weights to disk (`lunarlander_dqn.pth`), plots the reward history, and runs an evaluation loop to benchmark performance.

<img width="600" height="400" alt="demo" src="https://github.com/user-attachments/assets/3d79766a-3ec0-4f4b-a8e6-065df5066bf2" />
