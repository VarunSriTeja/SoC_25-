# AI/ML Learning Journey: Conceptual Reflection, Progress & Assignments

This repository documents my learning journey across Python, Neural Networks, Convolutional Neural Networks (CNNs), PyTorch, and Reinforcement Learning (RL). It includes a reflection on what I already knew, how much I improved, and the new concepts I learned each week — both theoretically and practically.

---

##  Weekly Overview

###  Week 1: Python & Snake Game with Pygame

####  What I Already Knew
- Basic Python syntax, control flow (if-else, loops)
- Functions and some exposure to object-oriented programming

#### What I Improved
- Deeper understanding of OOP concepts (classes, constructors, encapsulation)
- Python modules like `random`, `time`
- Structuring code using game loops and event handling

#### New Concepts Learned
- Game development using **Pygame**
- Handling game states and frame refresh logic
- Using `pygame.Rect`, collision detection, and key event listeners

####  Assignment
- Built a fully functional **Snake Game** using Pygame
- Saved under: `Week1/snake_game.py`

#### Resources Used
- W3Schools Python
- GeeksforGeeks Pygame
- Snake Game Tutorial by Tech with Tim

---

### Week 2: Neural Networks & Convolutional Neural Networks (CNNs)

#### What I Already Knew
- Supervised Learning, basic understanding of loss functions
- High-level idea of what neural networks are

#### What I Improved
- Clear understanding of **forward and backward propagation**
- Activation functions: ReLU, Sigmoid, Tanh
- Practical implementation using **PyTorch**

#### New Concepts Learned
- Neural Network architecture (input, hidden, output layers)
- **Training process**: loss calculation, gradients, weight update
- CNN Concepts:
  - Convolutions, Filters, Stride, Padding
  - Pooling layers (Max/Avg Pooling)
  - Feature extraction in images

#### Assignments
- Built and trained:
  - A basic feedforward **Neural Network**
  - A **CNN model** using PyTorch for image classification
- Files in: `Week2/NN.ipynb/`

#### Resources Used
- 3Blue1Brown NN Series
- Andrew Ng's Deep Learning playlist
- TensorFlow Playground
- PyTorch CIFAR-10 Classifier Tutorial

---

### Week 3: Reinforcement Learning (Theory)

#### 🔍 What I Already Knew
- Very basic intuition that RL is about agents taking actions in environments

#### What I Improved
- Strong theoretical grounding in **Markov Decision Processes (MDPs)**
- Concepts of **reward, policy, value function, Q-function**

#### New Concepts Learned
- Exploration vs Exploitation
- Bellman Equation
- Value Iteration & Policy Iteration
- Monte Carlo and Temporal Difference Learning
- TD(0), SARSA, Q-Learning (intuitive only, no code here)

#### 📁 Resources Used
- David Silver's RL Lectures
- Sutton & Barto RL Book (select chapters)
- SpinningUp RL Introduction

---

###  Week 4: Deep Q-Learning & DQNs

####  What I Already Knew
- Basic idea that DQN is used for decision-making in complex environments like games

#### What I Improved
- Understanding of how **Q-learning is extended using Deep Learning**
- Use of **Replay Buffers**, **Target Networks**

#### New Concepts Learned
- Deep Q Networks (DQNs): using neural nets to approximate Q-values
- Challenges like instability, and how they're solved (fixed Q-targets, experience replay)
- Hyperparameters: epsilon, gamma, learning rate
- Differences between:
  - **Q-learning** (tabular)
  - **Deep Q-learning** (theory)
  - **Deep Q-Network (DQN)** (architecture)

#### Resources Used
- Sentdex DQN Playlist
- Baeldung & Medium articles on Q-learning vs DQN
- PyTorch DQN implementation (video walkthroughs)
