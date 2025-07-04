# Bayesian Reinforcement Learning

This project is an exploration into the intersection of Bayesian inference and reinforcement learning. It features a smart agent that learns to navigate a 2D grid world, but the real contribution of the repository is the "Observer" that can infer what the agent is "thinking" just by watching its actions.

At its core, this repository demonstrates how an outside observer can build a mental model of an agent's beliefs about the world. The agent itself has a private, internal map of its surroundings, including the location of walls it has discovered. It uses a pre-trained policy, developed with reinforcement learning, to decide the best path to its target. The observer, however, doesn't get to see this private map. It only sees where the agent moves.

This is where Rational Speech Act (RSA) reasoning comes into play. The observer starts with a general, uncertain belief about the world's layout. When the agent takes an action, the observer reasons backward. It considers thousands of possible environments the agent could be in and simulates the most "rational" action for each one using a model of the agent's policy. If the agent moves right, for example, the observer concludes that the true environment is more likely one in which moving right was the optimal choice. By repeatedly updating its beliefs this way, the observer can create a surprisingly accurate picture of the agent's hidden knowledge, turning simple actions into a rich source of information.

## Getting Started

To get this running on your own machine, you'll first need to clone the repository and set up the necessary Python libraries.


    git clone https://github.com/henrild/bayesianrl.git
    cd bayesianrl
    pip install -r requirements.txt

## How to Run

The main simulation can be kicked off by running the rsa_agent.py script. This will launch a visual simulation where you can see the agent navigating the grid, and side-by-side, you'll see the observer's evolving belief map, showing you what it infers about the hidden walls.


    python rsa_agent.py


If you'd like to train the navigation model from scratch, you can use the train_heuristic.py script. This will train a new agent using several predefined map layouts and save the resulting model, which the RSA agent then uses as its foundation for decision-making.


    python train_heuristic.py


## Core Components

  - `env.py`: This file builds the 2D grid world where the agent lives, handling all the rules of movement and interaction.

  - `train_heuristic.py`: Here you'll find the logic for training the base navigation agent using reinforcement learning (specifically PPO from Stable-Baselines3).

  - `rsa_agent.py`: This is the heart of the project, containing the RSAAgent that acts in the world and the Observer that uses RSA-style reasoning to infer the agent's hidden belief state.

  - `base_agent.py`: This file defines a simpler agent that acts directly on the pre-trained policy, without the complex RSA reasoning layer. It serves as a baseline for comparison.
