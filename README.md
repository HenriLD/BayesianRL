# Pragmatic Explainability for Entropic Reinforcement Learning (PEER)
This project introduces PEER (Pragmatic Explainability for Entropic Reinforcement Learning), a method for inferring an agent's hidden beliefs by observing its actions. The primary contribution is a formal, model-based approach to explainability in the context of high-entropy, stochastic policies.

The core challenge with modern reinforcement learning algorithms that maximize an entropy objective is that their resulting policies are inherently stochastic. This makes it difficult to determine the precise, underlying reason for a given action. PEER addresses this by applying principles from language pragmatics, treating an agent's physical actions as "utterances" that can reveal its private knowledge about the world. It provides a computational framework to interpret these actions, not as random samples, but as informative signals.

The method is operationalized through a recursive model of pragmatic reasoning. The agent uses an internal model of an observer to simulate how its actions will be interpreted, while the external observer uses a model of the agent's policy to infer the most likely world state given a particular action. Both operate on a shared assumption derived from pragmatic theory: that the agent, within the constraints of its primary goal, will act in a way that is maximally informative to a rational observer. This is achieved via an iterative reasoning process that amplifies small differences in expected utility, allowing the system to converge on an action that most unambiguously signals the agent's private knowledge.

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
