# Deep Q-Network for Ms. Pac-Man

## What's in it

> **TLDR:** This repository implements a DQN (Deep Q-Network) for the retro game Pacman

### Deep Q-Networks

Deep Q-Networks are deep neural networks that learn an optimal action strategy based on the Q-values of different states.
The DQN is used to approximate a Q function that estimates the expected future reward for each action in a given state. Training is done by learning from experience, where the agent collects information and stores it in a replay buffer. Therefore, mini-batches are selected to learn from. By minimizing a loss function, which represents the difference between the estimated Q-value and the actual reward of the actions in the experiences, the network adjusts its weights to achieve a more accurate estimation of the Q-function.
For this specific use case, the DQN consists of additional convolutional layers to extract information from the states. Therefore, a "snapshot" is fed into the network.

### Results

- more detailed evaluation is provided in the [notebook](./pacman.ipynb)
- best episode


https://github.com/LWaetzig/DQN-for-Ms.-PacMan/assets/92372282/c713f2e9-8449-4ea8-bffd-0e5bfb2a71b9



> **note:** Normally the video should be displayed here. If not, you can find the video in the [videos](./videos) directory.




## Requirements

- python environment >= 3.9.10
- install packages listed in [requirements.txt](./requirements.txt)
  > **note:** sometimes the installation of atari and atari-rom-license failes "no match found" or the packages just will not be installed. Run the following commands again.

```bash
pip install "gymnasium[atari]"
pip install "gymnasium[accept-rom-license]"
```
