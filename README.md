# Deep Q-Network for MS Pacman

## What's in it
> **TLDR:** This repository implements a DQN (Deep Q-Network) for the retro game Pacman

### Deep Q-Networks
Deep Q-Networks are deep neural networks that learn an optimal action strategy based on the Q-values of different states.
The DQN is used to approximate a Q function that estimates the expected future reward for each action in a given state. Training is done by learning from experience, where the agent collects information and stores it in a replay buffer. Therefore, mini-batches are selected to learn from. By minimizing a loss function, which represents the difference between the estimated Q-value and the actual reward of the actions in the experiences, the network adjusts its weights to achieve a more accurate estimation of the Q-function.
For this specific use case, the DQN consists of additional convolutional layers to extract information from the states. Therefore, a "snapshot" is fed into the network.


### Results
- more detailed evaluation is provided in the [notebook](./pacman.ipynb) and in [Project Documentation]()
- best episode
<br>
<video controls autoplay width="200px" height="auto">
    <source src="videos/rl-video-episode-6.mp4" type="video/mp4" />
</video>

## Requirements
- python environment >3.9.10
- install packages listed in [requirements.txt](./requirements.txt)
> **note:** sometimes the installation of atari and atari-rom-license failes "no match found" or the packages just will not be installed. Run the following commands again.
```bash
pip install "gym[atari]"
pip install "gym[accept-rom-license]"
```