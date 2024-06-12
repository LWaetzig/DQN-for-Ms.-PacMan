import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def preprocess_state(
    state: np.array,
    stack_states: bool = True,
    stack_size: int = 4,
    create_tensor: bool = True,
) -> torch.tensor:
    """preprocess the state

    Args:
        state (np.array): state of the environment

    Returns:
        np.array: preprocessed state
    """
    # convert to grayscale
    state = cv.cvtColor(state, cv.COLOR_RGB2GRAY)
    # resize the image
    state = cv.resize(state, (84, 84))
    # normalize the image
    state = state / 255.0

    if stack_states:
        state = np.stack([state] * stack_size)
    if create_tensor:
        state = torch.FloatTensor(state).unsqueeze(0)

    return state


def create_plots(
    episode_rewards: list,
    episode_lengths: list,
    episode_losses: list,
    episode_epsilons: list,
    save_fig: bool = True,
    save_path: str = "results.png",
) -> None:
    """create plots during training

    Args:
        episode_rewards (list): list of rewards per episode
        episode_lengths (list): list of lengths per episode
        episode_losses (list): list of losses per episode
        episode_epsilons (list): list of epsilons per episode
        save_fig (bool, optional): decide whether to save plot. Defaults to True.
        save_path (str, optional): save path for plot. Defaults to "results.png".
    """

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    axes = axes.flatten()

    # create plot for episode rewards
    axes[0].plot(
        np.arange(len(episode_rewards)), episode_rewards, label="Episode Rewards"
    )
    mvg_avg_reward = pd.Series(episode_rewards).rolling(10).mean().dropna()
    axes[0].plot(
        np.arange(len(mvg_avg_reward)),
        mvg_avg_reward,
        label="Moving Average",
    )
    axes[0].set_ylabel("Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_title("Episode Rewards")
    axes[0].grid(True)
    axes[0].legend()

    # create plot for episode length
    axes[1].plot(
        np.arange(len(episode_lengths)), episode_lengths, label="Episode Length"
    )
    mvg_avg_length = pd.Series(episode_lengths).rolling(10).mean().dropna()
    axes[1].plot(
        np.arange(len(mvg_avg_length)),
        mvg_avg_length,
        label="Moving Average",
    )
    axes[1].set_ylabel("Episode Length")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Episode")
    axes[1].set_title("Episode Length\n (in seconds)")
    axes[1].grid(True)
    axes[1].legend()

    # create plot for loss
    axes[2].plot(np.arange(len(episode_losses)), episode_losses, label="Loss")
    mvg_avg_loss = pd.Series(episode_losses).rolling(10).mean().dropna()
    axes[2].plot(
        np.arange(len(mvg_avg_loss)),
        mvg_avg_loss,
        label="Moving Average",
    )
    axes[2].set_ylabel("Loss")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Episode")
    axes[2].set_title("Loss per Episode")
    axes[2].grid()
    axes[2].legend()

    # create plot for epsilon
    axes[3].plot(np.arange(len(episode_epsilons)), episode_epsilons, label="Epsilon")
    axes[3].set_ylabel("Epsilon")
    axes[3].set_xlabel("Episode")

    axes[3].set_title("Epsilon Decay")
    axes[3].grid(True)
    axes[3].legend()

    fig.tight_layout()
    # save figure
    if save_fig:
        fig.savefig(f"{save_path}.png")
