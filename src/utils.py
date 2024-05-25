import cv2 as cv
import gymnasium as gym
import numpy as np


def preprocess_state(state: np.array) -> np.array:
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

    return state
