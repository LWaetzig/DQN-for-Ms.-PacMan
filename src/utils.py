import cv2 as cv
import numpy as np
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
