import matplotlib.pyplot as plt
import numpy as np
from config import action_space


def rgb2grey(img):
    return np.dot(img[..., :], [0.299, 0.587, 0.114])


def normalize(img):
    return img / 255.


def state_transform(state):
    state = rgb2grey(state)

    # Crop img:
    state = state[16:-16, 16:-16]

    state = normalize(state)
    # state = state * 2 - 1

    # state = np.expand_dims(state, axis=0)
    return state


def action_transform(action):
    action_index = action.argmax()
    actions = [[-action[action_index], 0, 0],   # LEFT STEER
               [action[action_index], 0, 0],    # RIGHT STEER
               [0, action[action_index], 0],    # GAS
               [0, 0, action[action_index]],    # BRAKE
               [0, 0, 0]]                       # DO NOTHING
    return actions[action_index]
