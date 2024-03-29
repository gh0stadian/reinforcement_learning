import matplotlib.pyplot as plt
import numpy as np


def rgb2grey(img):
    return np.dot(img[..., :], [0.299, 0.587, 0.114])


def normalize(img):
    return img / 255.


def state_transform(state):
    state = rgb2grey(state)
    # Crop img:
    state = state[18:, :]
    state[:2] = 0.0
    state[177:] = 0.0
    state = normalize(state)
    return state


def action_transform(action):
    return action
