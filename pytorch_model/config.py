train_config = {
    'batch_size': 256,
    'lr': 1e-2,
    'gamma': 0.99,
    'sync_rate': 10,
    'replay_size': 10000,
    'eps_decay': 200,
    'eps_start': 0.9,
    'eps_end': 0.05,
    'warm_start_steps': 8500,
    'reward_decreasing_limit': 60,
}

model_config = {
    'conv_layers': [32, 64, 128, 256],
    'lin_layers': [64, 32],
}

env_config = {
    'env_name': "CarRacing-v1",
}

action_space = [[-1, 0.8, 0.2], [0, 1, 0.2], [1, 0.8, 0.2],
                [-1, 0.8, 0], [0, 1, 0], [1, 0.8, 0],
                [-1, 0, 0.2], [0, 0, 0.2], [1, 0, 0.2],
                [-1, 0, 0], [0, 0, 0], [1, 0, 0]]
