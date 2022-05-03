train_config = {
    'batch_size': 512,
    'lr': 1e-2,
    'gamma': 0.99,
    'sync_rate': 10,
    'replay_size': 10000,
    'warm_start_size': 2000,
    'eps_last_frame': 1000,
    'eps_start': 1.0,
    'eps_end': 0.01,
    'episode_length': 250,
    'warm_start_steps': 2000,
    'log_video_epoch': 1000,
    'reward_decreasing_limit': 60,
}
model_config = {
    'conv_layers': [32, 64, 128, 256],
    'lin_layers': [64, 32],
}
env_config = {
    'env_name': "CarRacing-v1",
}

action_space = [[1, 0, 0],
                [-1, 0, 0],
                [0, .7, 0],
                [0, 0, 0],
                [0, 0, 1]]
