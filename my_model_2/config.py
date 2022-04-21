train_config = {
    'batch_size': 128,
    'lr': 1e-2,
    'gamma': 0.99,
    'sync_rate': 10,
    'replay_size': 1000,
    'warm_start_size': 1000,
    'eps_last_frame': 1000,
    'eps_start': 1.0,
    'eps_end': 0.01,
    'episode_length': 200,
    'warm_start_steps': 1000,
    'log_video_epoch': 200,
    'reward_decreasing_limit': 20,
}
model_config = {
    'conv_layers': [32, 64, 128],
    'lin_layers': [32],
}
env_config = {
    'env_name': "CarRacing-v1",
}

action_space = [[1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 1]]
