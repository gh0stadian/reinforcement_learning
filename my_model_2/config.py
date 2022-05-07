train_config = {
    'batch_size': 64,
    'lr': 1e-3,
    'gamma': 0.99,
    'sync_rate': 10,
    'replay_size': 10000,
    'warm_start_size': 7000,
    'eps_last_frame': 1000,
    'eps_start': 1.0,
    'eps_end': 0.01,
    'eps_decay': 200,
    'episode_length': 64,
    'warm_start_steps': 2000,
    'log_video_episode': 20,
}

model_config = {
    'conv_layers': [32, 64, 128, 256],
    'lin_layers': [64, 32]
}

env_config = {
    'env_name': 'SpaceInvaders-v4',
}

wrappers_config = {
    'fire_reset': True,
    'max_n_skip': False,
    'clip_reward': True
}