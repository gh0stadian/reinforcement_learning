train_config = {
    'batch_size': 64,
    'lr': 1e-4,
    'gamma': 0.99,
    'sync_rate': 10,
    'replay_size': 10000,
    'eps_decay': 200,
    'eps_start': 0.9,
    'eps_end': 0.05,
    'warm_start_steps': 8000,
    'reward_decreasing_limit': 60,
    'record_video_episode': 20,
    'save_model_episode': 60,
}

model_config = {
    'conv_layers': [32, 64, 128, 256],
    'lin_layers': [64],
}

env_config = {
    'env_name': "SpaceInvaders-v4",
}

wrappers_config = {
    'fire_reset': True,
    'max_n_skip': False,
    'clip_reward': True
}

load_model = {
    "path": "checkpoints/05_07_2022_T_19_02_02",
}
