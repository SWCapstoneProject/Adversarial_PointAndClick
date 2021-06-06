from point_and_click_env import Env

# Reinforcement Learning related constants
DISCOUNT_RATE = 0.95
REPLAY_MEMORY = 100000
BATCH_SIZE = 32
TARGET_UPDATE_FREQUENCY = 1000
E_DECAY = 0.9998
E_MIN = 0.05

env = Env()
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n
