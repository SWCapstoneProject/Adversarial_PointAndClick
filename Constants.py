from point_and_click_env import Env

# 1. Neural Network Related
DISCOUNT_RATE = 0.95
REPLAY_MEMORY = 100000
BATCH_SIZE = 32
TARGET_UPDATE_FREQUENCY = 1000
MAX_EPISODES = 50000000
MODEL_SAVE_PERIOD = 10000
CSV_SAVE_PERIOD = 100000
E_DECAY = 0.9998
E_MIN = 0.05
PRINT_FREQUENCY = 10

env = Env()
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n
