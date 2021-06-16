"""
Testing Module (CSV extractor)

1) Tests a trained agent and records trajectory_data data, action data into a csv file
2) Trained model (agent) should be located in ./
3) Original deep_q_network.py, point_and_click_model.p files should be located in ./
4) Original modules/*.py files should exist.

This code was written by Hyunwoo Lee.
"""

import tensorflow as tf
import deep_q_network as dqn
import pandas as pd
import numpy as np
from point_and_click_test_env import Env

# Creates a testing environment instance
env = Env()

# Constants defining our neural network
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n
MAX_TEST_EPISODES = 10000

sess = tf.Session()
mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")

new_saver = tf.train.Saver()
new_saver.restore(sess, tf.train.latest_checkpoint('./'))

for episode in range(MAX_TEST_EPISODES):
    done = False
    state = env.reset()

    while not done:
        # Get the q table
        q_values = mainDQN.predict(state)

        # Get the action
        action = np.argmax(q_values)

        # Get new state and reward from environment
        next_state, reward, done, _ = env.step(action)
        state = next_state


replay_buffer = pd.DataFrame(env.step_buffer, columns=['time', 'user', 'task', 'trial', 'cursor_x', 'cursor_y', 'target_x', 'target_y', 'target_radius', 'target_speed', 'click_action', 'click_success'])
action_buffer = pd.DataFrame(env.action_buffer, columns=['time', 'user', 'task', 'trial', 'th', 'threshold_id'])

replay_buffer.to_csv('trajectory_data.csv', index=False)
action_buffer.to_csv('action.csv', index=False)


