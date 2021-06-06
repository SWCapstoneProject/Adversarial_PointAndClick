"""
Training History Plotting Module

1) Plots the graph of loss, reward, q-value

This code was written by Hyunwoo Lee.
"""

import pandas as pd
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PATH = './training_csv_outputs_directory_goes_here'

def moving_avg(x, N=30):
    return np.convolve(x, np.ones((N,))/N, mode='valid')

base_dir = Path(PATH)
agent_1_csv_files = glob.glob(str(base_dir / '*agent_1.csv'))
agent_2_csv_files = glob.glob(str(base_dir / '*agent_2.csv'))

agent1_df = pd.concat((pd.read_csv(f, names=['loss', 'q_value', 'score', 'time', 'effort', 'click']) for f in agent_1_csv_files))
agent2_df = pd.concat((pd.read_csv(f, names=['loss', 'q_value', 'score', 'time', 'effort', 'click']) for f in agent_2_csv_files))

# 1. Q - Value
agent1_np_qvalues = agent1_df['q_value'].to_numpy()
agent2_np_qvalues = agent2_df['q_value'].to_numpy()

plt.figure(figsize=(16, 4))
plt.plot(moving_avg(agent1_np_qvalues, 1), label='agent1_qvalue')
plt.plot(moving_avg(agent2_np_qvalues, 1), label='agent2_qvalue')
plt.legend(loc=2)
plt.savefig('qvalue.png')

# 2. Loss
agent1_np_loss = agent1_df['loss'].to_numpy()
agent2_np_loss = agent2_df['loss'].to_numpy()

plt.figure(figsize=(16, 4))
plt.plot(moving_avg(agent1_np_loss, 1), label='agent1_loss')
plt.plot(moving_avg(agent2_np_loss, 1), label='agent2_loss')
plt.legend(loc=2)
plt.savefig('loss.png')

# 3. Reward
agent1_np_reward = agent1_df['score'].to_numpy()
agent2_np_reward = agent2_df['score'].to_numpy()

plt.figure(figsize=(16, 4))
plt.plot(moving_avg(agent1_np_reward, 1), label='agent1_reward')
plt.plot(moving_avg(agent2_np_reward, 1), label='agent2_reward')
plt.legend(loc=2)
plt.savefig('reward.png')
