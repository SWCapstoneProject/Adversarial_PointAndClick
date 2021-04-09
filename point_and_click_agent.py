from Constants import BATCH_SIZE, DISCOUNT_RATE
from point_and_click_env import Env
import random
import numpy as np


def replay_train(agent_dqn, target_dqn, train_batch: list) -> float:
    """
        Trains `agent_dqn` with target Q values given by `target_dqn`
        Args:
            agent_dqn : agent DQN that will be trained
            target_dqn : Target DQN that will predict Q_target
            train_batch (list): Minibatch of replay memory
                Each element is (s, a, r, s', done)
                [(state, action, reward, next_state, done), ...]
        Returns:
            float: After updating `agent_dqn`, it returns a `loss`
    """

    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch])
    rewards = np.array([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch])

    X = states
    Q_target = rewards + DISCOUNT_RATE * np.max(target_dqn.predict(next_states), axis=1) * ~done

    y = agent_dqn.predict(states)
    y[np.arange(len(X)), actions] = Q_target

    # Train our network using target and predicted Q values on each episode
    return agent_dqn.update(X, y)


class Agent:

    def __init__(self, name):
        self.name = name
        self.done = False
        self.score = 0
        self.count = 0
        self.step_count = 0
        self.loss = 0
        self.q_value = 0
        self.state = None

    def update_replay_buffer(self, replay_buffer, action, reward, next_state, agent_dqn, target_dqn):
        replay_buffer.append((self.state, action, reward, next_state, self.done))

        if len(replay_buffer) > BATCH_SIZE:
            mini_batch = random.sample(replay_buffer, BATCH_SIZE)
            loss_temp, _ = replay_train(agent_dqn, target_dqn, mini_batch)
            self.loss += loss_temp
            self.count += 1

    def update_step_result(self, reward, next_state):
        self.score += reward
        self.state = next_state
        self.step_count += 1
