"""
This code is the modified code from https://github.com/hunkim/ReinforcementZeroToAll/
Double DQN (Nature 2015)
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
Notes:
    The difference is that now there are two DQNs (DQN & Target DQN)
    y_i = r_i + 𝛾 * max(Q(next_state, action; 𝜃_target))
    Loss: (y_i - Q(state, action; 𝜃))^2
    Every C step, 𝜃_target <- 𝜃
"""
import os
import numpy as np
import tensorflow as tf
import random
from collections import deque
import deep_q_network as dqn
from point_and_click_env import Env
from score_logger import ScoreLogger
from typing import List

env_1 = Env()
env_2 = Env()
score_logger_1 = ScoreLogger('mouse model_1', 1000, 100000)
score_logger_2 = ScoreLogger('mouse model_2', 1000, 100000)

# Constants defining our neural network
INPUT_SIZE = env_1.observation_space.shape[0]
OUTPUT_SIZE = env_1.action_space.n

DISCOUNT_RATE = 0.95
REPLAY_MEMORY = 100000
BATCH_SIZE = 32
TARGET_UPDATE_FREQUENCY = 1000
MAX_EPISODES = 5000000
SAVE_PERIOD = 10
E_DECAY = 0.9998
E_MIN = 0.05

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
Trains `mainDQN` with target Q values given by `targetDQN`
Args:
    mainDQN (dqn.DQN): Main DQN that will be trained
    targetDQN (dqn.DQN): Target DQN that will predict Q_target
    train_batch (list): Minibatch of replay memory
        Each element is (s, a, r, s', done)
        [(state, action, reward, next_state, done), ...]
Returns:
    float: After updating `mainDQN`, it returns a `loss`
"""



def replay_train(mainDQN: dqn.DQN, targetDQN: dqn.DQN, train_batch: list) -> float:
    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch])
    rewards = np.array([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch])

    X = states
    Q_target = rewards + DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1) * ~done

    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(X, y)

"""
Creates TF operations that copy weights from `src_scope` to `dest_scope`
Args:
    dest_scope_name (str): Destination weights (copy to)
    src_scope_name (str): Source weight (copy from)
Returns:
    List[tf.Operation]: Update operations are created and returned
"""

def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def main():
    # store the previous observations in replay memory
    replay_buffer_1 = deque(maxlen=REPLAY_MEMORY)
    replay_buffer_2 = deque(maxlen=REPLAY_MEMORY)

    with tf.Session() as sess:
        mainDQN_1 = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main1")
        mainDQN_2 = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main2")
        targetDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        sess.run(tf.global_variables_initializer())

        # initial copy q_net -> target_net
        copy_ops1 = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        copy_ops2 = get_copy_var_ops(dest_scope_name="target",
                                     src_scope_name="main2")
        #
        sess.run(copy_ops1)
        sess.run(copy_ops2)

        step_count_1 = 0
        step_count_2 = 0

        e = 1

        for episode in range(MAX_EPISODES + 1):
            if e > E_MIN: e *= E_DECAY
            done_1 = False
            score_1 = 0
            count_1 = 0
            loss_1 = 0
            q_value_1 = 0

            done_2 = False
            score_2 = 0
            count_2 = 0
            loss_2 = 0
            q_value_2 = 0

            state_1 = env_1.reset()
            state_2 = env_2.reset()

            while not (done_1 and done_2):
                # Get the q table
                q_values_1 = mainDQN_1.predict(state_1)
                q_value_1 += np.mean(q_values_1)

                q_values_2 = mainDQN_1.predict(state_2)
                q_value_2 += np.mean(q_values_2)

                # Get the action
                action_1 = np.argmax(q_values_1)
                action_2 = np.argmax(q_values_2)

                if np.random.rand() < e:
                    action_1 = env_1.action_space.sample()
                    action_2 = env_2.action_space.sample()

                # Get new state and reward from environment
                next_state_1, effort_reward_1, click_reward_1, done_1, _ = env_1.step(action_1)
                next_state_2, effort_reward_2, click_reward_2, done_2, _ = env_2.step(action_2)

                # TODO : Determine who won
                if done_1:
                    if done_2:
                        if click_reward_1 == 14 and click_reward_2 == 14:

                            click_reward_1 = 7
                            click_reward_2 = 7
                            reward_1 = click_reward_1 + effort_reward_1
                            reward_2 = click_reward_2 + effort_reward_2

                        elif click_reward_1 == 14 and click_reward_2 == -1:
                            # 1 won 2 fail
                            reward_1 = click_reward_1 + effort_reward_1
                            reward_2 = click_reward_2 + effort_reward_2

                        elif click_reward_1 == -1 and click_reward_2 == 14:
                            # 1 fail 2 won
                            reward_1 = click_reward_1 + effort_reward_1
                            reward_2 = click_reward_2 + effort_reward_2

                        else:
                            # 1 fail 2 fail
                            reward_1 = click_reward_1 + effort_reward_1
                            reward_2 = click_reward_2 + effort_reward_2

                    else:
                        if click_reward_1 == 14:
                            # 1 won 2 very bad & 2 stop
                            reward_1 = click_reward_1 + effort_reward_1
                            reward_2 = click_reward_2 + effort_reward_2

                        elif click_reward_1 == -1:
                            # 1 fail 2 continue

                            extra_effort = 0
                            # nested while loop

                            while not done_2:
                                # Get the q table
                                q_values_2 = mainDQN_2.predict(state_2)
                                q_value_2 += np.mean(q_values_2)

                                # Get the action
                                action_2 = np.argmax(q_values_2)

                                if np.random.rand() < e:
                                    action_2 = env_2.action_space.sample()

                                # Get new state and reward from environment
                                next_state, local_extra_effort, click_reward, done_2, _ = env_2.step(action_2)
                                extra_effort += local_extra_effort

                                # Save the experience to our buffer
                                replay_buffer_2.append((state_2, action_2, extra_effort + click_reward, next_state_2, done_2))

                                if len(replay_buffer_2) > BATCH_SIZE:
                                    minibatch = random.sample(replay_buffer_2, BATCH_SIZE)
                                    loss_temp, _ = replay_train(mainDQN_2, targetDQN, minibatch)
                                    loss_2 += loss_temp
                                    count_2 += 1

                                if step_count_2 % TARGET_UPDATE_FREQUENCY == 0:
                                    sess.run(copy_ops2)

                                score_2 += local_extra_effort + click_reward
                                state_2 = next_state
                                step_count_2 += 1

                            reward_1 = effort_reward_1 + click_reward_1
                            reward_2 = effort_reward_2 + click_reward_2 + extra_effort + click_reward

                else:
                    if done_2:
                        if click_reward_2 == 14:
                            # 2 won 1 very bad & 1 stop
                            reward_1 = click_reward_1 + effort_reward_1
                            reward_2 = click_reward_2 + effort_reward_2
                        elif click_reward_2 == -1:

                            # 2 fail 1 continue

                            extra_effort = 0
                            # nested while loop

                            while not done_1:
                                # Get the q table
                                q_values_1 = mainDQN_1.predict(state_1)
                                q_value_1 += np.mean(q_values_1)

                                # Get the action
                                action_1 = np.argmax(q_values_1)

                                if np.random.rand() < e:
                                    action_1 = env_1.action_space.sample()

                                # Get new state and reward from environment
                                next_state, local_extra_effort, click_reward, done_1, _ = env_1.step(action_1)
                                extra_effort += local_extra_effort

                                # Save the experience to our buffer
                                replay_buffer_1.append(
                                    (state_1, action_1, extra_effort + click_reward, next_state_1, done_1))

                                if len(replay_buffer_1) > BATCH_SIZE:
                                    minibatch = random.sample(replay_buffer_1, BATCH_SIZE)
                                    loss_temp, _ = replay_train(mainDQN_1, targetDQN, minibatch)
                                    loss_1 += loss_temp
                                    count_1 += 1

                                if step_count_1 % TARGET_UPDATE_FREQUENCY == 0:
                                    sess.run(copy_ops1)

                                score_1 += local_extra_effort + click_reward
                                state_1 = next_state
                                step_count_1 += 1

                            reward_1 = effort_reward_1 + click_reward_1 + extra_effort + click_reward
                            reward_2 = effort_reward_2 + click_reward_2

                    else:
                        if click_reward_1 == 14 and click_reward_2 == 14:
                            # tie
                            click_reward_1 = 7
                            click_reward_2 = 7
                            reward_1 = click_reward_1 + effort_reward_1
                            reward_2 = click_reward_2 + effort_reward_2
                        elif click_reward_1 == -1 and click_reward_2 == 14:
                            # 2 won 1 fail
                            reward_1 = click_reward_1 + effort_reward_1
                            reward_2 = click_reward_2 + effort_reward_2
                        elif click_reward_1 == 14 and click_reward_2 == -1:
                            # 2 fail 1 won
                            reward_1 = click_reward_1 + effort_reward_1
                            reward_2 = click_reward_2 + effort_reward_2
                        else:
                            # 1 fail 2 fail
                            reward_1 = click_reward_1 + effort_reward_1
                            reward_2 = click_reward_2 + effort_reward_2


                # Save the experience to our buffer
                replay_buffer_1.append((state_1, action_1, reward_1, next_state_1, done_1))
                replay_buffer_2.append((state_2, action_2, reward_2, next_state_2, done_2))

                if len(replay_buffer_1) > BATCH_SIZE:
                    minibatch_1 = random.sample(replay_buffer_1, BATCH_SIZE)
                    loss_temp_1, _ = replay_train(mainDQN_1, targetDQN, minibatch_1)
                    loss_1 += loss_temp_1
                    count_1 += 1

                if len(replay_buffer_2) > BATCH_SIZE:
                    minibatch_2 = random.sample(replay_buffer_2, BATCH_SIZE)
                    loss_temp_2, _ = replay_train(mainDQN_2, targetDQN, minibatch_2)
                    loss_2 += loss_temp_2
                    count_2 += 1

                if step_count_1 % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(copy_ops1)

                if step_count_2 % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(copy_ops2)

                score_1 += reward_1
                state_1 = next_state_1

                score_2 += reward_2
                state_2 = next_state_2

                step_count_1 += 1
                step_count_2 += 1

            # Log the data
            if count_1 == 0:
                score_logger_1.add_csv(loss_1, q_value_1, score_1, env_1.time, env_1.effort, env_1.click, episode, agent_number=1)
                score_logger_2.add_csv(loss_2, q_value_2, score_2, env_2.time, env_2.effort, env_2.click, episode, agent_number=2)
            else:
                score_logger_1.add_csv(loss_1 / count_1, q_value_1 / count_1, score_1, env_1.time, env_1.effort, env_1.click, episode, agent_number=1)
                score_logger_2.add_csv(loss_2 / count_2, q_value_2 / count_2, score_2, env_1.time, env_2.effort, env_2.click, episode, agent_number=2)

            if episode % 10 == 0 or os.path.exists('./check'):
                _, _, _, ave = score_logger_1.score_show()
                _, _, _, ave_loss = score_logger_1.loss_show()
                _, _, _, ave_q = score_logger_1.q_value_show()
                time_mean = sum(env_1.time_mean) / len(env_1.time_mean)
                time_std = (sum([((x - time_mean) ** 2) for x in env_1.time_mean]) / len(env_1.time_mean)) ** 0.5
                error_rate = 1 - (sum(env_1.error_rate) / len(env_1.error_rate))
                print("Agent 1 Episode: {:}, Reward: {:.4}, Loss: {:.4}, Q Value: {:.4}, Time: {:.4} (SD: {:.4}), ER: {:.4}".format(
                    episode, float(ave), float(ave_loss), float(ave_q), float(time_mean), float(time_std), float(error_rate)))

                _, _, _, ave = score_logger_2.score_show()
                _, _, _, ave_loss = score_logger_2.loss_show()
                _, _, _, ave_q = score_logger_2.q_value_show()
                time_mean = sum(env_2.time_mean) / len(env_2.time_mean)
                time_std = (sum([((x - time_mean) ** 2) for x in env_2.time_mean]) / len(env_2.time_mean)) ** 0.5
                error_rate = 1 - (sum(env_2.error_rate) / len(env_2.error_rate))
                print("Agent 2 Episode: {:}, Reward: {:.4}, Loss: {:.4}, Q Value: {:.4}, Time: {:.4} (SD: {:.4}), ER: {:.4}".format(
                    episode, float(ave), float(ave_loss), float(ave_q), float(time_mean), float(time_std), float(error_rate)))

            # Save the model
            if episode % SAVE_PERIOD == 0 and episode >= SAVE_PERIOD:
                _, score_ave, _, _ = score_logger_1.score_show()
                _, loss_ave, _, _ = score_logger_1.loss_show()
                mainDQN_1.save(episode, score_ave, loss_ave)
                print("Saved the model for Agent 1", episode, score_ave, loss_ave)

                _, score_ave, _, _ = score_logger_2.score_show()
                _, loss_ave, _, _ = score_logger_2.loss_show()
                mainDQN_2.save(episode, score_ave, loss_ave)
                print("Saved the model for Agent 2", episode, score_ave, loss_ave)


if __name__ == "__main__":
    main()
