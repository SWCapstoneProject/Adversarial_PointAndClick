from typing import List
import numpy as np
from Constants import *
import tensorflow as tf
import os


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


def compute_extra_reward(agent, agent_env, replay_buffer, agent_dqn, target_dqn, sess, vars, e):

    total_extra_effort = 0

    while not agent.done:

        q_values = agent_dqn.predict(agent.state)
        agent.q_value += np.mean(q_values)

        # Get the action
        action = np.argmax(q_values)

        if np.random.rand() < e:
            action = agent_env.action_space.sample()

        # Get new state and reward from environment
        next_state, effort_reward, click_reward, agent.done, _ = agent_env.step(action)
        total_extra_effort += effort_reward

        # Save the experience to our buffer
        agent.update_replay_buffer(replay_buffer, action, effort_reward + click_reward, next_state, agent_dqn, target_dqn)

        if agent.step_count % TARGET_UPDATE_FREQUENCY == 0:
            sess.run(vars)

        agent.update_step_result(effort_reward + click_reward, next_state)

    return total_extra_effort, click_reward


def determine_reward(my_agent, opponent,
                     my_agent_env, opponent_env,
                     my_agent_replay_buffer, opponent_replay_buffer,
                     my_effort_reward, my_click_reward,
                     opponent_effort_reward, opponent_click_reward,
                     my_agent_dqn, opponent_dqn, my_target_dqn, op_target_dqn, sess,
                     my_agent_vars, opponent_vars, e_my_agent, e_opponent):

    if my_agent.done:
        if opponent.done:
            # When both agents clicked at the same time
            # No need to recompute reward
            pass

        else:
            # When My agent clicked first, opponent didn't click yet

            if my_click_reward == 14:
                # My agent succeeded first
                # If opponent succeeds afterwards, give click_reward = 9
                extra_effort_reward, extra_click_reward = compute_extra_reward(opponent, opponent_env,
                                                                               opponent_replay_buffer, opponent_dqn, op_target_dqn,
                                                                               sess, opponent_vars, e_opponent)
                opponent_effort_reward = opponent_effort_reward + extra_effort_reward
                if extra_click_reward == 14:
                    extra_click_reward = 9
                opponent_click_reward = extra_click_reward
                opponent_env.fail_rate.append(0)

            else:
                # My agent failed first
                # Wait until opponent finishes the episode

                extra_effort_reward, extra_click_reward = compute_extra_reward(opponent, opponent_env,
                                                                               opponent_replay_buffer, opponent_dqn, op_target_dqn,
                                                                               sess, opponent_vars, e_opponent)
                opponent_effort_reward = opponent_effort_reward + extra_effort_reward
                opponent_click_reward = extra_click_reward

    else:
        if opponent.done:
            # When opponent clicked first, my agent didn't click yet

            if opponent_click_reward == 14:
                # Opponent succeeded first
                # If my_agent succeeds afterwards, give click_reward = 9
                extra_effort_reward, extra_click_reward = compute_extra_reward(my_agent, my_agent_env,
                                                                               my_agent_replay_buffer, my_agent_dqn, my_target_dqn,
                                                                               sess, my_agent_vars, e_my_agent)
                my_effort_reward += extra_effort_reward
                if extra_click_reward == 14:
                    extra_click_reward = 9
                my_click_reward = extra_click_reward
                my_agent_env.fail_rate.append(0)

            else:
                # Opponent failed first
                # Wait until my_agent finishes the episode

                extra_effort_reward, extra_click_reward = compute_extra_reward(my_agent, my_agent_env,
                                                                               my_agent_replay_buffer, my_agent_dqn, my_target_dqn,
                                                                               sess, my_agent_vars, e_my_agent)
                my_effort_reward += extra_effort_reward
                my_click_reward = extra_click_reward

        else:
            # My agent, opponent didn't click yet
            # no need to recompute click reward (click reward == 0)
            pass

    my_reward = my_effort_reward + my_click_reward
    opponent_reward = opponent_effort_reward + opponent_click_reward

    return my_reward, opponent_reward, my_click_reward, opponent_click_reward


def log_data(agent, agent_env, score_logger, episode, click_reward, print_frequency, agent_number):

    error_rate = 1 - (sum(agent_env.error_rate) / len(agent_env.error_rate))
    fail_rate = 1 - (sum(agent_env.fail_rate) / len(agent_env.fail_rate))

    if agent.count == 0:
        score_logger.add_csv(agent.loss, agent.q_value, agent.score, agent_env.time,
                             agent_env.effort, click_reward, episode, error_rate, fail_rate, agent_number)
    else:
        score_logger.add_csv(agent.loss / agent.count, agent.q_value / agent.count, agent.score,
                             agent_env.time, agent_env.effort, click_reward, episode, error_rate, fail_rate, agent_number)

    if episode % print_frequency == 0 or os.path.exists('./check'):
        _, _, _, ave = score_logger.score_show()
        _, _, _, ave_loss = score_logger.loss_show()
        _, _, _, ave_q = score_logger.q_value_show()
        time_mean = sum(agent_env.time_mean) / len(agent_env.time_mean)
        time_std = (sum([((x - time_mean) ** 2) for x in agent_env.time_mean]) / len(agent_env.time_mean)) ** 0.5
        print("Agent {} Episode: {:}, Reward: {:.4}, Loss: {:.4}, Q Value: {:.4}, Time: {:.4} (SD: {:.4}), ER: {:.4}, FR: {:.4}".format(
            agent_number, episode, float(ave), float(ave_loss), float(ave_q), float(time_mean), float(time_std), float(error_rate), float(fail_rate)))


def save_model(model_savepath, score_logger, agent_dqn, episode, agent_type):
    _, score_ave, _, _ = score_logger.score_show()
    _, loss_ave, _, _ = score_logger.loss_show()

    agent_dqn.save(model_savepath, episode, score_ave, loss_ave)
    print("{} model saved".format(agent_type), episode, score_ave, loss_ave)


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory
