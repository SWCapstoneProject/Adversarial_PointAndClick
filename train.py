"""
This code is the modified code from https://github.com/hunkim/ReinforcementZeroToAll/
Double DQN (Nature 2015)
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
Notes:
    The difference is that now there are two DQNs (DQN & Target DQN) for each agent,
    and that there are 2 agents that are jointly trained within an adversarial environment.

    y_i = r_i + 𝛾 * max(Q(next_state, action; 𝜃_target))
    Loss: (y_i - Q(state, action; 𝜃))^2
    Every C step, 𝜃_target <- 𝜃

This code was written by Jinhyung Park, Gyucheol Shim, Hyunwoo Lee.
"""

from Utils import *
from point_and_click_agent import *
import deep_q_network as dqn
from score_logger import ScoreLogger
from collections import deque
import argparse


def main(mode, max_episodes, model_savepath, model_save_period, csv_savepath, csv_save_period, print_frequency):

    my_agent_replay_buffer = deque(maxlen=REPLAY_MEMORY)
    opponent_replay_buffer = deque(maxlen=REPLAY_MEMORY)

    my_agent_score_logger = ScoreLogger(env_name="my_agent", ave_num=1000, save_period=csv_save_period, csv_savepath=csv_savepath)
    opponent_score_logger = ScoreLogger(env_name="opponent", ave_num=1000, save_period=csv_save_period, csv_savepath=csv_savepath)

    if mode == 'same_agent':
        # 2 agents with same human factors
        my_agent_env = Env(agent_name='my_agent')
        opponent_env = Env(agent_name='opponent')
    else:
        # 2 agents with different human factors
        my_agent_env = Env(agent_name='my_agent', nc=[0.2, 0.02], cMu=0.3, cSigma=0.06, nu=40, delta=0.25, sigma=0.15)
        opponent_env = Env(agent_name='opponent', nc=[0.24, 0.024], cMu=0.185, cSigma=0.09015, nu=19.931, delta=0.399, sigma=0.18)

    with tf.Session() as sess:

        my_agent_dqn = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="my_agent")
        opponent_dqn = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="opponent")
        my_target_dqn = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="my_target")
        op_target_dqn = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="op_target")

        sess.run(tf.global_variables_initializer())

        # initial copy q_net -> target_net
        my_agent_vars = get_copy_var_ops(dest_scope_name="my_target", src_scope_name="my_agent")
        opponent_vars = get_copy_var_ops(dest_scope_name="op_target", src_scope_name="opponent")

        sess.run(my_agent_vars)
        sess.run(opponent_vars)

        e_my_agent = 1
        e_opponent = 1

        for episode in range(max_episodes + 1):
            if e_my_agent > E_MIN:
                e_my_agent *= E_DECAY
            if e_opponent > E_MIN:
                e_opponent *= E_DECAY

            my_agent = Agent(name='my_agent')
            opponent = Agent(name='opponent')

            my_agent.state = my_agent_env.reset()
            opponent.state = opponent_env.reset()

            while not (my_agent.done and opponent.done):
                my_agent_q_values = my_agent_dqn.predict(my_agent.state)
                my_agent.q_value += np.mean(my_agent_q_values)

                opponent_q_values = opponent_dqn.predict(opponent.state)
                opponent.q_value += np.mean(opponent_q_values)

                # Get the action
                my_action = np.argmax(my_agent_q_values)
                opponent_action = np.argmax(opponent_q_values)

                if np.random.rand() < e_my_agent:
                    my_action = my_agent_env.action_space.sample()

                if np.random.rand() < e_opponent:
                    opponent_action = opponent_env.action_space.sample()

                # Get new state and reward from environment
                my_next_state, my_effort_reward, my_click_reward, my_agent.done, _ = my_agent_env.step(my_action)
                opponent_next_state, opponent_effort_reward, opponent_click_reward, opponent.done, _ = opponent_env.step(opponent_action)

                # Determine Reward
                my_reward, opponent_reward, my_click_reward, opponent_click_reward = determine_reward(my_agent, opponent,
                                                              my_agent_env, opponent_env,
                                                              my_agent_replay_buffer, opponent_replay_buffer,
                                                              my_effort_reward, my_click_reward,
                                                              opponent_effort_reward, opponent_click_reward,
                                                              my_agent_dqn, opponent_dqn, my_target_dqn, op_target_dqn, sess,
                                                              my_agent_vars, opponent_vars, e_my_agent, e_opponent)

                # Save the experience to our buffer
                my_agent.update_replay_buffer(my_agent_replay_buffer, my_action, my_reward, my_next_state, my_agent_dqn, my_target_dqn)
                opponent.update_replay_buffer(opponent_replay_buffer, opponent_action, opponent_reward, opponent_next_state, opponent_dqn, op_target_dqn)

                if my_agent.step_count % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(my_agent_vars)

                if opponent.step_count % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(opponent_vars)

                my_agent.update_step_result(my_reward, my_next_state)
                opponent.update_step_result(opponent_reward, opponent_next_state)

            # After One Episode has been finished, log the current state of each agent
            log_data(my_agent, my_agent_env, my_agent_score_logger, episode, my_click_reward, print_frequency, agent_number=1)
            log_data(opponent, opponent_env, opponent_score_logger, episode, opponent_click_reward, print_frequency, agent_number=2)

            # Save the model
            if episode % model_save_period == 0 and episode >= model_save_period:
                save_model(model_savepath, my_agent_score_logger, my_agent_dqn, episode, agent_type="My Agent")
                save_model(model_savepath, opponent_score_logger, opponent_dqn, episode, agent_type="Opponent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, default='same_agent', choices=['same_agent', 'diff_agent'])
    parser.add_argument('--max_episodes', required=False, default='5000000')
    parser.add_argument('--model_savepath', required=True)
    parser.add_argument('--model_save_period', required=False, default='10000')
    parser.add_argument('--csv_savepath', required=True)
    parser.add_argument('--csv_save_period', required=False, default='100000')
    parser.add_argument('--print_frequency', required=False, default='10')

    FLAGS = parser.parse_args()

    MODE = FLAGS.mode
    MAX_EPISODES = int(FLAGS.max_episodes)
    MODEL_SAVE_PERIOD = int(FLAGS.model_save_period)
    CSV_SAVE_PERIOD = int(FLAGS.csv_save_period)
    PRINT_FREQUENCY = int(FLAGS.print_frequency)
    MODEL_SAVEPATH = FLAGS.model_savepath
    CSV_SAVEPATH = FLAGS.csv_savepath

    main(MODE, MAX_EPISODES, MODEL_SAVEPATH, MODEL_SAVE_PERIOD, CSV_SAVEPATH, CSV_SAVE_PERIOD, PRINT_FREQUENCY)
