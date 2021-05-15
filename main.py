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

from Utils import *
from point_and_click_agent import *
import deep_q_network as dqn
from score_logger import ScoreLogger
from collections import deque

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    # Store the previous observations in replay memory
    my_agent_replay_buffer = deque(maxlen=REPLAY_MEMORY)
    opponent_replay_buffer = deque(maxlen=REPLAY_MEMORY)

    my_agent_score_logger = ScoreLogger(env_name="my_agent", ave_num=1000, save_period=CSV_SAVE_PERIOD)
    opponent_score_logger = ScoreLogger(env_name="opponent", ave_num=1000, save_period=CSV_SAVE_PERIOD)

    my_agent_env = Env()
    opponent_env = Env()

    with tf.Session() as sess:

        my_agent_dqn = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="my_agent")
        opponent_dqn = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="opponent")
        target_dqn = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        sess.run(tf.global_variables_initializer())

        # initial copy q_net -> target_net
        my_agent_vars = get_copy_var_ops(dest_scope_name="target", src_scope_name="my_agent")
        opponent_vars = get_copy_var_ops(dest_scope_name="target", src_scope_name="opponent")

        sess.run(my_agent_vars)
        sess.run(opponent_vars)

        e_my_agent = 1
        e_opponent = 1

        for episode in range(MAX_EPISODES + 1):
            if e_my_agent > E_MIN:
                e_my_agent *= E_DECAY
            if e_opponent > E_MIN:
                e_opponent *= E_DECAY

            my_agent = Agent(name='my_agent')
            opponent = Agent(name='opponent')

            my_agent.state = my_agent_env.reset()
            opponent.state = opponent_env.reset()

            while not (my_agent.done and opponent.done):
                # Get the q table
                my_q_values = my_agent_dqn.predict(my_agent.state)
                my_agent.q_value += np.mean(my_q_values)

                opponent_q_values = opponent_dqn.predict(opponent.state)
                opponent.q_value += np.mean(opponent_q_values)

                # Get the action
                my_action = np.argmax(my_q_values)
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
                                                              my_agent_dqn, opponent_dqn, target_dqn, sess,
                                                              my_agent_vars, opponent_vars, e_my_agent, e_opponent)
                # Save the experience to our buffer
                my_agent.update_replay_buffer(my_agent_replay_buffer, my_action, my_reward, my_next_state, my_agent_dqn, target_dqn)
                opponent.update_replay_buffer(opponent_replay_buffer, opponent_action, opponent_reward, opponent_next_state, opponent_dqn, target_dqn)

                if my_agent.step_count % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(my_agent_vars)

                if opponent.step_count % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(opponent_vars)

                my_agent.update_step_result(my_reward, my_next_state)
                opponent.update_step_result(opponent_reward, opponent_next_state)

            # After One Episode has been finished, log the current state of each agent
            log_data(my_agent, my_agent_env, my_agent_score_logger, episode, my_click_reward, agent_number=1)
            log_data(opponent, opponent_env, opponent_score_logger, episode, opponent_click_reward, agent_number=2)

            # Save the model
            if episode % MODEL_SAVE_PERIOD == 0 and episode >= MODEL_SAVE_PERIOD:
                save_model(my_agent_score_logger, my_agent_dqn, episode, agent_type="My agent")
                save_model(opponent_score_logger, opponent_dqn, episode, agent_type="Opponent")


if __name__ == "__main__":
    main()
