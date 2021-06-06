import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

import point_and_click_model as pac
import modules.motor_control_module as motor
import modules.visual_perception_module as visual

from collections import deque

SEED = 1


class Env(gym.Env):
    """
    Description:
        TBD
    Source:
        TBD
    Observation:
        Type: Box(9)
        Num	Observation                 Min         Max
        0	Cursor Position X           -1          1 (m)     # Cursor
        1	Cursor Position Y           -1          1 (m)
        2   Cursor Velocity X           -Inf        Inf (m/s)
        3   Cursor Velocity Y           -Inf        Inf (m/s)
        4	Target Position X           -1          1 (m)     # Target
        5	Target Position Y           -1          1 (m)
        6   Target Velocity X           -0.5        0.5 (m/s)
        7   Target Velocity Y           -0.5        0.5 (m/s)
        8   Target Radius                0.0096     0.024 (m)
        9   Hand Position X             -1.5        1.5 (m)
        10  Hand Position Y             -1.5        1.5 (m)

    Actions:
        Type: Discrete(50)
        Num	Action
        // Actions are being changed
        0   Th = Tp + 0 s,      Click decision (K) = 0
        1   Th = Tp + 0.1 s,    Click decision (K) = 0          : Changing the Th
        ...
        24  Th = Tp + 2.4 s,    Click decision (K) = 0          : Changing the Th
        25  Th = Tp + 0 s,      Click decision (K) = 0          : Changing the Click decision K
        26  Th = Tp + 0.1 s,    Click decision (K) = 1          : Changing the Th
        ...
        48  Th = Tp + 2.3 s,    Click decision (K) = 1          : Changing the Th
        49  Th = Tp + 2.4 s,    Click decision (K) = 1          : Changing the Th

    Reward:
        Click Success Reward (14) - Sum of the acceleration     : When the cursor successes to click and catch the target
        Click Failure Reward (-1) - Sum of the acceleration     : When the cursor clicks the target but fails to catch the target
        - Sum of the acceleration                               : Any other steps

    Starting State:
        All observations are assigned a uniform random value in window
    Episode Termination:
        When the cursor clicks the target
    """

    def __init__(self, agent_name='agent', nc=[0.2, 0.02], cMu=0.185, cSigma=0.09015, nu=19.931, delta=0.399, sigma=0.15):

        # User Parameters for BUMP model
        self.name = agent_name
        self.Tp = 0.1  # Planning time
        self.nc = nc   # Motor noise parameter

        # User Parameters for ICP model
        self.cMu = cMu
        self.cSigma = cSigma
        self.nu = nu
        self.delta = delta
        self.fixed = False

        # Visual Perception Related Parameters
        self.sigma = sigma

        # Hand to Mouse Parameters
        self.forearm = 0.257
        self.mouseGain = 1

        # Action Parameters
        self.Th = self.Tp + (np.arange(25.0) * 0.1)
        self.ThresholdID = (np.arange(2.0) * 1)
        self.action_size = len(self.Th) * len(self.ThresholdID)

        # Simulation Parameter
        self.Interval = 0.05
        self.p = 1

        # Space Boundary
        self.window_width = 0.4608
        self.window_height = 0.2592
        low = np.array([-1, -1, -np.finfo(np.float32).max, -np.finfo(np.float32).max, -1, -1, -0.5, -0.5, 0.0096, -1, -1])
        high = np.array([1, 1, np.finfo(np.float32).max, np.finfo(np.float32).max, 1, 1, 0.5, 0.5, 0.024, 1, 1])

        self.action_space = spaces.Discrete(self.action_size)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed(seed=SEED)
        self.viewer = None
        self.state = np.concatenate((self.np_random.uniform(low=0, high=self.window_width, size=(1,)),
                                     self.np_random.uniform(low=0, high=self.window_height, size=(1,)),
                                     self.np_random.uniform(low=-1, high=1, size=(2,)),
                                     self.np_random.uniform(low=0, high=self.window_width, size=(1,)),
                                     self.np_random.uniform(low=0, high=self.window_height, size=(1,)),
                                     self.np_random.uniform(low=-0.36, high=0.36, size=(2,)),
                                     self.np_random.uniform(low=0.0096, high=0.024, size=(1,)),
                                     self.np_random.uniform(low=-0.12, high=0.12, size=(2,))), axis=None)

        self.init_run = True
        self.time = 0
        self.effort = 0
        self.click = 0
        self.time_mean = deque(maxlen=1000)
        self.error_rate = deque(maxlen=1000)
        self.fail_rate = deque(maxlen=1000)

        # True values
        self.cursorPos = [0, 0]
        self.cursorVel = [0, 0]
        self.targetPos = [0, 0]
        self.targetVel = [0, 0]
        self.handPos = [0, 0]

        self.effortWeight = 1
        self.timeWeight = 0
        self.clickWeight = 14
        self.clickFailWeight = -1

    def seed(self, seed=SEED):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # State space
        c_pos_x, c_pos_y, c_vel_x, c_vel_y, t_pos_x, t_pos_y, t_vel_x, t_vel_y, target_radius, h_pos_x, h_pos_y = self.state

        # Initial distance and speed
        if self.init_run:
            self.time = 0
            self.effort = 0
            self.click = 0
            self.init_run = False

            # Mouse clutching
            hand_boundary = (self.forearm / 2)
            hand_dist = (h_pos_x ** 2 + h_pos_y ** 2) ** 0.5
            if hand_dist > hand_boundary:
                clutch_time = np.random.normal(0.1898, 0.079, 1)[0]
                if clutch_time < 0: clutch_time = 0
                clutch_idx = int(np.ceil(clutch_time / self.Interval))

                # User value
                t_vel_x, t_vel_y = visual.visual_speed_noise(self.targetVel[0], self.targetVel[1], self.sigma)

                t_pos_x, t_vel_x = motor.boundary(clutch_idx, t_pos_x, t_vel_x, self.Interval, self.window_width, target_radius)
                t_pos_y, t_vel_y = motor.boundary(clutch_idx, t_pos_y, t_vel_y, self.Interval, self.window_height, target_radius)
                self.state = (self.cursorPos[0], self.cursorPos[1], 0, 0, t_pos_x, t_pos_y, t_vel_x, t_vel_y, target_radius, 0, 0)

                # True value
                self.targetPos[0], self.targetVel[0] = motor.boundary(clutch_idx, self.targetPos[0], self.targetVel[0], self.Interval, self.window_width, target_radius)
                self.targetPos[1], self.targetVel[1] = motor.boundary(clutch_idx, self.targetPos[1], self.targetVel[1], self.Interval, self.window_height, target_radius)
                self.cursorVel = [0, 0]
                self.handPos = [0, 0]

                reward = -(self.timeWeight * clutch_time)
                done = False
                return np.array(self.state), reward, 0, done, {}

        # Action space
        threshold_id = self.ThresholdID[action // len(self.Th)]
        th = int(round(self.Th[action % len(self.Th)] / self.Interval))
        tp = int(round(self.Tp / self.Interval))

        # Input of the Point-and-Click model
        state_true = self.cursorPos[0], self.cursorPos[1], self.targetPos[0], self.targetPos[1], self.targetVel[0], self.targetVel[1], target_radius, self.handPos[0], self.handPos[1]
        state_cog = c_pos_x, c_pos_y, c_vel_x, c_vel_y
        para_bump = th, tp, self.nc, self.fixed
        para_icp = threshold_id, self.cMu, self.cSigma, self.nu, self.delta, self.p
        para_env = self.Interval, self.window_width, self.window_height, self.forearm

        # Point-and-Click model
        c_otg_dx, c_otg_dy, c_otg_vel_x, c_otg_vel_y, time_click, cursor_delta, effort, h_pos_x, h_pos_y, vel_p, target_info, hand_delta = \
            pac.model(state_true, state_cog, para_bump, para_icp, para_env, self.sigma)

        active_time = len(c_otg_dx) * self.Interval
        time_reward = -(self.timeWeight * active_time)
        effort_reward = -(self.effortWeight * effort)
        click_reward = 0
        done = False

        # If the click is executed
        if time_click <= active_time:
            index_of_click_timing = math.floor(time_click / self.Interval)
            time1 = (time_click / self.Interval) - index_of_click_timing

            if index_of_click_timing == 0:
                cursor_pos_x = self.cursorPos[0] + time1 * c_otg_dx[0]
                cursor_pos_y = self.cursorPos[1] + time1 * c_otg_dy[0]
            else:
                cursor_pos_x = self.cursorPos[0] + np.sum(c_otg_dx[:index_of_click_timing]) + time1 * c_otg_dx[index_of_click_timing]
                cursor_pos_y = self.cursorPos[1] + np.sum(c_otg_dy[:index_of_click_timing]) + time1 * c_otg_dy[index_of_click_timing]

            target_pos_x, temp_vel_x = motor.boundary(index_of_click_timing, self.targetPos[0], self.targetVel[0], self.Interval, self.window_width, target_radius)
            target_pos_x += time1 * self.Interval * temp_vel_x
            target_pos_y, temp_vel_y = motor.boundary(index_of_click_timing, self.targetPos[1], self.targetVel[1], self.Interval, self.window_height, target_radius)
            target_pos_y += time1 * self.Interval * temp_vel_y

            dist_target_cursor = ((target_pos_x - cursor_pos_x) ** 2 + (target_pos_y - cursor_pos_y) ** 2) ** 0.5

            time_reward = -(self.timeWeight * time_click)
            effort_reward = -(self.effortWeight * effort)
            done = True
            self.time += time_click
            self.time_mean.append(self.time)
            self.effort += effort

            if dist_target_cursor < target_radius:
                click_reward = self.clickWeight
                self.click = click_reward
                self.error_rate.append(1)
                self.fail_rate.append(1)
            else:
                click_reward = self.clickFailWeight
                self.click = click_reward
                self.error_rate.append(0)
                self.fail_rate.append(0)

        # User values
        c_pos_x = self.cursorPos[0] + cursor_delta[0]
        c_pos_y = self.cursorPos[1] + cursor_delta[1]
        c_vel_x = vel_p[0]
        c_vel_y = vel_p[1]
        t_pos_x, t_vel_x = motor.boundary(len(c_otg_dx), target_info[0], target_info[2], self.Interval, self.window_width, target_radius)
        t_pos_y, t_vel_y = motor.boundary(len(c_otg_dy), target_info[1], target_info[3], self.Interval, self.window_height, target_radius)
        h_pos_x_ideal = self.handPos[0] + hand_delta[0]
        h_pos_y_ideal = self.handPos[1] + hand_delta[1]

        # Default
        self.state = (c_pos_x, c_pos_y, c_vel_x, c_vel_y, t_pos_x, t_pos_y, t_vel_x, t_vel_y, target_radius, h_pos_x_ideal, h_pos_y_ideal)

        # True values
        self.cursorPos[0] += np.sum(c_otg_dx)
        self.cursorPos[1] += np.sum(c_otg_dy)
        self.cursorVel[0] = c_otg_vel_x[-1]
        self.cursorVel[1] = c_otg_vel_y[-1]
        self.targetPos[0], self.targetVel[0] = motor.boundary(len(c_otg_dx), self.targetPos[0], self.targetVel[0], self.Interval, self.window_width, target_radius)
        self.targetPos[1], self.targetVel[1] = motor.boundary(len(c_otg_dy), self.targetPos[1], self.targetVel[1], self.Interval, self.window_height, target_radius)
        self.handPos = [h_pos_x, h_pos_y]

        # Final reward
        effort_reward = time_reward + effort_reward

        if time_click > active_time:
            self.time += active_time
            self.effort += effort

        return np.array(self.state), effort_reward, click_reward, done, {}

    def reset(self):
        cp_x, cp_y, cv_x, cv_y, _, _, _, _, _, hp_x, hp_y = self.state
        self.targetPos = np.concatenate((self.np_random.uniform(low=0, high=self.window_width, size=(1,)),
                          self.np_random.uniform(low=0, high=self.window_height, size=(1,))), axis=None)
        self.targetVel = self.np_random.uniform(low=-0.36, high=0.36, size=(2,))
        target_radius = self.np_random.uniform(low=0.0096, high=0.024, size=(1,))
        tp = int(round(self.Tp / self.Interval))

        tp_x, tv_x = motor.boundary(tp, self.targetPos[0], -self.targetVel[0], self.Interval, self.window_width, target_radius)
        tp_y, tv_y = motor.boundary(tp, self.targetPos[1], -self.targetVel[1], self.Interval, self.window_height, target_radius)

        tv_x, tv_y = visual.visual_speed_noise(-tv_x, -tv_y, self.sigma)

        tp_x, tv_x = motor.boundary(tp, tp_x, tv_x, self.Interval, self.window_width, target_radius)
        tp_y, tv_y = motor.boundary(tp, tp_y, tv_y, self.Interval, self.window_height, target_radius)

        self.state = np.concatenate((cp_x, cp_y, cv_x, cv_y, tp_x, tp_y, tv_x, tv_y, target_radius, hp_x, hp_y), axis=None)
        self.init_run = True
        return np.array(self.state)
