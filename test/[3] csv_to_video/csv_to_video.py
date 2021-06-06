"""
Video Generating Module

1) Converts a csv file into video
2) The input csv file must contain per timestep cursor position, target position, and radius

This code was written by Jinhyung Park.
"""


import pandas as pd
import cv2
import os
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from Utils import make_directory

# CSV Data Related Constants
# This module reads the file test_trajectory.csv with columns
# 'time', 'user', 'task', 'trial', 'cursor_x', 'cursor_y', 'target_x', 'target_y', 'radius'
NUM_USERS = 1
NUM_TASKS = 7
NUM_TRIALS = 199

# Output Video Format
VIDEO_SAVEPATH = make_directory(os.path.join('../../videos/Adversarial'))
CSV_NAME = 'trajectory_my_agent_4090000.csv'
CODEC = cv2.VideoWriter_fourcc(*'MPV4')
FPS = 30
VIDEO_WIDTH = 1200
VIDEO_HEIGHT = 1200


def add_cursor_and_target_to_trajectory_frame(cursor_x, cursor_y, target_x, target_y, trajectory_image, target_radius):
    cursor_x = int(VIDEO_WIDTH * cursor_x)
    cursor_y = int(VIDEO_WIDTH * cursor_y)
    target_x = int(VIDEO_WIDTH * target_x)
    target_y = int(VIDEO_WIDTH * target_y)

    new_image = deepcopy(trajectory_image)

    # Cursor (Black)
    cv2.circle(new_image, (cursor_x, cursor_y), 1, (0, 0, 0), -1)

    # Target (Red)
    cv2.circle(new_image, (target_x, target_y), 10, (0, 0, 255), -1)

    return new_image


def update_trajectory_frame(base_image, cursor_x, cursor_y, target_x, target_y, prev_cursor_x, prev_cursor_y, prev_target_x, prev_target_y):

    cursor_x = int(VIDEO_WIDTH * cursor_x)
    cursor_y = int(VIDEO_WIDTH * cursor_y)
    target_x = int(VIDEO_WIDTH * target_x)
    target_y = int(VIDEO_WIDTH * target_y)

    prev_cursor_x = int(VIDEO_WIDTH * prev_cursor_x)
    prev_cursor_y = int(VIDEO_WIDTH * prev_cursor_y)
    prev_target_x = int(VIDEO_WIDTH * prev_target_x)
    prev_target_y = int(VIDEO_WIDTH * prev_target_y)

    # Cursor (Blue)
    cv2.line(base_image,
             # Start Index
             (prev_cursor_x, prev_cursor_y),
             # End Index
             (cursor_x, cursor_y),
             # BGR color - Black
             (0, 0, 0),
             # Thickness
             4)

    # Target (Red)
    cv2.line(base_image,
             # Start Index
             (prev_target_x, prev_target_y),
             # End Index
             (target_x, target_y),
             # BGR color - Red
             (0, 0, 255),
             # Thickness
             4)


def main():
    data = pd.read_csv(f'./{CSV_NAME}', names=['time', 'user', 'task', 'trial',
                                                       'cursor_x', 'cursor_y',
                                                       'target_x', 'target_y',
                                                       'target_radius', 'target_speed', 'click_action', 'click_success'])

    offset = 0
    for user_number in range(NUM_USERS):
        for task_number in range(NUM_TASKS):
            for trial_number in tqdm(range(1, NUM_TRIALS + 1)):

                output_video = cv2.VideoWriter(os.path.join(VIDEO_SAVEPATH, f'User_{user_number}_Task_{task_number}_Trial_{trial_number}.MP4'), CODEC, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

                # Create a white-background image
                trajectory_image = 255 * np.ones((VIDEO_WIDTH, VIDEO_HEIGHT, 3), np.uint8)

                steps = 1
                prev_trial_number = int(data['trial'][offset + steps])
                current_trial_number = int(data['trial'][offset + steps])

                target_radius = float(data['target_radius'][offset + steps])

                while prev_trial_number == current_trial_number:
                    cursor_x = float(data['cursor_x'][offset + steps])
                    cursor_y = float(data['cursor_y'][offset + steps])
                    target_x = float(data['target_x'][offset + steps])
                    target_y = float(data['target_y'][offset + steps])

                    if steps == 1:
                        prev_cursor_x = float(data['cursor_x'][offset + steps])
                        prev_cursor_y = float(data['cursor_y'][offset + steps])
                        prev_target_x = float(data['target_x'][offset + steps])
                        prev_target_y = float(data['target_y'][offset + steps])

                    else:
                        prev_cursor_x = float(data['cursor_x'][offset + steps - 1])
                        prev_cursor_y = float(data['cursor_y'][offset + steps - 1])
                        prev_target_x = float(data['target_x'][offset + steps - 1])
                        prev_target_y = float(data['target_y'][offset + steps - 1])

                    update_trajectory_frame(trajectory_image, cursor_x, cursor_y, target_x, target_y, prev_cursor_x, prev_cursor_y, prev_target_x, prev_target_y)
                    frame = add_cursor_and_target_to_trajectory_frame(cursor_x, cursor_y, target_x, target_y, trajectory_image, target_radius)
                    output_video.write(frame)

                    steps += 1
                    try:
                        current_trial_number = int(data['trial'][offset + steps])

                    except:
                        # Reached Last line
                        break

                output_video.release()
                cv2.destroyAllWindows()
                offset = offset + steps


if __name__ == '__main__':
    main()
