"""
Video Generating & Comparing Module

1) Compares the result of an agent that was trained in adversarial environment and in non-adversarial environment
2) Converts 2 csv files (generated from different agents) into a single video,
   with each half of the video showing the result of each agent
3) Each input csv file must contain per timestep cursor position, target position, and radius

This code was written by Jinhyung Park.
"""

import pandas as pd
import cv2
import os
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from Utils import make_directory
from PIL import Image

# Output Video Related Constants
VIDEO_SAVEPATH = make_directory(os.path.join('../../videos/Adversarial_VS_Non-Adversarial'))
CSV_OURS = 'trajectory_my_agent_4090000.csv'
CSV_BASELINE = 'trajectory_non_adv_agent_4310000.csv'

# CSV Data Related Constants
NUM_USERS = 1
NUM_TASKS = 1
NUM_TRIALS = 199

# Video Related Constants
CODEC = cv2.VideoWriter_fourcc(*'MPV4')
FPS = 10
VIDEO_WIDTH = 1200
VIDEO_HEIGHT = 1200
WIDTH_MAX = 0.4608
HEIGHT_MAX = 0.2592


def add_cursor_and_target_to_trajectory_frame(cursor_x, cursor_y, target_x, target_y, trajectory_image, target_radius):
    cursor_x = int(VIDEO_WIDTH * (cursor_x / WIDTH_MAX))
    cursor_y = int(VIDEO_WIDTH * (cursor_y / HEIGHT_MAX))
    target_x = int(VIDEO_WIDTH * (target_x / WIDTH_MAX))
    target_y = int(VIDEO_WIDTH * (target_y / HEIGHT_MAX))

    new_image = deepcopy(trajectory_image)

    # Cursor (Black)
    cv2.circle(new_image, (cursor_x, cursor_y), 7, (0, 0, 0), -1)

    # Target (Red)
    cv2.circle(new_image, (target_x, target_y), 30, (0, 0, 255), -1)

    return new_image


def update_trajectory_frame(base_image, cursor_x, cursor_y, target_x, target_y, prev_cursor_x, prev_cursor_y, prev_target_x, prev_target_y):

    cursor_x = int(VIDEO_WIDTH * (cursor_x / WIDTH_MAX))
    cursor_y = int(VIDEO_WIDTH * (cursor_y / HEIGHT_MAX))
    target_x = int(VIDEO_WIDTH * (target_x / WIDTH_MAX))
    target_y = int(VIDEO_WIDTH * (target_y / HEIGHT_MAX))

    prev_cursor_x = int(VIDEO_WIDTH * (prev_cursor_x / WIDTH_MAX))
    prev_cursor_y = int(VIDEO_WIDTH * (prev_cursor_y / HEIGHT_MAX))
    prev_target_x = int(VIDEO_WIDTH * (prev_target_x / WIDTH_MAX))
    prev_target_y = int(VIDEO_WIDTH * (prev_target_y / HEIGHT_MAX))

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
    data_ours = pd.read_csv(f'./{CSV_OURS}', names=['time', 'user', 'task', 'trial',
                                                    'cursor_x', 'cursor_y',
                                                    'target_x', 'target_y',
                                                    'target_radius', 'target_speed', 'click_action', 'click_success'])

    data_baseline = pd.read_csv(f'./{CSV_BASELINE}', names=['time', 'user', 'task', 'trial',
                                                            'cursor_x', 'cursor_y',
                                                            'target_x', 'target_y',
                                                            'target_radius', 'target_speed', 'click_action', 'click_success'])

    offset_ours = 0
    offset_baseline = 0

    for user_number in range(NUM_USERS):
        for task_number in range(NUM_TASKS):
            for trial_number in tqdm(range(1, NUM_TRIALS + 1)):

                output_video = cv2.VideoWriter(os.path.join(VIDEO_SAVEPATH, f'User_{user_number}_Task_{task_number}_Trial_{trial_number}.MP4'), CODEC, FPS, (VIDEO_WIDTH * 2 + 150, VIDEO_HEIGHT + 100))

                steps_ours = 1
                steps_baseline = 1

                # Create a white-background image
                trajectory_image = 255 * np.ones((VIDEO_WIDTH, VIDEO_HEIGHT, 3), np.uint8)
                trajectory_image_baseline = 255 * np.ones((VIDEO_WIDTH, VIDEO_HEIGHT, 3), np.uint8)

                prev_trial_number = int(data_ours['trial'][offset_ours + steps_ours])
                current_trial_number = int(data_ours['trial'][offset_ours + steps_ours])

                prev_trial_number_baseline = int(data_baseline['trial'][offset_baseline + steps_baseline])
                current_trial_number_baseline = int(data_baseline['trial'][offset_baseline + steps_baseline])

                target_radius = float(data_ours['target_radius'][offset_ours + steps_ours])
                target_radius_baseline = float(data_baseline['target_radius'][offset_baseline + steps_baseline])

                local_frames_ours = []
                local_frames_baseline = []

                while prev_trial_number == current_trial_number:
                    cursor_x = float(data_ours['cursor_x'][offset_ours + steps_ours])
                    cursor_y = float(data_ours['cursor_y'][offset_ours + steps_ours])
                    target_x = float(data_ours['target_x'][offset_ours + steps_ours])
                    target_y = float(data_ours['target_y'][offset_ours + steps_ours])

                    if steps_ours == 1:
                        prev_cursor_x = float(data_ours['cursor_x'][offset_ours + steps_ours])
                        prev_cursor_y = float(data_ours['cursor_y'][offset_ours + steps_ours])
                        prev_target_x = float(data_ours['target_x'][offset_ours + steps_ours])
                        prev_target_y = float(data_ours['target_y'][offset_ours + steps_ours])

                    else:
                        prev_cursor_x = float(data_ours['cursor_x'][offset_ours + steps_ours - 1])
                        prev_cursor_y = float(data_ours['cursor_y'][offset_ours + steps_ours - 1])
                        prev_target_x = float(data_ours['target_x'][offset_ours + steps_ours - 1])
                        prev_target_y = float(data_ours['target_y'][offset_ours + steps_ours - 1])

                    update_trajectory_frame(trajectory_image, cursor_x, cursor_y, target_x, target_y, prev_cursor_x, prev_cursor_y, prev_target_x, prev_target_y)
                    frame_ours = add_cursor_and_target_to_trajectory_frame(cursor_x, cursor_y, target_x, target_y, trajectory_image, target_radius)
                    local_frames_ours.append(frame_ours)

                    steps_ours += 1
                    try:
                        current_trial_number = int(data_ours['trial'][offset_ours + steps_ours])
                    except:
                        # Reached Last line
                        break

                while prev_trial_number_baseline == current_trial_number_baseline:
                    cursor_x_baseline = float(data_baseline['cursor_x'][offset_baseline + steps_baseline])
                    cursor_y_baseline = float(data_baseline['cursor_y'][offset_baseline + steps_baseline])
                    target_x_baseline = float(data_baseline['target_x'][offset_baseline + steps_baseline])
                    target_y_baseline = float(data_baseline['target_y'][offset_baseline + steps_baseline])

                    if steps_baseline == 1:
                        prev_cursor_x_baseline = float(data_baseline['cursor_x'][offset_baseline + steps_baseline])
                        prev_cursor_y_baseline = float(data_baseline['cursor_y'][offset_baseline + steps_baseline])
                        prev_target_x_baseline = float(data_baseline['target_x'][offset_baseline + steps_baseline])
                        prev_target_y_baseline = float(data_baseline['target_y'][offset_baseline + steps_baseline])

                    else:
                        prev_cursor_x_baseline = float(data_baseline['cursor_x'][offset_baseline + steps_baseline - 1])
                        prev_cursor_y_baseline = float(data_baseline['cursor_y'][offset_baseline + steps_baseline - 1])
                        prev_target_x_baseline = float(data_baseline['target_x'][offset_baseline + steps_baseline - 1])
                        prev_target_y_baseline = float(data_baseline['target_y'][offset_baseline + steps_baseline - 1])

                    update_trajectory_frame(trajectory_image_baseline, cursor_x_baseline, cursor_y_baseline, target_x_baseline, target_y_baseline,
                                            prev_cursor_x_baseline, prev_cursor_y_baseline, prev_target_x_baseline, prev_target_y_baseline)

                    frame_baseline = add_cursor_and_target_to_trajectory_frame(cursor_x_baseline, cursor_y_baseline, target_x_baseline, target_y_baseline, trajectory_image_baseline, target_radius_baseline)
                    local_frames_baseline.append(frame_baseline)

                    steps_baseline += 1
                    try:
                        current_trial_number_baseline = int(data_baseline['trial'][offset_baseline + steps_baseline])
                    except:
                        # Reached Last line
                        break

                ours_ptr = 0
                baseline_ptr = 0

                while ours_ptr < len(local_frames_ours) and baseline_ptr < len(local_frames_baseline):
                    merged = Image.new(mode="RGB", size=(VIDEO_WIDTH * 2 + 150, VIDEO_HEIGHT + 100), color=(0, 0, 0))
                    merged.paste(im=Image.fromarray(local_frames_ours[ours_ptr]), box=(50, 50))
                    merged.paste(im=Image.fromarray(local_frames_baseline[baseline_ptr]), box=(VIDEO_WIDTH + 100, 50))
                    output_video.write(np.array(merged))
                    ours_ptr += 1
                    baseline_ptr += 1

                if ours_ptr == len(local_frames_ours):
                    # ours frame exhausted - use the last frame
                    while baseline_ptr < len(local_frames_baseline):
                        merged = Image.new(mode="RGB", size=(VIDEO_WIDTH * 2 + 150, VIDEO_HEIGHT + 100), color=(0, 0, 0))
                        merged.paste(im=Image.fromarray(local_frames_ours[-1]), box=(50, 50))
                        merged.paste(im=Image.fromarray(local_frames_baseline[baseline_ptr]), box=(VIDEO_WIDTH + 100, 50))
                        output_video.write(np.array(merged))
                        baseline_ptr += 1

                elif baseline_ptr == len(local_frames_baseline):
                    # baseline frame exhausted - use the last frame
                    while ours_ptr < len(local_frames_ours):
                        merged = Image.new(mode="RGB", size=(VIDEO_WIDTH * 2 + 150, VIDEO_HEIGHT + 100), color=(0, 0, 0))
                        merged.paste(im=Image.fromarray(local_frames_ours[ours_ptr]), box=(50, 50))
                        merged.paste(im=Image.fromarray(local_frames_baseline[-1]), box=(VIDEO_WIDTH + 100, 50))
                        output_video.write(np.array(merged))
                        ours_ptr += 1

                # To look for last screen
                for i in range(int(FPS * 1.5)):
                    merged = Image.new(mode="RGB", size=(VIDEO_WIDTH * 2 + 150, VIDEO_HEIGHT + 100), color=(0, 0, 0))
                    merged.paste(im=Image.fromarray(local_frames_ours[-1]), box=(50, 50))
                    merged.paste(im=Image.fromarray(local_frames_baseline[-1]), box=(VIDEO_WIDTH + 100, 50))
                    output_video.write(np.array(merged))

                output_video.release()
                cv2.destroyAllWindows()

                offset_ours = offset_ours + steps_ours
                offset_baseline = offset_baseline + steps_baseline


if __name__ == '__main__':
    main()
