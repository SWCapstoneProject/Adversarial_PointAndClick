"""
Correlation Graph Module

This module plots a correlation graph of trial completion time and click failure rate of two agents.
This module compares all of the following relations.

1) Non-Adversarial-Environment Trained Agent (Baseline) VS Adversarial-Environment Trained Agent (Agent 1 - My_agent)
2) Non-Adversarial-Environment Trained Agent (Baseline) VS Adversarial-Environment Trained Agent (Agent 2 - Opponent)
3) Two agents that were trained in Adversarial Environment (Agent 1 & 2 - My agent & Opponent)
4) Improved Decision Making Skill Agent VS Improved Motor Execution Agent

This code was written by Gyucheol Shim.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CSV File Path
non_adv_click_data = pd.read_csv('trajectory_data/trajectory_non_adv.csv')
adv_my_agent_click_data = pd.read_csv('trajectory_data/trajectory_adv_my_agent.csv')
adv_opponent_click_data = pd.read_csv('trajectory_data/trajectory_adv_opponent.csv')
improved_decision_making_click_data = pd.read_csv('./trajectory/trajectory_improved_Decision_Making.csv')
improved_motor_execution_click_data = pd.read_csv('./trajectory/trajectory_improved_Motor_Execution.csv')


def extract_data(click_datas, do_print):
    """ Extracts data for target radius binning and target speed binning analysis from raw csv file
    """

    user0 = click_datas[click_datas.user == 0]
    user0 = user0[user0.click_action == 1]
    user0 = user0.reset_index(drop=True)

    radius_avg = []
    fail_rate_r = []
    complete_time_r = []
    radius_start = 0.009
    radius_range = 0.001

    for i in range(15):
        radius = user0
        radius = radius[radius.target_radius >= radius_start]
        radius = radius[radius.target_radius < radius_start + radius_range]
        radius = radius.reset_index(drop=True)

        radius_avg.append(radius['target_radius'].mean())
        fail_rate_r.append(1.0 - radius['click_success'].mean())
        complete_time_r.append(radius['time'].mean())

        print('radius( {0:0.3f} ~ {1:0.3f} ) : average: {2:0.4f}  fail_rate: {3:0.4f}%  completion_time_mean: {4:0.4f}s'.format(radius_start, radius_start + radius_range, radius_avg[i] ,fail_rate_r[i], complete_time_r[i]))
        radius_start += radius_range

    corr_radius = pd.DataFrame({'radius_avg': radius_avg,
                                'fail_rate': fail_rate_r,
                                'complete_time': complete_time_r})

    print('\n----------------------------------------------------\n')

    speed_avg = []
    fail_rate_s = []
    complete_time_s = []
    speed_start = 0.00
    speed_range = 0.05

    for i in range(10):
        speed = user0
        speed = speed[speed.target_speed >= speed_start]
        speed = speed[speed.target_speed < speed_start + speed_range]
        speed = speed.reset_index(drop=True)

        speed_avg.append(speed['target_speed'].mean())
        fail_rate_s.append(1.0 - speed['click_success'].mean())
        complete_time_s.append(speed['time'].mean())

        print('speed( {0:0.3f} ~ {1:0.3f} ) : average: {2:0.4f}  fail_rate: {3:0.4f}%  completion_time_mean: {4:0.4f}s'.format(speed_start, speed_start + speed_range, speed_avg[i] ,fail_rate_s[i], complete_time_s[i]))
        speed_start += speed_range

    corr_speed = pd.DataFrame({'speed_avg': speed_avg,
                               'fail_rate': fail_rate_s,
                               'complete_time': complete_time_s})

    if do_print == True:
        corr_radius.plot(kind='scatter', x='radius_avg', y='fail_rate')
        fit_weight = np.polyfit(corr_radius['radius_avg'], corr_radius['fail_rate'], 1) # 'avg' 컬럼을 x값으로, 5 컬럼을 y값으로 하여 1차식으로 피팅한다.
        trend_f = np.poly1d(fit_weight)

        plt.plot(corr_radius['radius_avg'], trend_f(corr_radius['radius_avg']),"r-")
        plt.title("y={:.6f}x+({:.6f})".format(fit_weight[0], fit_weight[1]))
        plt.show()

        corr_speed.plot(kind='scatter', x='speed_avg', y='fail_rate')
        fit_weight = np.polyfit(corr_speed['speed_avg'], corr_speed['fail_rate'], 1) # 'avg' 컬럼을 x값으로, 5 컬럼을 y값으로 하여 1차식으로 피팅한다.
        trend_f = np.poly1d(fit_weight)

        plt.plot(corr_speed['speed_avg'], trend_f(corr_speed['speed_avg']),"r-")
        plt.title("y={:.6f}x+({:.6f})".format(fit_weight[0], fit_weight[1]))
        plt.show()

    corr = {'radius': corr_radius, 'speed': corr_speed}
    return corr


def print_corr_graph(first_str, first_corr, second_str, second_corr):
    """ Plots a correlation graph from binning data
    """
    corr_str = "R = {:.4f}\n( y={:.2f}x + {:.2f} )"
    binning_str = ["Target Radius Binning", "Target Speed Binning"]

    for i, binnig in enumerate(['radius', 'speed']):
        corr_complete_time = pd.DataFrame({first_str: first_corr[binnig]['complete_time'],
                                           second_str: second_corr[binnig]['complete_time']})
        time_df = corr_complete_time.corr()
        print(time_df)
        time_r = time_df[first_str][second_str]

        ct_fit_weight = np.polyfit(first_corr[binnig]['complete_time'], second_corr[binnig]['complete_time'], 1)
        ct_trend_f = np.poly1d(ct_fit_weight)

        corr_fail_rate = pd.DataFrame({first_str: first_corr[binnig]['fail_rate'],
                                       second_str: second_corr[binnig]['fail_rate']})
        fali_rate_df = corr_fail_rate.corr()
        print(fali_rate_df)
        fali_rate_r = fali_rate_df[first_str][second_str]

        fr_fit_weight = np.polyfit(first_corr[binnig]['fail_rate'], second_corr[binnig]['fail_rate'], 1)
        fr_trend_f = np.poly1d(fr_fit_weight)

        plt.rc('font', size=12)
        plt.figure(figsize=(5,5))
        plt.plot([0.0, 1.2], [0.0, 1.2], "lightgray")
        plt.plot(first_corr[binnig]['fail_rate'], fr_trend_f(first_corr[binnig]['fail_rate']), label='fail rate')
        plt.plot(first_corr[binnig]['complete_time'], ct_trend_f(first_corr[binnig]['complete_time']), label='trial completion time')
        plt.text(0.35, 0.15, corr_str.format(fali_rate_r, fr_fit_weight[0], fr_fit_weight[1]))
        plt.text(0.18, 0.8, corr_str.format(time_r, ct_fit_weight[0], ct_fit_weight[1]))
        plt.title(binning_str[i])
        plt.scatter(first_corr[binnig]['fail_rate'], fr_trend_f(first_corr[binnig]['fail_rate']), color='cornflowerblue')
        plt.scatter(first_corr[binnig]['complete_time'], ct_trend_f(first_corr[binnig]['complete_time']), color='darksalmon')
        plt.grid()
        plt.axis('square')
        plt.axis([0.0, 1.2, 0.0, 1.2])
        plt.xlabel(first_str)
        plt.ylabel(second_str)
        plt.legend()
        plt.show()
        plt.clf()

    return


if __name__ == '__main__':

    non_adv_corr = extract_data(non_adv_click_data, True)
    adv_my_agent_corr = extract_data(adv_my_agent_click_data, True)
    adv_opponent_corr = extract_data(adv_opponent_click_data, True)

    print_corr_graph('Non-Adversarial Model', non_adv_corr, 'Adversarial Model', adv_my_agent_corr)
    print_corr_graph('Non-Adversarial Model', non_adv_corr, 'Adversarial Model', adv_opponent_corr)
    print_corr_graph('Adversarial Model-1', adv_my_agent_corr, 'Adversarial Model-2', adv_opponent_corr)

    improved_decision_making_corr = extract_data(improved_decision_making_click_data, True)
    improved_motor_execution_corr = extract_data(improved_motor_execution_click_data, True)
    print_corr_graph('Improved Decision-Making Skill Agent', improved_decision_making_corr, 'Improved Motor Execution Agent', improved_motor_execution_corr)
