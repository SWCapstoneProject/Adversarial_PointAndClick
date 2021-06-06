# Simulating Point-and-Click Behavior in Implicit Adversarial Environment (2021)

**By <a href="http://github.com/jinhyung426/" target="_blank">Jinhyung Park</a>, <a href="https://github.com/Clap2rap" target="_blank">Hyunwoo Lee</a>, <a href="https://github.com/qwert92a" target="_blank">Gyucheol Shim</a> from Yonsei University (Seoul, Korea)**<br/>
**Supervised by Prof. Byungjoo Lee (Department of Computer Science, Yonsei University)**

<p align="center">
  <img width="650" height="400" src="https://github.com/SWCapstoneProject/Adversarial_PointAndClick/blob/main/misc/teaser.JPG">
</p>

<p align="center">
  <img width="850" height="375" src="https://github.com/SWCapstoneProject/Adversarial_PointAndClick/blob/main/misc/teaser2.png">
</p>

## Introduction
We propose a Point-and-Click Simulator that well respects the nature of the real world Point-and-Click environment where more than 2 agents compete to acquire the same target.
By jointly training 2 agents in an adversarial environment using Reinforcement Learning, we show how the optimal policy of Point-and-Click tasks can change in adversarial environments, compared to non-adversarial environments.
Our comparison with the agent that was trained in a non-adversarial environment (Do et al.) shows that the optimal policy of Point-and-Click tasks in an adversarial environment 
differs in both perspectives - qualitative and quantitative (trial completion time & click failure rate).
The result of an ablation study which trained 2 agents with different human factors also implies the importance of certain human factors.<br/> 

This respository provides the code for training, visualizing, and analyzing the results.<br/>
For more information, please refer presentation_utils or our <a href="https://www.youtube.com/watch?v=DLQu1RDsS6w&t=140s" target="_blank">youtube video</a>. (English version comming soon!) <br/>

## How To Run
### 1. Train
    pip install -r requirements.txt
    mkdir ./saved_models
    mkdir ./outputs
    python train.py --mode=same_agent --model_savepath=./saved_models --csv_savepath=./outputs

### 2. Plot Train History
    python evaluation/train_history_graph/plot_graph.py

### 3. Test & Extract Trajectory Data
    python evaluation/test/test.py

### 4-1. Visualize Result
    python evaluation/video_generator/csv_to_video.py

### 4-2. Visualize and Compare 2 Agents
    python evaluation/video_generator/multiple_csvs_to_video.py

### 5. Plot Correlation Graph
    python evaluation/correlation/correlation_plotter.py

### Reference
   Our study is a follow-up study of 

    S.Do, M. Chang, B.Lee, A Simulation Model of Intermittently Controlled Point-and-Click Behaviour, CHI, 2021.


### Side-note
  1. trajectory_data in evaluation/video_generator and evaluation/correlation are slightly different (generated with different test episodes, but tested with same model)

