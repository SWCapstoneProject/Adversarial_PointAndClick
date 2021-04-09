# Simulating Realistic Point-and-Click Behavior in Result-Driven Adversarial Environment

**By <a href="http://github.com/jinhyung426/" target="_blank">Jinhyung Park</a>, <a href="https://github.com/Clap2rap" target="_blank">Hyeonwoo Lee</a>, <a href="https://github.com/qwert92a" target="_blank">Gyucheol Sim</a> from Yonsei University (Seoul, Korea)**

![teaser](https://github.com/SWCapstoneProject/MulitAgent_PointAndClick/tree/main/utils/teaser.png)

## How To Run
    pip install -r requirements.txt
    python main.py


## Major Update Logs

### 2021.04.09 19:00 - Initial Refactoring
**By Jinhyung Park**

#### 1. Constants.py 
 - Hyperparameter 및 기타 상수 저장
#### 2. main.py 
 - main 함수가 있는 py 파일 (기존의 point_and_click_agent.py 파일을 정리)
 - episode 별로 for loop를 돌 때 매 iteration마다 초기화되는 사항들과 초기화되지 않고 누적되는 사항들을 고려하여서 분리
 - 매 iteration마다 누적되거나 동일한 객체를 사용하는 것들 : 각 agent별 DQN, replay_buffer, score_logger, environment
 - 매 iteration마다 초기화되는 사항들 : 각 agent별 정보(done, count, step_count, loss, q_value, state) (단, 매 iteration마다 동일한 environment를 사용하되, env.reset()은 매번 호출)
 
#### 3. point_and_click_agent.py
 - 각 agent별 정보(done, count, step_count, loss, q_value, state)를 저장하는 agent class (dqn은 별도로 분리)
#### 4. Utils.py
 - agent의 멤버함수로 정의하기 어려운 기타 모든 함수들을 포함
#### 5. point_and_click_env.py
 - Env.seed() 함수에서 seed의 디폴트값이 None 이었던 것을 SEED(=1)라는 상수로 고정
 - 그 외의 부분들은 전부 기존 코드와 동일
#### 6. score_logger.py
 - ScoreLogger.add_csv() 에 agent_number라는 매개변수 추가 (csv 파일을 저장할 때 agent별로 별도의 csv 파일을 생성하기 위함)
 - 그 외의 부분들은 전부 기존 코드와 동일
