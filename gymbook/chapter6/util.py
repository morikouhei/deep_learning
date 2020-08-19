import gym
import datetime
import os
import numpy as np
import pytz
from stable_baselines.results_plotter import load_results, ts2xy

# ログフォルダの生成
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)

# グローバル変数
num_update = 0  # 更新数
best_mean_reward = -np.inf  # ベスト平均報酬


# コールバック関数の実装
def callback(_locals, _globals):
    global num_update
    global best_mean_reward

    # 10更新毎の処理
    if (num_update + 1) % 10 == 0:
        # 報酬配列の取得
        _, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(y) > 0:
            # 平均報酬がベスト平均報酬以上の時はモデルを保存
            mean_reward = np.mean(y[-10:])
            update_model = mean_reward > best_mean_reward
            if update_model:
                best_mean_reward = mean_reward
                _locals["self"].save("airstriker_model")
                # ログ
                print('time: {}, num_update: {}, mean: {:.2f}, best_mean: {:.2f}, model_update: {}'.format(
                    datetime.datetime.now(pytz.timezone('Asia/Tokyo')),
                    num_update, mean_reward, best_mean_reward, update_model))
    num_update += 1
    return True

class AirstrikerDiscretizer(gym.ActionWrapper):
    def __init__(self, env):
        super(AirstrikerDiscretizer, self).__init__(env)
        buttons = ["B","A","MODE","START","UP","DOWN","LEFT","RIGHT","C","Y","X","Z"]
        actions = [["LEFT"],["RIGHT"],["B"]]
        self._actions = []
        for action in actions:
            arr = np.array([False]*12)
            for button in buttons:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


class CustomRewardAndDoneEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardAndDoneEnv, self).__init__(env)

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        reward /= 20
        if info["gameover"] == 1:
            done = True
        return state, reward, done, info

