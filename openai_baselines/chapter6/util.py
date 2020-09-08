import gym
import os
import numpy as np
import datetime
import pytz
from stable_baselines.results_plotter import load_results, ts2xy

# ログフォルダの生成
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)

# コールバック
best_mean_reward = -np.inf
nupdates = 1
def callback(_locals, _globals):
    global nupdates
    global best_mean_reward

    # 10更新毎
    if (nupdates + 1) % 10 == 0:
        # 平均エピソード長、平均報酬の取得
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(y) > 0:
            # 最近10件の平均報酬
            mean_reward = np.mean(y[-10:])

            # 平均報酬がベスト報酬以上の時はモデルを保存
            update_model = mean_reward > best_mean_reward
            if update_model:
                best_mean_reward = mean_reward
                _locals['self'].save('airstriker_model')

            # ログ
            print('time: {}, nupdates: {}, mean: {:.2f}, best_mean: {:.2f}, model_update: {}'.format(
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')),
                nupdates, mean_reward, best_mean_reward, update_model))

    nupdates += 1
    return True


# Airstrikerラッパー
class AirstrikerDiscretizer(gym.ActionWrapper):
    # 初期化
    def __init__(self, env):
        super(AirstrikerDiscretizer, self).__init__(env)
        buttons = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
        actions = [['LEFT'], ['RIGHT'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    # 行動の取得
    def action(self, a):
        return self._actions[a].copy()


# CustomRewardAndDoneラッパー
class CustomRewardAndDoneEnv(gym.Wrapper):
    # 初期化
    def __init__(self, env):
        super(CustomRewardAndDoneEnv, self).__init__(env)

    # ステップ
    def step(self, action):
        state, rew, done, info = self.env.step(action)

        # 報酬の変更
        rew /= 20

        # エピソード完了の変更
        if info['gameover'] == 1:
            done = True

        return state, rew, done, info