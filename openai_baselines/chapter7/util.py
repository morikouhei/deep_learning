import gym
import os
import numpy as np
import datetime
import pytz
from stable_baselines.results_plotter import load_results, ts2xy

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
best_mean_reward = -np.inf
nupdates = 1

def callback(_locals, _globals):
    global nupdates
    global best_mean_reward

    if (nupdates + 1) % 10 == 0:
        x,y = ts2xy(load_results(log_dir), "timesteps")
        if len(y) > 0:
            mean_reward = np.mean(y[-10:])
            update_model = mean_reward > best_mean_reward
            if update_model:
                best_mean_reward = mean_reward
                _locals["self"].save("sonic_model")

            print("time: {}, nupdates: {}, mean: {:.2f}, best_mean: {:.2f}, model_update: {}".format(datetime.datetime.now(pytz.timezone("Asia/Tokyo")),
                                                                                                     nupdates, mean_reward/0.1, best_mean_reward/0.1, update_model))

        nupdates += 1
        return True

class CustomRewardAndDoneEnv(gym.Wrapper):

    def __init__(self, env):
        super(CustomRewardAndDoneEnv, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs):
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        self._cur_x = info["x"]
        reward = max(0, self._cur_x - self._max_x) * 0.1
        self._max_x = max(self._max_x, self._cur_x)
        if info["lives"] == 2 or info["x"] > 9600:
            done = True

        return state, reward, done, info