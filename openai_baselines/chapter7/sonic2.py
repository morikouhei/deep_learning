import retro
import os
import time
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from baselines.common.retro_wrappers import *
from stable_baselines.bench import Monitor
from util import CustomRewardAndDoneEnv, callback, log_dir
from stable_baselines.common import set_global_seeds

env = retro.make(game="SonicTheHedgehog-Genesis", state="GreenHillZone.Act1")
env = SonicDiscretizer(env)
env = CustomRewardAndDoneEnv(env)
env = StochasticFrameSkip(env, n=4, stickprob=0.25)
env = Downsample(env, 2)
env = Rgb2gray(env)
env = FrameStack(env, 4)
env = ScaledFloatFrame(env)
env = TimeLimit(env, max_episode_steps=4500)
env = Monitor(env, log_dir, allow_early_resets=True)

env.seed(0)
set_global_seeds(0)
env = DummyVecEnv([lambda : env])

model = PPO2(policy=CnnPolicy, env=env, verbose=0, learning_rate=0.000025)

model.learn(total_timesteps=20000000, callback=callback)

state = env.reset()
total_reward = 0
while True:
    env.render()
    time.sleep(1/120)
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        print("reward", total_reward)
        total_reward = 0
        state = env.reset()