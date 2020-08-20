import time
import retro
from baselines.common.retro_wrappers import *
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.vec_env import DummyVecEnv
from util import log_dir, AirstrikerDiscretizer

env = retro.make(game="Airstriker-Genesis", state="Level1")
env = AirstrikerDiscretizer(env)
env = StochasticFrameSkip(env, n=4, stickprob=0.25)
env = Downsample(env, 2)
env = Rgb2gray(env)
env = FrameStack(env, 4)
env = ScaledFloatFrame(env)
env = Monitor(env, log_dir, allow_early_resets=True)

env.seed(0)
set_global_seeds(0)
env = DummyVecEnv([lambda: env])

callback = EvalCallback(env, best_model_save_path='./logs/',
                        log_path='./logs/', eval_freq=10,
                        deterministic=True, render=False)
model = PPO2("CnnPolicy", env, verbose=0)
model.learn(total_timesteps=128000, callback=callback)

state = env.reset()
total_reward = 0
while True:

    env.render()
    time.sleep(1 / 60)
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    total_reward += reward

    if done:
        print(total_reward)
        state = env.reset()
        total_reward = 0
