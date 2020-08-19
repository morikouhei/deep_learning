import gym
import time
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from baselines.common.atari_wrappers import *


def main():
    env = gym.make("BowlingNoFrameskip-v0")
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    env = DummyVecEnv([lambda :env])

    dataset = ExpertDataset(expert_path="bowling_demo.npz", verbose=1)

    model = PPO2("CnnPolicy", env, verbose=1)
    model.pretrain(dataset, n_epochs=1000)
    model.save("bowling_model")

    state = env.reset()
    total_reward = 0
    while True:
        env.render()
        time.sleep(1/60)
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        total_reward += reward[0]
        if done:
            print(total_reward)
            state = env.reset()
            total_reward = 0

if __name__ == "__main__":

    main()