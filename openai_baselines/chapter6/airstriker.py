import retro
import time
import os
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from baselines.common.retro_wrappers import *
from util import log_dir, callback, AirstrikerDiscretizer, CustomRewardAndDoneEnv

# 環境の生成 (1)
env = retro.make(game='Airstriker-Genesis', state='Level1')
env = AirstrikerDiscretizer(env) # 行動空間を離散空間に変換
env = CustomRewardAndDoneEnv(env) # 報酬とエピソード完了の変更
env = StochasticFrameSkip(env, n=4, stickprob=0.25) # スティッキーフレームスキップ
env = Downsample(env, 2) # ダウンサンプリング
env = Rgb2gray(env) # グレースケール
env = FrameStack(env, 4) # フレームスタック
env = ScaledFloatFrame(env) # 状態の正規化
env = Monitor(env, log_dir, allow_early_resets=True)
print('行動空間: ', env.action_space)
print('状態空間: ', env.observation_space)

# シードの指定
env.seed(0)
set_global_seeds(0)

# ベクトル化環境の生成
env = DummyVecEnv([lambda: env])

# モデルの生成
#model = PPO2('CnnPolicy', env, verbose=0)

# モデルの読み込み
model = PPO2.load('airstriker_model', env=env, verbose=0)

# モデルの学習
#model.learn(total_timesteps=128000, callback=callback)

# モデルのテスト
state = env.reset()
total_reward = 0
while True:
    # 環境の描画
    env.render()

    # スリープ
    time.sleep(1/60)

    # モデルの推論
    action, _ = model.predict(state)

    # 1ステップ実行
    state, reward, done, info = env.step(action)
    total_reward += reward[0]

    # エピソード完了
    if done:
        print('reward:', total_reward)
        state = env.reset()
        total_reward = 0