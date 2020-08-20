import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from go_left import GoLeft

# 環境の生成
env = GoLeft()
env = DummyVecEnv([lambda: env])

# モデルの生成
model = PPO2('MlpPolicy', env, verbose=1)

# モデルの読み込み
# model = PPO2.load('go_left_model')

# モデルの学習
model.learn(total_timesteps=12800)

# モデルの保存
model.save('go_left_model')

# モデルのテスト
state = env.reset()
total_reward = 0
while True:
    # 描画
    env.render()

    # モデルの推論
    action, _ = model.predict(state, deterministic=True)

    # 1ステップ実行
    state, reward, done, info = env.step(action)
    total_reward += reward[0]

    # エピソード完了
    if done:
        print('reward:', total_reward)
        state = env.reset()
        total_reward = 0