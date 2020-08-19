import gym
import time
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from baselines.common.atari_wrappers import *
from util import callback, log_dir

# 定数
ENV_ID = 'BreakoutNoFrameskip-v0' # 環境ID
NUM_ENV = 8 # 環境数

# 環境を生成する関数
def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env = NoopResetEnv(env, noop_max=30) # 環境リセット後の数ステップ間の行動「Noop」
        env = MaxAndSkipEnv(env, skip=4) # 4フレームごとに行動を選択
        env = FireResetEnv(env) # 環境リセット後の行動「Fire」
        env = WarpFrame(env) # 画面イメージを84x84のグレースケールに変換
        env = ScaledFloatFrame(env) # 状態の正規化
        env = ClipRewardEnv(env) # 報酬の「-1」「0」「1」クリッピング　
        env = EpisodicLifeEnv(env) # ライフ1減でエピソード完了
        if rank == 0:
            env = Monitor(env, log_dir, allow_early_resets=True)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

# メイン関数の定義
def main():
    # 学習環境の生成
    train_env = DummyVecEnv([make_env(ENV_ID, i) for i in range(NUM_ENV)])

    # モデルの生成
    model = PPO2('CnnPolicy', train_env, verbose=0, cliprange=0.1)

    # モデルの読み込み
    # model = PPO2.load('breakout_model', env=train_env, verbose=0)

    # モデルの学習
    model.learn(total_timesteps=1280000, callback=callback)

    # テスト環境の生成
    test_env = DummyVecEnv([make_env(ENV_ID, 9)])

    # モデルのテスト
    state = test_env.reset()
    total_reward = 0
    while True:
        # 環境の描画
        test_env.render()
        time.sleep(1/60)

        # モデルの推論
        action, _ = model.predict(state)

        # 1ステップ実行
        state, reward, done, info = test_env.step(action)

        # エピソードの完了
        total_reward += reward[0]
        if done:
            print('reward:', total_reward)
            state = test_env.reset()
            total_reward = 0

# メインの実行
if __name__ == "__main__":
    main()