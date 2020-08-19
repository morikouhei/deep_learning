import numpy as np
import gym

# 左への移動を学ぶ環境
class GoLeft(gym.Env): # (1)
    # 定数を定義
    GRID_SIZE = 5
    LEFT = 0
    RIGHT = 1

    # 初期化
    def __init__(self):
        super(GoLeft, self).__init__()

        # グリッドのサイズ
        self.grid_size = self.GRID_SIZE

        # 初期位置の指定
        self.agent_pos = self.GRID_SIZE - 1

        # 行動空間と状態空間の型の定義 (2)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=0, high=self.GRID_SIZE - 1, shape=(1,), dtype=np.float32)

    # 環境のリセット (3)
    def reset(self):
        # 初期位置の指定
        self.agent_pos = self.GRID_SIZE - 1

        # 初期位置をfloat32のnumpy配列に変換
        return np.array(self.agent_pos).astype(np.float32)

    # 環境の1ステップ実行 (3)
    def step(self, action):
        # 移動
        if action == self.LEFT:
            self.agent_pos -= 1
        elif action == self.RIGHT:
            self.agent_pos += 1
        else:
            raise ValueError("Received invalid action={}".format(action))
        self.agent_pos = np.clip(self.agent_pos, 0, self.GRID_SIZE)

        # エピソード完了の計算
        done = self.agent_pos == 0

        # 報酬の計算
        reward = 1 if done else -0.1

        return np.array(self.agent_pos).astype(np.float32), reward, done, {}

    # 環境の描画 (3)
    def render(self, mode='console', close=False):
        if mode != 'console':
            raise NotImplementedError()

        # エージェントは「A」、他は「.」で表現
        print("." * self.agent_pos, end="")
        print("A", end="")
        print("." * (self.GRID_SIZE - 1 - self.agent_pos))