import gym
import sys
import retro
from gym.spaces import *

# 引数の取得
ENV_ID = sys.argv[1]

# 空間の出力
def print_spaces(label, space):
    # 空間の出力
    print(label, space)

    # Box/Discreteの場合は最大値と最小値も表示
    if isinstance(space, Box):
        print('    最小値: ', space.low)
        print('    最大値: ', space.high)
    if isinstance(space, Discrete):
        print('    最小値: ', 0)
        print('    最大値: ', space.n-1)

# 環境の生成
env = retro.make(game=ENV_ID)

# 行動空間と状態空間の型の出力
print('環境ID: ', ENV_ID)
print_spaces('行動空間: ', env.action_space)
print_spaces('状態空間: ', env.observation_space)