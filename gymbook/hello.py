import gym
from go_left import GoLeft

# 環境の生成
env = GoLeft()

# 1エピソードのループ
state = env.reset()
while True:
    # 環境の描画
    env.render()

    # ランダム行動の取得
    action = env.action_space.sample()

    # 1ステップ実行
    state, reward, done, info = env.step(action)
    print('reward:', reward)

    # エピソード完了
    if done:
        print('done')
        break