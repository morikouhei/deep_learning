import retro
import time

# 環境の生成
env = retro.make(game='Airstriker-Genesis', state='Level1')

# ランダム行動による動作確認
state = env.reset()
while True:
    # 環境の描画
    #env.render()

    # スリープ
    time.sleep(1/60)

    # 1ステップ実行
    state, reward, done, info = env.step(env.action_space.sample())
    print('reward:', reward)

    # エピソード完了
    if done:
        print('done')
        state = env.reset()