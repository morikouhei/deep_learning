import retro
import time

env = retro.make(game="SonicTheHedgehog-Genesis", state="GreenHillzone.Act1")
state = env.reset()
while True:
    env.render()
    time.sleep(1/60)
    state, reward, done, info = env.step(env.action_space.sample())
    print("reward", reward)