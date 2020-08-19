import random
import pyglet
import gym
import time
from pyglet.window import key
from stable_baselines.gail import generate_expert_traj
from baselines.common.atari_wrappers import *

env = gym.make("BowlingNoFrameskip-v0")
env = MaxAndSkipEnv(env, skip=4)
env = WarpFrame(env)
env.render()

win = pyglet.window.Window(width=300, height=100, vsync=False)
key_handler = pyglet.window.key.KeyStateHandler()
win.push_handlers(key_handler)
pyglet.app.platform_event_loop.start()


def get_key_state():
    key_state = set()
    win.dispatch_events()
    for key_code, pressed in key_handler.items():
        if pressed:
            key_state.add(key_code)
    return key_state


while len(get_key_state()) == 0:
    time.sleep(1 / 30)


def human_expert(_state):
    key_state = get_key_state()
    action = 0
    if key.SPACE in key_state:
        action = 1
    elif key.UP in key_state:
        action = 2
    elif key.DOWN in key_state:
        action = 3
    time.sleep(1 / 30)
    env.render()

    return action


generate_expert_traj(human_expert, "bowling_demo", env, n_episodes=1)
