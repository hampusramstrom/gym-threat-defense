
import time

import gym
import gym_threat_defense  # noqa


env = gym.make('threat-defense-v0')

env.reset()

sn, r, done, info = env.step(0)

while not done:
    env.render('rgb_array')
    time.sleep(1)

    sn, r, done, info = env.step(0)
