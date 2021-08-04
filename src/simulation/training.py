import matplotlib.pyplot as plt

from src.data_utils.preprocessing_utils import gadf
from src.simulation.environment import Environment


def training_loop(env: Environment):
    observation = env.reset()
    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(gadf(observation).shape)
        print(reward)
        print(done)
        print(info)
        if done:
            print("info:", info)
            break

    plt.cla()
    env.render_all()
    plt.show()
