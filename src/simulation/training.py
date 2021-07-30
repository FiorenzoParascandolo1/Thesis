import matplotlib.pyplot as plt

from src.simulation.environment import Environment


def training_loop(env: Environment):
    observation = env.reset()
    while True:
        action = env.env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("info:", info)
            break

    plt.cla()
    env.render_all()
    plt.show()
