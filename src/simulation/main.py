from src.simulation.environment import Environment
from src.simulation.training import training_loop

params = {
    'Provider': "histdata",
    'Instrument': "SPXUSD",
    'Years': [2017, 2018],
    'TimeGroup': "1d",
    'EnvType': "stocks-v0",
    'WindowSize': 10,
}


def main():
    env = Environment(params)
    training_loop(env)


if __name__ == "__main__":
    main()
