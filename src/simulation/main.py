from src.simulation.environment import Environment

params = {
    'Provider': "histdata",
    'Instrument': "SPXUSD",
    'Years': [2017, 2018],
    'TimeGroup': "1d",
    'EnvType': "stocks-v0",
    'WindowSize': 10,
    'FrameBound': (10, 300)
}


def main():
    env = Environment(params)
    env.print_information()


if __name__ == "__main__":
    main()
