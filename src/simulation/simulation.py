import random

from src.simulation.simulation_utils import compute_reward


def simulation(params: dict) -> None:
    """
    Start the simulation.

    :param params: dict with parameters.
    """

    dataframe = params['Dataframe']
    period = params['Period']
    p_l = 0

    for i in range(len(dataframe) - period):
        # extract the observation
        obs = dataframe.iloc[i:i + 2 * period]
        # choice of action
        action = random.randint(1, period)
        # choice of amount
        amount = random.randint(1, 50)
        # compute profit/loss
        p_l += compute_reward(action=action, amount=amount, next_observation=obs.iloc[period - 1:])

        if i % params['Show_every'] == 0:
            print("p_l:", p_l)
