import pandas as pd


def compute_reward(action: int,
                   amount: int,
                   next_observation: pd.DataFrame) -> float:
    """
    Compute the rewards associated with a certain action.

    :param action: it represents the instant to sell supposing that buy is done in instant 0.
    :param amount: the amount of contract to buy.
    :param next_observation: the future time series.
    :return: profit/loss
    """
    # TODO: consider fees
    return (next_observation.iloc[action]['Close'] - next_observation.iloc[0]['Close']) * amount
