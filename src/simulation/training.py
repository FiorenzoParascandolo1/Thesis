import pandas as pd
from gym_anytrading.envs import Positions
import torch
import numpy as np
from hurst import compute_Hc
from src.policy.policy import PPO
from src.simulation.environment import Environment


def training_loop(env: Environment,
                  policy: PPO):
    """
    Training loop for experiments.
    param env:
    param policy:
    returns:

    TODO: evaluate the benefit of the profit GAF, evaluate the possibility to use Hurst like GAF image
    """

    step = 1
    position = 0
    last_price_long = 0

    # Reset the environment
    observation = env.reset()
    # Store prices to compute Hurst
    prices = observation[:-1, 4].tolist()
    new_observation = observation[:-1]
    profit_column = np.expand_dims(np.zeros(new_observation.shape[0]), 1)
    last_price_short = observation[-1, 4]

    while True:
        # Info_obs is used to create GAF image. TODO: position information is currently not used
        info_obs = (np.concatenate((new_observation, profit_column), 1), position)
        # Transform info_obs in the GAF image
        image = policy.transform(info_obs)
        # Compute Hurst exponent
        hurst = compute_Hc(prices, kind='price', simplified=True)[0]
        # Create information tensor which will be added to the activation for the last FC layer
        if position == 1:
            info = torch.tensor([(new_observation[-1, 4] - last_price_long) / last_price_long,
                                 hurst,
                                 env.shares_months / 100000000,
                                 1],
                                dtype=torch.float32).unsqueeze(dim=0)
        else:
            info = torch.tensor([(last_price_short - new_observation[-1, 4]) / last_price_short,
                                 hurst,
                                 env.shares_months / 100000000,
                                 0],
                                dtype=torch.float32).unsqueeze(dim=0)

        # Select the action
        trade_actions, action_prob = policy.select_action(image, info)
        # Perform step environment
        packed_info = env.step([trade_actions, action_prob])
        # Update profit column
        profit_column = profit_column[1:]
        profit_column = np.append(profit_column, np.expand_dims(np.zeros(1), 1), axis=0)

        # Update profit column
        if position == 1 and trade_actions == 1 or position == 1 and trade_actions == 0:
            profit_column[-1] = (packed_info[0][-2, 4] - last_price_long) / last_price_long * env.wallet.cap_inv
        else:
            profit_column[-1] = 0

        del prices[0]
        prices.append(packed_info[0][-2, 4])
        new_position = 0 if env.get_position() in [Positions.Short] else 1

        # Update last_price_long/last_price_short used to compute additional info
        if position == 0 and new_position == 1:
            last_price_long = new_observation[-1, 4]
        if position == 1 and new_position == 0:
            last_price_short = new_observation[-1, 4]

        # Update buffer with done and reward
        policy.buffer.is_terminals.append(packed_info[4])
        policy.buffer.rewards.append(packed_info[1])

        if packed_info[3] is not None:
            # Check if a transition buy/sell is finished
            if new_position == 0 and position == 1:
                # Print the profit/loss obtained until the current step
                print("step:", step,
                      "tot_reward:", env.wallet.total_gain + env.wallet.total_loss,
                      "commission_paid:", env.wallet.tot_commissions)

        # Update the policy
        if step % 90 == 0:
            policy.update()

        done = packed_info[2]

        # If the experiment is finished generate the dataframe for performances
        if done:
            pd.DataFrame({"EquityTradingSystem": env.wallet.history["EquityTradingSystem"],
                          "EquityBenchmark": env.wallet.history["EquityBenchmark"],
                          "ProfitLoss": env.wallet.history["ProfitLoss"],
                          "WalletSeries": env.wallet.history["WalletSeries"],
                          "Position": env.wallet.history["Position"]}).to_csv('report.csv')
            # Render performances
            env.render_performances()
            # Stop the experiment
            break

        # Update position, observation and step number
        new_observation = packed_info[0][:-1, :]
        position = new_position
        step += 1
