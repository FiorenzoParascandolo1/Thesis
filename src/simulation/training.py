import math
import pandas as pd
from gym_anytrading.envs import Positions
import torch
import matplotlib.pyplot as plt
import numpy as np
from hurst import compute_Hc
from scipy.stats import entropy

from src.policy.policy import PPO
from src.simulation.environment import Environment


def training_loop(env: Environment,
                  policy: PPO):
    step = 1
    hit = 0
    tot_op = 1e-5
    tot_reward = 0
    position = 0

    observation = env.reset()[:, 1:]
    new_observation = observation
    profit_column = np.expand_dims(np.zeros(new_observation.shape[0]), 1)
    prices = observation[:-2, 4].tolist()
    last_price_short = observation[-2, 4]
    last_price_long = 0

    while True:
        info_obs = (np.concatenate((new_observation, profit_column), 1)[1:-1], position)
        image = policy.transform(info_obs)
        hurst = compute_Hc(prices, kind='price', simplified=True)[0]

        if position == 1:
            info = torch.tensor([(new_observation[-2, 4] - last_price_long) / last_price_long, hurst],
                                dtype=torch.float32).unsqueeze(dim=0)
        else:
            info = torch.tensor([(last_price_short - new_observation[-2, 4]) / last_price_short, hurst],
                                dtype=torch.float32).unsqueeze(dim=0)

        trade_actions, action_prob = policy.select_action(image, info)
        profit_column = profit_column[1:]
        profit_column = np.append(profit_column, np.expand_dims(np.zeros(1), 1), axis=0)
        if position == 1 and trade_actions == 1 or position == 1 and trade_actions == 0:
            profit_column[-2] = (new_observation[-2, 4] - last_price_long) / last_price_long * env.cap_inv
        else:
            profit_column[-2] = 0
        packed_info = env.step([trade_actions, action_prob])
        del prices[0]
        prices.append(new_observation[-2, 4])
        new_position = 0 if env.get_position() in [Positions.Short] else 1

        if position == 0 and new_position == 0:
            profit_column[-2] = 0
        if position == 0 and new_position == 1:
            last_price_long = new_observation[-2, 4]
            profit_column[-2] = 0
        if position == 1 and new_position == 0:
            last_price_short = new_observation[-2, 4]

        policy.buffer.is_terminals.append(packed_info[3]['done'])
        policy.buffer.rewards.append(packed_info[1])
        if step >= 1:
            if new_position == 0 and position == 1:
                if packed_info[1] > 0:
                    hit += 1
                    tot_op += 1
                if packed_info[1] < 0:
                    tot_op += 1

                reward = packed_info[1]
                tot_reward += reward

                print("step", step, ":")
                print("number of completed trades:", env.tot_operation)
                print("% of profit investments:", env.profit_trades / env.tot_operation)
                print("tot_reward:", env._total_reward)

                print()

        if step % 90 == 0:
            policy.update()
        done = packed_info[2]
        if done:
            pd.Series(env.wallet_series).to_csv('wallet.csv')
            env.render_all()
            print("done")
            break

        new_observation = packed_info[0][:, 1:]
        position = new_position
        step += 1
