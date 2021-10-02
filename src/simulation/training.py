import pandas as pd
from gym_anytrading.envs import Positions
import torch
import numpy as np
from hurst import compute_Hc
from src.policy.policy import PPO
from src.simulation.environment import Environment


def training_loop(env: Environment,
                  policy: PPO):
    step = 1
    position = 0

    observation = env.reset()
    prices = observation[:-1, 4].tolist()
    new_observation = observation[:-1]
    profit_column = np.expand_dims(np.zeros(new_observation.shape[0]), 1)
    last_price_short = observation[-1, 4]
    last_price_long = 0

    while True:
        info_obs = (np.concatenate((new_observation, profit_column), 1), position)
        image = policy.transform(info_obs)
        hurst = compute_Hc(prices, kind='price', simplified=True)[0]

        if position == 1:
            info = torch.tensor([(new_observation[-1, 4] - last_price_long) / last_price_long, hurst],
                                dtype=torch.float32).unsqueeze(dim=0)
        else:
            info = torch.tensor([(last_price_short - new_observation[-1, 4]) / last_price_short, hurst],
                                dtype=torch.float32).unsqueeze(dim=0)

        trade_actions, action_prob = policy.select_action(image, info)

        packed_info = env.step([trade_actions, action_prob])

        profit_column = profit_column[1:]
        profit_column = np.append(profit_column, np.expand_dims(np.zeros(1), 1), axis=0)
        if position == 1 and trade_actions == 1 or position == 1 and trade_actions == 0:
            profit_column[-1] = (packed_info[0][-2, 4] - last_price_long) / last_price_long * env.wallet.cap_inv
        else:
            profit_column[-1] = 0

        del prices[0]
        prices.append(packed_info[0][-2, 4])
        new_position = 0 if env.get_position() in [Positions.Short] else 1

        if position == 0 and new_position == 1:
            last_price_long = new_observation[-1, 4]
        if position == 1 and new_position == 0:
            last_price_short = new_observation[-1, 4]

        policy.buffer.is_terminals.append(packed_info[3]['Done'])
        policy.buffer.rewards.append(packed_info[1])
        if step >= 1:
            if new_position == 0 and position == 1:
                print("step:", step, "tot_reward:", env.wallet.total_gain + env.wallet.total_loss)

        if step % 90 == 0:
            policy.update()
        done = packed_info[2]
        if done:
            pd.DataFrame({"EquityTradingSystem": env.wallet.history["EquityTradingSystem"],
                          "EquityBenchmark": env.wallet.history["EquityBenchmark"],
                          "ProfitLoss": env.wallet.history["ProfitLoss"],
                          "WalletSeries": env.wallet.history["WalletSeries"],
                          "Position": env.wallet.history["Position"],
                          "Done": env.wallet.history["Done"]}).to_csv('report.csv')
            env.render_performances()
            print("done")
            break

        new_observation = packed_info[0][:-1, :]
        position = new_position
        step += 1
