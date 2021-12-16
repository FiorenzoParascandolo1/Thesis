import pandas as pd
from gym_anytrading.envs import Positions
from src.data_utils.preprocessing_utils import add_features_on_time
from src.policy.policy import PPO
from src.simulation.environment import Environment
import wandb


def training_loop(params: dict):
    """
    Training loop for experiments.
    param dict: hyper parameters
    returns:
    """
    wandb.init(project="trading_system", entity="fiorenzoparascandolo", config=params)

    df = add_features_on_time(pd.read_csv(params["FileName"]))
    df = df[df.volume != 0]
    window_size = params['Periods'][-1] * params['Pixels'] + 2
    env = Environment(df=df,
                      window_size=window_size,
                      frame_bound=(window_size, len(df)),
                      starting_wallet=df['close'].iloc[window_size - 2] * params['WalletFactor'],
                      bet_size_factor=params['BetSizeFactor'],
                      wandb=wandb)
    policy = PPO(params, wandb)

    step = 1
    position = 0

    # Reset the environment
    observation = env.reset()

    while True:

        # Select the action
        trade_actions, action_prob = policy.select_action(observation[0], observation[1])
        # Perform step environment
        packed_info = env.step((trade_actions, action_prob))

        new_position = 0 if env.get_position() in [Positions.Short] else 1

        # Update buffer with done and reward
        policy.buffer.is_terminals.append(packed_info[4])
        policy.buffer.rewards.append(packed_info[1])

        if new_position == 0 and position == 1 or new_position == 1 and position == 0:
            policy.update_memory_for_finite_trajectories(new_position,
                                                         packed_info[0][0][-2, 4],
                                                         packed_info[0][0][-1, 4],
                                                         env.wallet.cap_inv)
            """
            policy.buffer.infos.append(policy.buffer.infos[-1])
            policy.buffer.logprobs.append(policy.buffer.logprobs[-1])
            policy.buffer.states.append(policy.buffer.states[-1])
            policy.buffer.actions.append(policy.buffer.actions[-1])
            policy.buffer.is_terminals.append(False)
            if new_position == 1:
                policy.buffer.rewards.append((packed_info[0][0][-1, 4] - packed_info[0][0][-2, 4]) /
                                             packed_info[0][0][-2, 4] * env.wallet.cap_inv)
            else:
                policy.buffer.rewards.append((packed_info[0][0][-2, 4] - packed_info[0][0][-1, 4]) /
                                             packed_info[0][0][-2, 4] * env.wallet.cap_inv)
            """

        if packed_info[3] is not None:
            # Check if a transition buy/sell is finished
            if new_position == 0 and position == 1:
                # Print the profit/loss obtained until the current step
                print("step:", step,
                      "tot_reward:", env.wallet.total_gain + env.wallet.total_loss,
                      "commission_paid:", env.wallet.tot_commissions)

        # Update the policy
        if step % params['UpdateTimestamp'] == 0:
            policy.update()

        done = packed_info[2]

        # If the experiment is finished generate the dataframe for performances
        if done:
            """
            pd.DataFrame({"EquityTradingSystem": env.wallet.history["EquityTradingSystem"],
                          "EquityBenchmark": env.wallet.history["EquityBenchmark"],
                          "ProfitLoss": env.wallet.history["ProfitLoss"],
                          "WalletSeries": env.wallet.history["WalletSeries"],
                          "Position": env.wallet.history["Position"]}).to_csv('report.csv')
            """
            # Render performances
            env.render_performances()
            # Stop the experiment
            break

        # Update position, observation and step number
        observation = packed_info[0]
        position = new_position
        step += 1
