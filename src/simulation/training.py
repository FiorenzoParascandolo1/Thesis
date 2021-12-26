import pandas as pd
from gym_anytrading.envs import Positions
from src.data_utils.preprocessing_utils import add_features_on_time, clean_dataframe
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

    df = pd.read_csv(params["FileName"])
    df = clean_dataframe(df)

    window_size = params['Periods'][-1] * params['Pixels'] + 2
    env = Environment(df=df,
                      window_size=window_size,
                      frame_bound=(window_size, len(df)),
                      starting_wallet=df['close'].iloc[window_size - 2] * params['WalletFactor'],
                      bet_size_factor=params['BetSizeFactor'],
                      periods=params['Periods'],
                      pixels=params['Pixels'],
                      manage_symmetries=params['ManageSymmetries'],
                      render=params['Render'],
                      name=params["FileName"].partition('.')[0],
                      wandb=wandb)

    policy = PPO(params, wandb)

    step = 1
    position = 0

    # Reset the environment
    observation = env.reset()

    while True:

        # Select the action
        trade_actions, action_prob, explanation = policy.select_action(observation[0], observation[1])
        # Perform step environment
        packed_info = env.step((trade_actions, action_prob, explanation))

        new_position = 0 if env.get_position() in [Positions.Short] else 1

        # Update buffer with done and reward
        policy.buffer.is_terminals.append(packed_info[4])
        policy.buffer.rewards.append(packed_info[1])

        if packed_info[-1] is not None:
            policy.update_memory_for_finite_trajectories(packed_info[-1])

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
