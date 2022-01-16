import pandas as pd
from torch.nn import functional as F
from src.data_utils.preprocessing_utils import clean_dataframe
from src.policy.policy import PPO
import torch
from src.simulation.environment import Environment
import wandb
from torch.distributions import Categorical
import matplotlib.pyplot as plt


def training_loop(params: dict):
    """
    Training loop for experiments.
    param dict: hyper parameters
    returns:
    """
    wandb.init(project="forex", entity="fiorenzoparascandolo", config=params)

    df = pd.read_csv(params["FileName"], delimiter="\t")
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
                      pip=params['Pip'],
                      leverage=params['Leverage'],
                      gaf=params['Type'],
                      wandb=wandb)

    policy = PPO(params, wandb)

    step = 1
    # Reset the environment
    observation = env.reset()

    while True:

        # Select the action
        if params['Architecture'] not in ['Random']:
            trade_action, action_prob, explanation = policy.select_action(observation[0], observation[1])
        else:
            explanation = None
            action_prob = F.softmax(torch.randn(1, 2), dim=1)
            dist = Categorical(action_prob)
            trade_action = dist.sample()

        # Perform step environment
        packed_info = env.step((trade_action, action_prob, explanation))

        # Update buffer with done and reward
        policy.buffer.is_terminals.append(packed_info[4])
        policy.buffer.rewards.append(packed_info[1])

        # Update the policy
        if len(policy.buffer.actions) >= params['LenMemory']:
            if params['Architecture'] not in ['Random']:
                policy.update()
            else:
                policy.buffer.clear(clear=True)

        done = packed_info[2]

        # If the experiment is finished generate the dataframe for performances
        if done:
            # plt.imshow(policy.policy_old.explanation_tensor)
            # Render performances
            env.wallet.wandb_final()
            # Stop the experiment
            break

        # Update position, observation and step number
        observation = packed_info[0]
        step += 1
