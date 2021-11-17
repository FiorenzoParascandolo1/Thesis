import torch
import torch.nn as nn
import random
import pandas as pd
import math
from torchvision.transforms import transforms
from torch.distributions import Categorical
from src.data_utils.preprocessing_utils import StackImages, PermuteImages, GADFTransformation, Rhombus, ManageSymmetries

from src.models.model import resCNN


class RolloutBuffer(object):
    def __init__(self,
                 len_memory: int,
                 horizon: int):
        """
        Agent's memory

        :param len_memory: buffer length.
        :param horizon: length of the time horizon to be considered for sampling
        :return:
        """
        self.len_memory = len_memory
        self.horizon = horizon

        self.actions = []
        self.states = []
        self.infos = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self,
              clear=False) -> None:
        """
        Clean the memory when it is full or when there is a signal from outside

        :param clear: cleaning signal.
        :return:
        """
        if len(self.actions) == self.len_memory or clear:
            del self.actions[:]
            del self.states[:]
            del self.infos[:]
            del self.logprobs[:]
            del self.rewards[:]
            del self.is_terminals[:]

    def generate_index(self) -> slice:
        """
        Extract a consecutive sample of elements from the buffer according to the time horizon considered
        """
        head = random.randint(self.horizon, len(self.rewards) - 1)
        return slice(head - self.horizon, head, 1)


class Image_transformer(object):
    def __init__(self,
                 periods: list,
                 pixels: int):
        """
        Class to manage transformation of time-series into images

        :param periods: time periods considered
        :param pixels: number of pixels
        :return:
        """
        self.transform = transforms.Compose([GADFTransformation(periods=periods,
                                                                pixels=pixels),
                                             StackImages()])
        self.max_list = [0 for _ in range(5)]
        self.min_list = [math.inf for _ in range(5)]

    def generate_image(self,
                       series: pd.Series) -> torch.Tensor:

        self.max_list = [max(self.max_list[j], max(series[:, j])) for j in range(5)]
        self.min_list = [min(self.min_list[j], min(series[:, j])) for j in range(5)]

        return self.transform((series, self.max_list, self.min_list))


class ActorCritic(nn.Module):
    """
    Actor-Critic neural networks used in PPO
    """

    def __init__(self):
        super(ActorCritic, self).__init__()
        self.actor = resCNN()
        self.critic = resCNN(actor=False)

    def act(self,
            state: torch.Tensor,
            info: torch.Tensor) -> tuple:
        """
        Actor network is used to act in the environment (to gain experience)

        :param state: GAF image tensor.
        :param info: info tensor = [current_profit, Hurst, number of shares traded in this month].
        :return: the chosen action, action_log_prob used during the training, action probs used
        to calculate the amount of capital to be allocated
        TODO: encapsulate 'info' in 'x' somehow
        """
        action_probs = self.actor(state, info)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach(), action_probs.detach()

    def evaluate(self,
                 state: torch.Tensor,
                 info: torch.Tensor,
                 action: int) -> tuple:
        """
        Actor network is used to compute the action probs in order to produce action_logprobs and dist_entropy
        used at training time. Critic network is used to compute the state values used at training time

        :param state: GAF image tensor
        :param info: info tensor
        :param action: the chosen action
        :return: tuple of information used at training time
        TODO: encapsulate 'info' in 'x' somehow
        """
        action_probs = self.actor(state, info)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state, info)

        return action_logprobs, state_values, dist_entropy


class PPO:
    """
    https://arxiv.org/pdf/1707.06347.pdf
    TODO: manage better hyper parameters
    """

    def __init__(self,
                 params: dict):

        self.gamma = params['Gamma']
        self.eps_clip = params['EpsClip']
        self.buffer = RolloutBuffer(params['LenMemory'], params['Horizon'])
        self.policy = ActorCritic()
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': params['Lr']},
            {'params': self.policy.critic.parameters(), 'lr': params['Lr']}])
        self.policy_old = ActorCritic()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval()
        """
        self.transform = Image_transformer(pixels=params['Pixels'],
                                           periods=params['Periods'])
        """
        self.transform = transforms.Compose([GADFTransformation(periods=params['Periods'],
                                                                pixels=params['Pixels']),
                                             ManageSymmetries(pixels=params['Pixels']),
                                             StackImages()])
        self.MseLoss = nn.MSELoss()

    def select_action(self,
                      state: pd.Series,
                      info: torch.Tensor) -> tuple:
        """
        Actor network is used to act in the environment. Information used at training time are stored in memory.

        :param state: GAF image tensor.
        :param info: info tensor = [current_profit, Hurst, number of shares traded in this month].
        :return: the choosen action to act in the environment, action probs used
        to calculate the amount of capital to be allocated
        TODO: encapsulate 'info' in 'x' somehow
        """

        observation = self.transform(state)
        with torch.no_grad():
            action, action_logprob, action_prob = self.policy_old.act(observation, info)

        self.buffer.states.append(observation)
        self.buffer.infos.append(info)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item(), action_prob

    def update(self) -> None:
        """
        Update the network
        """
        if len(self.buffer.actions) > self.buffer.horizon:
            indexes = self.buffer.generate_index()

            # convert list to tensor
            old_states = torch.squeeze(torch.stack(self.buffer.states[indexes], dim=0)).detach()
            old_infos = torch.squeeze(torch.stack(self.buffer.infos[indexes], dim=0)).detach()
            old_actions = torch.squeeze(torch.stack(self.buffer.actions[indexes], dim=0)).detach()
            old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs[indexes], dim=0)).detach()
            terminals = self.buffer.is_terminals[indexes]
            rewards = torch.tensor(self.buffer.rewards[indexes])

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(state=old_states,
                                                                        info=old_infos,
                                                                        action=old_actions)

            returns = []
            future_gae = 0
            for t in reversed(range(len(rewards) - 1)):
                delta = rewards[t] + self.gamma * state_values[t + 1] * int(not (terminals[t])) - state_values[t]
                gaes = future_gae = delta + self.gamma * 0.99 * int(not (terminals[t])) * future_gae
                returns.insert(0, gaes + state_values[t])
                # Reinitialization of future_gae at the beginning of a new episode
                future_gae *= int(not (terminals[t]))
            rewards = torch.tensor(returns, dtype=torch.float32)

            # Normalizing the rewards
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs[:-1] - old_logprobs[:-1].detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values[:-1].detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(state_values[:-1], rewards) \
                   - 0.01 * dist_entropy[:-1].mean()
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            # Copy new weights into old policy
            self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def scheduler(self,
                  lr: float) -> None:
        """
        Update the learning rate
        param lr: new learning rate
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save(self,
             checkpoint_path: str) -> None:
        """
        Save neural network weights
        param checkpoint_path: string path
        """
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self,
             checkpoint_path: str) -> None:
        """
        Load neural network weights
        param checkpoint_path: string path
        """
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
