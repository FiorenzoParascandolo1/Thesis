import torch
import torch.nn as nn
import random
from torch.distributions import Categorical
import numpy as np
from src.models.model import Vgg, DeepFace, LocallyConnectedNetwork
import random

class RolloutBuffer(object):
    def __init__(self,
                 batch_size: int):
        """
        Agent's memory
        :param len_memory: buffer length.
        :param horizon: length of the time horizon to be considered for sampling
        :return:
        """
        self.batch_size = batch_size
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
        del self.actions[:-1]
        del self.states[:-1]
        del self.infos[:-1]
        del self.logprobs[:-1]
        del self.rewards[:-1]
        del self.is_terminals[:-1]

    def generate_index(self) -> list:
        """
        Extract a consecutive sample of elements from the buffer according to the time horizon considered
        """
        indexes = []
        for i in range(0, len(self.infos), self.batch_size):
            if i + self.batch_size > len(self.infos) - 1:
                indexes.append(slice(i, i + self.batch_size - 1, 1))
            else:
                indexes.append(slice(i, i + self.batch_size, 1))
        random.shuffle(indexes)
        return indexes


class ActorCritic(nn.Module):
    """
    Actor-Critic neural networks used in PPO
    """

    def __init__(self,
                 pixels: int,
                 architecture: str,
                 explanations: int,
                 manage_symmetries: bool):
        super(ActorCritic, self).__init__()
        if architecture in ['DeepFace']:
            self.actor = DeepFace(pixels=pixels)
            self.critic = DeepFace(pixels=pixels, actor=False)
        elif architecture in ['Vgg', 'Random']:
            self.actor = Vgg(pixels=pixels)
            self.critic = Vgg(pixels=pixels, actor=False)
        elif architecture in ['LocallyConnectedNetwork']:
            self.actor = LocallyConnectedNetwork(pixels=pixels)
            self.critic = LocallyConnectedNetwork(pixels=pixels, actor=False)

        self.pixels = pixels
        self.manage_symmetries = manage_symmetries
        self.explanations = explanations
        self.explanation_tensor = np.zeros((pixels * 2, pixels * 2))
        self.step = 0

    def act(self,
            state: torch.Tensor,
            info: torch.Tensor,
            explain: bool) -> tuple:
        """
        Actor network is used to act in the environment (to gain experience)
        :param state: GAF image tensor.
        :param info: info tensor = [current_profit, Hurst, number of shares traded in this month].
        :param explain:
        :return: the chosen action, action_log_prob used during the training, action probs used
        to calculate the amount of capital to be allocated
        """

        explanation = None

        if explain:
            state.requires_grad_()
            info.requires_grad_()
            action_probs = self.actor(state, info)
        else:
            with torch.no_grad():
                action_probs = self.actor(state, info)
        dist = Categorical(action_probs)
        action = dist.sample()

        if explain:
            action_probs[0, action].backward()
            saliency_map = torch.abs(state.grad).squeeze()
            saliency_map = np.array(torch.mean(saliency_map, dim=0))
            saliency_map = 2 * (saliency_map - min(map(min, saliency_map))) \
                           / (max(map(max, saliency_map)) - min(map(min, saliency_map))) - 1

            self.step += 1

            if self.step >= 25000:
                self.explanation_tensor += saliency_map

            explanation = self.compute_explanations(saliency_map)

        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach(), action_probs.detach(), explanation

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
        """
        action_probs = self.actor(state, info)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state, info)

        return action_logprobs, state_values, dist_entropy

    def compute_explanations(self,
                             saliency_map: np.array) -> dict:

        indexes = np.c_[np.unravel_index(np.argpartition(saliency_map.ravel(), -self.explanations)[-self.explanations:],
                                         saliency_map.shape)]

        return self.map_indexes_in_candlesticks(indexes)

    def map_indexes_in_candlesticks(self,
                                    indexes: np.array) -> dict:

        explanations = {}

        for couple in indexes:
            i, j = couple
            img = 0

            if couple[0] < self.pixels and couple[1] < self.pixels:
                img = 1
            elif couple[0] < self.pixels <= couple[1]:
                img = 2 if not self.manage_symmetries else 3
                j -= self.pixels
            elif couple[1] < self.pixels <= couple[0]:
                img = 3 if not self.manage_symmetries else 5
                i -= self.pixels
            elif couple[0] >= self.pixels and couple[1] >= self.pixels:
                img = 4 if not self.manage_symmetries else 7
                i -= self.pixels
                j -= self.pixels

            if self.manage_symmetries and j > i:
                img += 1

            if img not in explanations.keys():
                explanations.update({img: []})
            explanations[img].append((i, j))

        return explanations


class PPO:
    """
    https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self,
                 params: dict,
                 wandb):

        self.gamma = params['Gamma']
        self.eps_clip = params['EpsClip']
        self.values_loss_coefficient = params['ValueLossCoefficient']
        self.entropy_loss_coefficient = params['EntropyLossCoefficient']
        self.lmbda = params['Lambda']
        self.epochs = params['Epochs']
        self.explain = params['Render']
        self.buffer = RolloutBuffer(params['BatchSize'])
        self.policy = ActorCritic(params['Pixels'],
                                  params['Architecture'],
                                  params['Explanations'],
                                  params['ManageSymmetries'])
        self.optimizer = torch.optim.Adam([{'params': self.policy.actor.parameters(), 'lr': params['Lr']},
                                           {'params': self.policy.critic.parameters(), 'lr': params['Lr']}])

        self.policy_old = ActorCritic(params['Pixels'],
                                      params['Architecture'],
                                      params['Explanations'],
                                      params['ManageSymmetries'])

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval()
        self.MseLoss = nn.MSELoss()
        self.wandb = wandb

    def select_action(self,
                      state: torch.Tensor,
                      info: torch.Tensor) -> tuple:
        """
        Actor network is used to act in the environment. Information used at training time are stored in memory.
        :param state: GAF image tensor.
        :param info: info tensor = [current_profit, Hurst, number of shares traded in this month].
        :return: the choosen action to act in the environment, action probs used
        to calculate the amount of capital to be allocated
        """

        action, action_logprob, action_prob, explanation = self.policy_old.act(state, info, self.explain)

        self.buffer.states.append(state)
        self.buffer.infos.append(info)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item(), action_prob, explanation

    def update(self) -> None:
        """
        Update the network
        """
        indexes_batch = self.buffer.generate_index()

        for indexes in indexes_batch:
            # convert list to tensor
            old_states = torch.squeeze(torch.stack(self.buffer.states[indexes], dim=0)).detach()
            old_infos = torch.squeeze(torch.stack(self.buffer.infos[indexes], dim=0)).detach()
            old_actions = torch.squeeze(torch.stack(self.buffer.actions[indexes], dim=0)).detach()
            old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs[indexes], dim=0)).detach()
            terminals = self.buffer.is_terminals[indexes]
            rewards = torch.tensor(self.buffer.rewards[indexes])

            for i in range(self.epochs):

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(state=old_states,
                                                                            info=old_infos,
                                                                            action=old_actions)

                returns = []
                future_gae = 0
                for t in reversed(range(len(rewards) - 1)):
                    delta = rewards[t] + self.gamma * state_values[t + 1] * int(not (terminals[t])) - state_values[t]
                    gaes = future_gae = delta + self.gamma * self.lmbda * int(not (terminals[t])) * future_gae
                    returns.insert(0, gaes + state_values[t])
                    # Reinitialization of future_gae at the beginning of a new episode
                    future_gae *= int(not (terminals[t]))
                returns = torch.tensor(returns, dtype=torch.float32)

                # Normalizing the rewards
                returns = (returns - returns.mean()) / (returns.std() + 1e-7)
                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs[:-1] - old_logprobs[:-1].detach())
                # Finding Surrogate Loss
                advantages = returns - state_values[:-1].detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.values_loss_coefficient * self.MseLoss(state_values[:-1], returns)
                entropy_loss = -self.entropy_loss_coefficient * dist_entropy[:-1].mean()
                # final loss of clipped objective PPO
                loss = policy_loss + value_loss + entropy_loss

                # take gradient step
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

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
