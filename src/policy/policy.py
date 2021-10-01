import torch
import torch.nn as nn
import random

from torchvision.transforms import transforms
from torch.distributions import Categorical
from src.data_utils.preprocessing_utils import StackImages, PermuteImages, GADFTransformation, Rhombus
from scipy.stats import entropy

from src.models.model import resCNN


class RolloutBuffer:
    def __init__(self, len_memory, horizon):
        self.len_memory = len_memory
        self.horizon = horizon

        self.actions = []
        self.states = []
        self.infos = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self, clear=False):
        if len(self.actions) == self.len_memory or clear:
            del self.actions[:]
            del self.states[:]
            del self.infos[:]
            del self.logprobs[:]
            del self.rewards[:]
            del self.is_terminals[:]

    def generate_index(self):
        head = random.randint(self.horizon, len(self.rewards))
        return slice(head - self.horizon, head, 1)


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.actor = resCNN()
        self.critic = resCNN(actor=False)

    def act(self, state, info):
        action_probs = self.actor(state, info)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach(), action_probs.detach()

    def evaluate(self, state, info, action):
        action_probs = self.actor(state, info)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state, info)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, params):

        self.gamma = params['Gamma']
        self.eps_clip = params['EpsClip']
        self.buffer = RolloutBuffer(params['LenMemory'], params['Horizon'])
        self.policy = ActorCritic()
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': params['Lr']},
            {'params': self.policy.critic.parameters(), 'lr': params['Lr']}
        ])
        self.policy_old = ActorCritic()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval()
        self.transform = transforms.Compose([GADFTransformation(periods=params['Periods'],
                                                                pixels=params['Pixels']),
                                             StackImages()])
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, info):

        with torch.no_grad():
            action, action_logprob, action_prob = self.policy_old.act(state, info)

        self.buffer.states.append(state)
        self.buffer.infos.append(info)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item(), action_prob

    def update(self):
        if len(self.buffer.actions) > self.buffer.horizon:
            indexes = self.buffer.generate_index()

            # convert list to tensor
            old_states = torch.squeeze(torch.stack(self.buffer.states[indexes], dim=0)).detach()
            old_infos = torch.squeeze(torch.stack(self.buffer.infos[indexes], dim=0)).detach()
            old_actions = torch.squeeze(torch.stack(self.buffer.actions[indexes], dim=0)).detach()
            old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs[indexes], dim=0)).detach()
            terminals = self.buffer.is_terminals[indexes]
            rewards = torch.tensor(self.buffer.rewards[indexes])
            # self.best_rewards = max(self.best_rewards, max(list(map(abs, rewards))))
            # rewards = rewards / self.best_rewards
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(state=old_states, info=old_infos,
                                                                        action=old_actions)

            """
            hurst_exponents = [old_infos[t, 1].item() for t in range(len(rewards) - 1)]
            mean_hurst = sum(hurst_exponents) / len(hurst_exponents)
            eps = 1 - entropy([mean_hurst, 1 - mean_hurst], base=2)
            """
            returns = []
            future_gae = 0
            for t in reversed(range(len(rewards) - 1)):
                """
                delta = rewards[t] + ((1 - entropy([old_infos[t, 1].item(), (1 - old_infos[t, 1].item())], base=2))
                                      if (not (terminals[t])) else 1.0) * state_values[t + 1] * int(
                    not (terminals[t])) - \
                        state_values[t]
                gaes = future_gae = delta + (
                        1 - entropy([old_infos[t, 1].item(), (1 - old_infos[t, 1].item())], base=2)) * 0.99 * int(
                    not (terminals[t])) * future_gae
                returns.insert(0, gaes + state_values[t])
                # Reinitialization of future_gae at the beginning of a new episode
                future_gae *= int(not (terminals[t]))
                """

                delta = rewards[t] + self.gamma * state_values[t + 1] * int(not (terminals[t])) - state_values[t]
                gaes = future_gae = delta + self.gamma * 0.99 * int(not (terminals[t])) * future_gae
                returns.insert(0, gaes + state_values[t])
                # Reinitialization of future_gae at the beginning of a new episode
                future_gae *= int(not (terminals[t]))

            # Normalizing the rewards
            # returns = [r / max(list(map(abs, returns))) for r in returns]
            rewards = torch.tensor(returns, dtype=torch.float32)
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
            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(state_values[:-1],rewards) \
                   - 0.01 * dist_entropy[:-1].mean()
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            # Copy new weights into old policy
            self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def scheduler(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
