"""
PPO Agent for Retirement Portfolio Allocation
==============================================

Replaces the off-policy Actor-Critic with on-policy PPO.

Key improvements over the original ac_agent.py:
  - On-policy: no replay buffer, no target networks, no distribution shift
  - Wealth normalized to O(1) before entering networks
  - Per-dimension std in the Gaussian policy (not isotropic)
  - GAE for stable advantage estimation
  - Larger hidden layers (32 vs 8)
  - Tanh activations (bounded gradients, standard for PPO)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, TransformedDistribution, Dirichlet
from torch.distributions.transforms import StickBreakingTransform
import numpy as np


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """
    State -> distribution on the action simplex.

    'gaussian':  Gaussian in R^{A-1}, then stick-breaking onto the simplex.
                 For 2 assets this is a logistic-normal (sigmoid of a Gaussian).
    'dirichlet': Dirichlet directly on the simplex.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=32, dist_type='gaussian'):
        super().__init__()
        self.dist_type = dist_type

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        if dist_type == 'gaussian':
            self.mean_head = nn.Linear(hidden_dim, action_dim - 1)
            self.log_std_head = nn.Linear(hidden_dim, action_dim - 1)
            self.stick_break = StickBreakingTransform()
        elif dist_type == 'dirichlet':
            self.alpha_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h = self.trunk(state)

        if self.dist_type == 'gaussian':
            mean = self.mean_head(h)
            std = F.softplus(self.log_std_head(h)) + 1e-3
            base = Independent(Normal(mean, std), reinterpreted_batch_ndims=1)
            return TransformedDistribution(base, [self.stick_break])
        else:
            alpha = F.softplus(self.alpha_head(h)) + 0.01
            return Dirichlet(alpha)


class ValueNetwork(nn.Module):
    """State -> scalar value estimate V(s)."""

    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout buffer  (on-policy: filled once, used for update, then cleared)
# ---------------------------------------------------------------------------

class RolloutBuffer:

    def __init__(self):
        self.clear()

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def as_tensors(self):
        return (
            torch.tensor(np.array(self.states),   dtype=torch.float32),
            torch.tensor(np.array(self.actions),   dtype=torch.float32),
            torch.tensor(np.array(self.log_probs), dtype=torch.float32),
            torch.tensor(np.array(self.rewards),   dtype=torch.float32),
            torch.tensor(np.array(self.dones),     dtype=torch.float32),
            torch.tensor(np.array(self.values),    dtype=torch.float32),
        )

    def __len__(self):
        return len(self.states)


# ---------------------------------------------------------------------------
# GAE  (Schulman et al., 2016)
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    advantages = torch.zeros(T)
    last_adv = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]                                  # 0 at episode end
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val * mask - values[t]
        advantages[t] = last_adv = delta + gamma * lam * mask * last_adv

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    Proximal Policy Optimization (clip variant).

    Typical usage:
        agent = PPOAgent(state_dim=3, action_dim=2)

        # collect
        for episode:
            for step:
                action, lp, val = agent.act(state)
                agent.buffer.store(state, action, lp, reward, done, val)

        # update
        stats = agent.update()
    """

    def __init__(self, state_dim, action_dim,
                 hidden_dim=32,
                 dist_type='gaussian',
                 lr=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_eps=0.2,
                 ppo_epochs=4,
                 minibatch_size=128,
                 value_coef=0.5,
                 max_grad_norm=0.5):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.dist_type = dist_type

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim, dist_type)
        self.value_net = ValueNetwork(state_dim, hidden_dim)

        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=lr,
        )
        self.buffer = RolloutBuffer()

    @torch.no_grad()
    def act(self, state_np):
        """Sample an action. Returns (action_np, log_prob, value)."""
        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        dist = self.policy(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value_net(state)
        return action.squeeze(0).numpy(), log_prob.item(), value.item()

    def update(self):
        """Run clipped PPO on the current buffer. Returns dict of mean losses."""
        states, actions, old_lp, rewards, dones, values = self.buffer.as_tensors()

        advantages, returns = compute_gae(
            rewards, values, dones, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(states)
        stats = {'policy_loss': [], 'value_loss': [], 'clip_frac': []}

        for _ in range(self.ppo_epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, self.minibatch_size):
                idx = perm[start : start + self.minibatch_size]

                mb_s   = states[idx]
                mb_a   = actions[idx]
                mb_olp = old_lp[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]

                # -- policy loss --
                dist = self.policy(mb_s)

                # clamp actions off simplex boundary for numerical safety
                safe_a = mb_a.clamp(1e-6, 1 - 1e-6)
                safe_a = safe_a / safe_a.sum(-1, keepdim=True)

                new_lp = dist.log_prob(safe_a)
                ratio = torch.exp(new_lp - mb_olp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # -- value loss --
                value_loss = F.mse_loss(self.value_net(mb_s), mb_ret)

                # -- step --
                loss = policy_loss + self.value_coef * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_(
                #    list(self.policy.parameters()) + list(self.value_net.parameters()),
                #    self.max_grad_norm,
                #)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    cf = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['clip_frac'].append(cf)

        self.buffer.clear()
        return {k: np.mean(v) for k, v in stats.items()}

    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'value_net': self.value_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, weights_only=False)
        self.policy.load_state_dict(ckpt['policy'])
        self.value_net.load_state_dict(ckpt['value_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
