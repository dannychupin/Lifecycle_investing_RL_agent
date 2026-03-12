import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import torch.optim as optim
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device is:", device)

class PolicyNetwork(nn.Module):
    def __init__(self, s_size, a_size, fc1_units=64, fc2_units=64, alpha_min=1.01):
        super().__init__()
        self.alpha_min = alpha_min
        
        self.mlp_base = nn.Sequential(
            nn.Linear(in_features=s_size, out_features=fc1_units),
            nn.Tanh(),
            nn.Linear(in_features=fc1_units, out_features=fc2_units),
            nn.Tanh()
        )
        self.actor_mean_head = nn.Linear(in_features=fc2_units, out_features=a_size)
        self.actor_concentration_head = nn.Linear(in_features=fc2_units, out_features=1)

    def forward(self, state):
        h = self.mlp_base(state)

        mean_action = F.softmax(self.actor_mean_head(h), dim=-1)

        # Clamp concentration BEFORE softplus to prevent overflow.
        raw_concentration_logit = torch.clamp(
            self.actor_concentration_head(h), min=-20.0, max=20.0
        )
        concentration_action = F.softplus(raw_concentration_logit)

        alpha_action = (concentration_action * mean_action) + self.alpha_min

        self.dist = Dirichlet(alpha_action)
        return mean_action

    def get_dist(self, state):
        self.forward(state)
        return self.dist

    def select_greedy_action(self, state):
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mu = self.forward(state_t)
        return mu.squeeze(0).detach().cpu().numpy()

class ValueNetwork(nn.Module):
    def __init__(self, s_size, fc1_units=64, fc2_units=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(s_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PPOAgent():
    def __init__(
        self,
        state_size,
        action_size,
        lr=3e-5,
        v_lr=5e-5,
        clip_epsilon=0.2,
        ppo_epochs=10,
        minibatch_size=64,
        lam=0.95,
        gamma=0.99,
        value_coeff=0.5,
        entropy_coeff=0.01,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.v_lr = v_lr
        self.pnetwork = PolicyNetwork(state_size, action_size).to(device)
        self.vnetwork = ValueNetwork(state_size).to(device)
        self.poptimizer = optim.Adam(self.pnetwork.parameters(), lr=self.lr)
        self.voptimizer = optim.Adam(self.vnetwork.parameters(), lr=self.v_lr)

        self.clear_buffers()

        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.lam = lam
        self.gamma = gamma
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

    def clear_buffers(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def get_action_and_value(self, state):
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            dist = self.pnetwork.get_dist(state_t)
            value_t = self.vnetwork(state_t).squeeze(0)
            
            raw_action = dist.sample()
            action_t = torch.clamp(raw_action, min=1e-5)
            action_t = action_t / action_t.sum(dim=-1, keepdim=True)
            
            logprob_t = dist.log_prob(action_t).sum(dim=-1)
            
        return action_t.squeeze(0).cpu().numpy(), logprob_t.squeeze(0).cpu().item(), value_t.cpu().item()

    def store_transition(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))

    def compute_gae(self, last_value):
        raw_rewards = np.array(self.rewards, dtype=np.float32)
        rewards = (raw_rewards - raw_rewards.mean()) / (raw_rewards.std() + 1e-8)
        values = self.values
        dones = self.dones
        gamma = self.gamma
        lam = self.lam

        N = len(rewards)
        advantages = np.zeros(N, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(N)):
            next_value = last_value if t == N - 1 else values[t + 1]
            next_nonterminal = 0.0 if dones[t] else 1.0

            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values, dtype=np.float32)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=device)

        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std(unbiased=False) + 1e-8)
        return adv_tensor, ret_tensor

    def update(self, last_value):
        if len(self.states) == 0:
            return

        advantages, returns = self.compute_gae(last_value)
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)

        N = states.shape[0]
        batch_size = self.minibatch_size

        for epoch in range(self.ppo_epochs):
            perm = np.random.permutation(N)
            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                idx = torch.tensor(idx, dtype=torch.long, device=device)

                b_states = states[idx]
                b_actions = actions[idx]
                b_oldlogp = old_log_probs[idx]
                b_adv = advantages[idx]
                b_ret = returns[idx]

                b_actions_safe = torch.clamp(b_actions, min=1e-5)
                b_actions_safe = b_actions_safe / b_actions_safe.sum(dim=-1, keepdim=True)

                dist = self.pnetwork.get_dist(b_states)
                new_logp = dist.log_prob(b_actions_safe).sum(dim=-1)
                entropy = dist.entropy().mean()

                # CLAMP THE EXPONENT directly to prevent backward pass NaN explosions.
                # ln(10) is roughly 2.302585
                log_ratio = torch.clamp(new_logp - b_oldlogp, max=2.302585)
                ratio = torch.exp(log_ratio)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = self.vnetwork(b_states).squeeze(-1)
                value_loss = F.mse_loss(value_pred, b_ret)

                self.poptimizer.zero_grad()
                (policy_loss - self.entropy_coeff * entropy).backward()
                nn.utils.clip_grad_norm_(self.pnetwork.parameters(), 0.5)
                self.poptimizer.step()

                self.voptimizer.zero_grad()
                (self.value_coeff * value_loss).backward()
                nn.utils.clip_grad_norm_(self.vnetwork.parameters(), 0.5)
                self.voptimizer.step()

        self.clear_buffers()

    def save_checkpoint(self, p_path='policy.pth', v_path='value.pth'):
        torch.save(self.pnetwork.state_dict(), p_path)
        torch.save(self.vnetwork.state_dict(), v_path)