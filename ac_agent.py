"""
    Version: April 4, 2026

    ac = Actor-Critic. Nothing "soft" here.

    Changes from sac_agent:
    - nothing "soft": no alpha, entropy
    - policy sampled variable number of times, instead of once
    - introduced option to clip gradient norms on everything

"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Normal, Independent, TransformedDistribution
from torch.distributions.transforms import StickBreakingTransform

import numpy as np
from copy import deepcopy

# set seed for debugging
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

### I. PARTIAL LAYER NORMALIZATION ###

"""
    Custom partial layer normalization on the last 1 feature of an input of state_dimension (units of $)
"""


class PartialLayerNorm(nn.Module):
    def __init__(self, state_dimension, last_dimension=1):
        super().__init__()
        self.state_dimension = state_dimension
        self.d = state_dimension - last_dimension

        self.ln = nn.LayerNorm(normalized_shape=last_dimension, elementwise_affine=True)

    def forward(self, state):
        first_d = state[..., :self.d]
        last = state[..., self.d:]
        last = self.ln(last)  # normalizes the last_dim block per sample

        return torch.cat([first_d, last], dim=-1)


### II. POLICY NETWORKS ###

"""
    Actor: policy function p: R^{dim_state} -> Dist(R^{dim_action})

    Action space is continuous, dim_action (number of asset returns)
    State space is continuous,
        dim_state =     dim_action (number of asset returns)
                        + dim_ancilla (number of macro variables, like returns to gdp, ...)
                        + 1 (wealth)

    NOTE: returns to cpi (aka inflation) is included into returns on assets and macro variables: these are REAL returns

    Define TWO 'distribution_type's of policy network:

    (1) 'gaussian': produces a Gaussian distribution on R^{dim_action - 1}, and "stick break" transforms it into a
                    distribution on the simplex in R^{dim_action}

    (2) 'dirichlet': produces a Dirichlet distribution directly on the simplex in R^{dim_action},
                     with alpha(s) = mean(s) * concentration(s) + alpha_min
"""


class ActorNetworkGaussian(nn.Module):
    def __init__(self, state_dimension, action_dimension, hidden1_dimension,
                 hidden2_dimension, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.min_std = 1e-3  # for numerical stability
        self.activation = activation

        self.mlp_trunk = nn.Sequential(
            PartialLayerNorm(state_dimension=state_dimension),
            nn.Linear(in_features=state_dimension, out_features=hidden1_dimension),
            self.activation,
            nn.Linear(in_features=hidden1_dimension, out_features=hidden2_dimension),
            self.activation
        )

        # mean in R^{action_dimension - 1}, to be post-processed by stick-breaking into R^{action_dimension}
        self.mean_head = nn.Linear(in_features=hidden2_dimension, out_features=action_dimension - 1)
        self.logstd_head = nn.Linear(in_features=hidden2_dimension, out_features=1)

        self.transform = StickBreakingTransform()

    def forward(self, state):
        h = self.mlp_trunk(state)

        # extract mean and std
        mean = self.mean_head(h)  # (B, A-1)
        logstd = self.logstd_head(h)  # (B, 1)
        std = F.softplus(logstd) + self.min_std
        std = std.expand_as(mean)  # (B, A), for Normal(mean, std)

        # produce the distribution
        base = Independent(Normal(mean, std), 1)  # event_dim=1 means log_prob has shape (B,)
        pi_given_state = TransformedDistribution(base, [self.transform])

        return pi_given_state  # KEY: supports .rsample() and .log_prob()


class ActorNetworkDirichlet(nn.Module):
    def __init__(self, state_dimension, action_dimension, hidden1_dimension,
                 hidden2_dimension, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.min_alpha = 1e-3  # for numerical stability
        self.activation = activation

        self.mlp_trunk = nn.Sequential(
            PartialLayerNorm(state_dimension=state_dimension),
            nn.Linear(in_features=state_dimension, out_features=hidden1_dimension),
            self.activation,
            nn.Linear(in_features=hidden1_dimension, out_features=hidden2_dimension),
            self.activation
        )

        self.mean_head = nn.Linear(in_features=hidden2_dimension, out_features=action_dimension)
        self.log_concentration_head = nn.Linear(in_features=hidden2_dimension, out_features=1)

    def forward(self, state):
        h = self.mlp_trunk(state)

        # extract mean and concentration
        mean = F.softmax(self.mean_head(h), dim=-1)  # (B, A-1)
        log_concentration = self.log_concentration_head(h)  # (B, 1)
        concentration = F.softplus(log_concentration)

        # define alpha
        alpha_of_state = (mean * concentration) + self.min_alpha  # alpha_of_state must be > 0

        pi_given_state = Dirichlet(alpha_of_state)

        return pi_given_state  # KEY: supports .rsample() and .log_prob()


### III. CRITIC NETWORK ###

"""
    Critic approximates Q: R^{dim_state} x R^{dim_action} -> R^1
"""


class CriticNetwork(nn.Module):

    def __init__(self, state_dimension, action_dimension, hidden1_dimension,
                 hidden2_dimension, activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.pln = PartialLayerNorm(state_dimension=state_dimension)
        self.activation = activation

        self.mlp_trunk = nn.Sequential(
            nn.Linear(in_features=state_dimension + action_dimension, out_features=hidden1_dimension),
            self.activation,
            nn.Linear(in_features=hidden1_dimension, out_features=hidden2_dimension),
            self.activation,
            nn.Linear(in_features=hidden2_dimension, out_features=1)
        )

    def forward(self, state, action):
        # apply partial layer norm to state
        state_normalized = self.pln(state)

        # concatenate and feed forward
        state_and_action = torch.cat((state_normalized, action), dim=-1)
        q_value = self.mlp_trunk(state_and_action)

        return q_value  # activate? don't activate?


class ReplayBuffer:
    # lives in Numpy. Need to convert to Torch tensors at sample time

    def __init__(self, max_memory_size, state_dimension, action_dimension, batch_size,
                 replace=False):
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size
        self.replace = replace

        self.memory_counter = 0

        # order for training will be SAS'R', not SARS': reward only comes after the NEW state, hence `new` reward
        self.state_memory = np.zeros((self.max_memory_size, state_dimension), dtype=np.float32)
        self.action_memory = np.zeros((self.max_memory_size, action_dimension), dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_memory_size, state_dimension), dtype=np.float32)
        self.new_reward_memory = np.zeros(self.max_memory_size, dtype=np.float32)

        # done is an int, 1 (yes) or 0 (no)
        self.done_memory = np.zeros(self.max_memory_size, dtype=np.float32)

    def store_transition(self, state, action, new_state, new_reward, is_done: bool):
        # state, action, new_state are np.arrays. new_reward, done are single values

        # get position of first unoccupied memory slot
        index = self.memory_counter % self.max_memory_size

        # add to the memory
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.new_state_memory[index] = new_state
        self.new_reward_memory[index] = new_reward

        # is this transition the end of an episode? is_done = 1 if yes, = 0 if no
        self.done_memory[index] = int(is_done)

        # update memory counter!
        self.memory_counter += 1

    def get_batch(self):
        memory_lower_bound = min(self.max_memory_size, self.memory_counter)

        # sample batch of indices
        batch = np.random.choice(memory_lower_bound, self.batch_size, replace=self.replace)

        # extract memory submatrix with just the rows in the list `batch`:
        state_batch = self.state_memory[batch]
        action_batch = self.action_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        new_reward_batch = self.new_reward_memory[batch]
        done_batch = self.done_memory[batch]

        # outputs are np arrays, not torch tensors
        # recall order is SAS'R', not SARS'
        return state_batch, action_batch, new_state_batch, new_reward_batch, done_batch


class ActorCritic:

    def __init__(self, state_dimension,
                 action_dimension,
                 hidden1_dimension,
                 hidden2_dimension,
                 activation,
                 tau,
                 lr_actor,
                 lr_critic,
                 max_memory_size,
                 batch_size,
                 num_policy_samples,
                 should_clip_grads=False,
                 clip_value=10.0,
                 gamma=0.99,
                 replace=False,
                 distribution_type='gaussian',
                 checkpoint_file_name="AC_model_weights/checkpoint.pt"):

        self.checkpoint_file_name = checkpoint_file_name
        self.action_dimension = action_dimension
        self.state_dimension = state_dimension
        self.gamma = gamma
        self.distribution_type = distribution_type
        self.should_clip_grads = should_clip_grads
        self.clip_value = clip_value

        self.tau = tau  # rate of approach of Q-target networks to true Qs (usually 0.005)

        # initialize replay buffer
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size
        self.num_policy_samples = num_policy_samples
        self.memory = ReplayBuffer(max_memory_size=self.max_memory_size,
                                   state_dimension=self.state_dimension,
                                   action_dimension=self.action_dimension,
                                   batch_size=self.batch_size,
                                   replace=replace)

        # actor. Outputs a distribution on action space
        if distribution_type == 'gaussian':
            self.policy = ActorNetworkGaussian(state_dimension=state_dimension,
                                               action_dimension=action_dimension,
                                               hidden1_dimension=hidden1_dimension,
                                               hidden2_dimension=hidden2_dimension,
                                               activation=activation)
        if distribution_type == 'dirichlet':
            self.policy = ActorNetworkDirichlet(state_dimension=state_dimension,
                                                action_dimension=action_dimension,
                                                hidden1_dimension=hidden1_dimension,
                                                hidden2_dimension=hidden2_dimension,
                                                activation=activation)

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)

        # Q1
        self.Q1 = CriticNetwork(state_dimension=state_dimension,
                                action_dimension=action_dimension,
                                hidden1_dimension=hidden1_dimension,
                                hidden2_dimension=hidden2_dimension,
                                activation=activation)
        self.Q1_optimizer = torch.optim.Adam(self.Q1.parameters(), lr=lr_critic)

        # Q1 target: turn off grad
        self.Q1_target = deepcopy(self.Q1)
        self.Q1_target.requires_grad_(False)
        self.Q1_target.eval()

        # Q2
        self.Q2 = CriticNetwork(state_dimension=state_dimension,
                                action_dimension=action_dimension,
                                hidden1_dimension=hidden1_dimension,
                                hidden2_dimension=hidden2_dimension,
                                activation=activation)
        self.Q2_optimizer = torch.optim.Adam(self.Q2.parameters(), lr=lr_critic)

        # Q2 target: turn off grad
        self.Q2_target = deepcopy(self.Q2)
        self.Q2_target.requires_grad_(False)
        self.Q2_target.eval()

    @staticmethod  # haven't used this yet
    def noisy(state, std=0.1):
        noise = torch.normal(mean=0, std=std, size=state.size())
        return state + noise

    # take tau-step toward parent model for target networks
    def update_target(self, target_model, model):
        with torch.no_grad():  # ensure the non-target model parameters aren't recruited to begin computational graphs
            for target_parameter, parameter in zip(target_model.parameters(), model.parameters()):
                target_parameter.mul_(1 - self.tau).add_(parameter, alpha=self.tau)

    def update(self):
        # if not enough in buffer, skip
        print(f'memory counter: {self.memory.memory_counter} and batch size: {self.memory.batch_size}')
        if self.memory.memory_counter < self.memory.batch_size:
            return
        # extract np batch from buffer. Recall the order is SAS'R'D', not SARS'D'
        state_batch, action_batch, new_state_batch, new_reward_batch, done_batch = self.memory.get_batch()

        # convert to tensors
        state_batch = torch.from_numpy(state_batch)
        action_batch = torch.from_numpy(action_batch)
        new_state_batch = torch.from_numpy(new_state_batch)
        new_reward_batch = torch.from_numpy(new_reward_batch)
        done_batch = torch.from_numpy(done_batch)

        # I. Q UPDATE

        # 1) produce target future rewards values from batch of new states

        self.policy.eval()

        with torch.no_grad():
            new_action_distribution_batch = self.policy.forward(new_state_batch)

            # prepare for construction of target values
            min_of_target_values_batch = torch.zeros_like(new_reward_batch)

            for _ in range(self.num_policy_samples):
                # take batch of samples
                new_action_batch = new_action_distribution_batch.sample()

                # compute target `y` values (squeeze last dimension since Q is scalar)
                Q1_target_values_batch = self.Q1_target.forward(new_state_batch, new_action_batch).squeeze(-1)
                Q2_target_values_batch = self.Q2_target.forward(new_state_batch, new_action_batch).squeeze(-1)

                # update minimum; (B,)
                min_of_target_values_batch = min_of_target_values_batch \
                                             + torch.min(Q1_target_values_batch, Q2_target_values_batch)

            # normalize minimum by number of policy samples
            min_of_target_values_batch = min_of_target_values_batch / self.num_policy_samples

            # calculate target values
            target_future_rewards_batch = new_reward_batch + self.gamma * (1 - done_batch) * min_of_target_values_batch

        # 2) use target values to update the critics Q1 and Q2
        # Q1:
        self.Q1.train()
        self.Q1_optimizer.zero_grad(set_to_none=True)
        Q1_value_batch = self.Q1.forward(state_batch, action_batch).squeeze(-1)
        Q1_loss = F.smooth_l1_loss(target_future_rewards_batch, Q1_value_batch)
        Q1_loss_copy = Q1_loss.detach().clone()  # record Q1 loss to track learning

        Q1_loss.backward()
        if self.should_clip_grads:
            torch.nn.utils.clip_grad_norm_(self.Q1.parameters(), max_norm=self.clip_value)
        self.Q1_optimizer.step()
        self.Q1.eval()

        # Q2:
        self.Q2.train()
        self.Q2_optimizer.zero_grad(set_to_none=True)
        Q2_value_batch = self.Q2.forward(state_batch, action_batch).squeeze(-1)
        Q2_loss = F.smooth_l1_loss(target_future_rewards_batch, Q2_value_batch)
        Q2_loss_copy = Q2_loss.detach().clone()  # record Q2 loss to track learning

        Q2_loss.backward()
        if self.should_clip_grads:
            torch.nn.utils.clip_grad_norm_(self.Q2.parameters(), max_norm=self.clip_value)
        self.Q2_optimizer.step()
        self.Q2.eval()

        # II. POLICY UPDATE

        # 1) zero out the grad on policy
        self.policy.train()
        self.policy_optimizer.zero_grad(set_to_none=True)

        # for speed: freeze parameters on Qs, so that grad only flows through their action arguments
        self.Q1.requires_grad_(False)
        self.Q2.requires_grad_(False)

        # 2) sample actions
        # IMPORTANT: use .rsample() to obtain alternative actions on (old) states
        action_distribution_batch = self.policy.forward(state_batch)
        min_of_Q_value_batch = torch.zeros_like(Q1_value_batch)

        # use updated Q-values to calculate minima by taking rsamples of current policy
        for _ in range(self.num_policy_samples):
            action_batch = action_distribution_batch.rsample()  # overwrite `action batch`
            Q1_value_batch = self.Q1.forward(state_batch, action_batch).squeeze(-1)
            Q2_value_batch = self.Q2.forward(state_batch, action_batch).squeeze(-1)
            min_of_Q_value_batch = min_of_Q_value_batch + torch.min(Q1_value_batch, Q2_value_batch)

        min_of_Q_value_batch = min_of_Q_value_batch / self.num_policy_samples

        # 3) calculate loss, and take a step. IMPORTANT: NEGATIVE SIGN!
        policy_loss = torch.mean(-min_of_Q_value_batch)
        policy_loss_copy = policy_loss.detach().clone()  # record to track learning

        policy_loss.backward()
        if self.should_clip_grads:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.clip_value)
        self.policy_optimizer.step()

        # 4) undo 1)
        self.policy.eval()
        self.Q1.requires_grad_(True)
        self.Q2.requires_grad_(True)

        # III. TARGET UPDATE

        self.update_target(self.Q1_target, self.Q1)
        self.update_target(self.Q2_target, self.Q2)

        return policy_loss_copy, Q1_loss_copy, Q2_loss_copy

    def save_checkpoint(self):
        """
        Save everything needed to either resume training or restore the best model.
        - model.state_dict(): all trainable weights + buffers (e.g., BatchNorm running stats)
        - optimizer.state_dict(): optimizer internal state (e.g., momentum, Adam moments)
        """
        torch.save({
            "policy": self.policy.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "Q1": self.Q1.state_dict(),
            "Q1_optimizer": self.Q1_optimizer.state_dict(),
            "Q2": self.Q2.state_dict(),
            "Q2_optimizer": self.Q2_optimizer.state_dict(),
            "Q1_target": self.Q1_target.state_dict(),
            "Q2_target": self.Q2_target.state_dict(),
        }, self.checkpoint_file_name)

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_file_name)

        # model weights
        self.policy.load_state_dict(checkpoint["policy"])
        self.Q1.load_state_dict(checkpoint["Q1"])
        self.Q2.load_state_dict(checkpoint["Q2"])
        self.Q1_target.load_state_dict(checkpoint["Q1_target"])
        self.Q2_target.load_state_dict(checkpoint["Q2_target"])

        # optimizers
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.Q1_optimizer.load_state_dict(checkpoint["Q1_optimizer"])
        self.Q2_optimizer.load_state_dict(checkpoint["Q2_optimizer"])
