"""
    Version: April 7, 2026

    Changes from sac_agent:
    - no automatic alpha for entropy; just a beta-schedule
    - TWO buffers: one for "normal" regime, and one for "near-ruin" regime
    - policy sampled variable number of times, instead of once
    - introduced option to clip gradient norms on everything

"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Normal, Independent, TransformedDistribution
from torch.distributions.transforms import StickBreakingTransform, SigmoidTransform

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# set seed for debugging
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

##################################
# I. PARTIAL LAYER NORMALIZATION #
##################################

"""
    Custom partial layer normalization on the last 1 feature of an input of state_dimension (units of $)
"""


class PartialLayerNorm(nn.Module):
    def __init__(self, state_dimension, num_normalized_dimensions=1, normalizing_constant=1e6):
        # normalize just 'wealth' (hence num_normalized_dimensions=1)
        super().__init__()
        self.state_dimension = state_dimension
        self.normalizing_constant = normalizing_constant
        self.d = state_dimension - num_normalized_dimensions

    def forward(self, state):
        normalized_end = state[..., self.d:] / self.normalizing_constant
        normalized_state = torch.concat([state[..., :self.d], normalized_end], dim=-1)

        return normalized_state


#######################
# II. POLICY NETWORKS #
#######################

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
                    distribution on the simplex in R^{dim_action}... BY DESIGN INITIALLY A BIT LOPSIDED ON THE SIMPLEX

    (2) 'dirichlet': produces a Dirichlet distribution directly on the simplex in R^{dim_action},
                     with alpha(s) = mean(s) * concentration(s) + alpha_min
"""


class ActorNetworkGaussian(nn.Module):
    def __init__(self, state_dimension, action_dimension, hidden1_dimension,
                 hidden2_dimension, activation=nn.ReLU()):
        super().__init__()
        self.min_std = 2.0  # pick to ensure bigger exploration
        self.activation = activation

        self.mlp_trunk = nn.Sequential(
            PartialLayerNorm(state_dimension=state_dimension),
            nn.Linear(in_features=state_dimension, out_features=hidden1_dimension),
            self.activation,
            nn.Linear(in_features=hidden1_dimension, out_features=hidden2_dimension),
            self.activation
        )

        self.mean_head = nn.Linear(in_features=hidden2_dimension, out_features=action_dimension - 1)
        self.logstd_head = nn.Linear(in_features=hidden2_dimension, out_features=1)

        self.transform = StickBreakingTransform()

    def forward(self, state):
        h = self.mlp_trunk(state)

        # extract mean and std
        mean = self.mean_head(h)  # (B, A-1)
        logstd = self.logstd_head(h)  # (B, 1)
        std = F.softplus(logstd) + self.min_std
        std = std.expand_as(mean)  # (B, A-1), for Normal(mean, std)

        # produce the distribution
        base = Independent(Normal(mean, std), 1)  # event_dim=1 means log_prob has shape (B,)
        pi_given_state = TransformedDistribution(base, [self.transform])

        return pi_given_state  # KEY: supports .rsample() and .log_prob()


class ActorNetworkDirichlet(nn.Module):
    def __init__(self, state_dimension, action_dimension, hidden1_dimension,
                 hidden2_dimension, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.min_alpha = 1.  # pick >= 1 so that distributions are "centralized bumps", not concentrated at boundary
        self.activation = activation

        self.mlp_trunk = nn.Sequential(
            PartialLayerNorm(state_dimension=state_dimension),
            nn.Linear(in_features=state_dimension, out_features=hidden1_dimension),
            self.activation,
            nn.Linear(in_features=hidden1_dimension, out_features=hidden2_dimension),
            self.activation
        )

        self.mean_head = nn.Linear(in_features=hidden2_dimension, out_features=action_dimension)
        self.raw_concentration_head = nn.Linear(in_features=hidden2_dimension, out_features=1)

    def forward(self, state):
        h = self.mlp_trunk(state)

        # extract mean and concentration
        mean = F.softmax(self.mean_head(h), dim=-1)  # (B, A-1)
        concentration_raw = self.raw_concentration_head(h)  # (B, 1)
        concentration = F.softplus(concentration_raw)

        # define alpha
        alpha_of_state = (mean * concentration) + self.min_alpha  # alpha_of_state must be > 0

        pi_given_state = Dirichlet(alpha_of_state)

        return pi_given_state  # KEY: supports .rsample() and .log_prob()


#######################
# III. CRITIC NETWORK #
#######################

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


#####################
# IV. REPLAY BUFFER #
#####################


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

    def print_stats(self):
        reward_min = np.min(self.new_reward_memory)
        reward_max = np.max(self.new_reward_memory)
        reward_mean = np.mean(self.new_reward_memory)
        reward_std = np.std(self.new_reward_memory)

        # get the new wealth
        wealth_min = np.min(self.new_state_memory[:, -1])
        wealth_max = np.max(self.new_state_memory[:, -1])
        wealth_mean = np.mean(self.new_state_memory[:, -1])
        wealth_std = np.std(self.new_state_memory[:, -1])

        print(f'Memories: {self.memory_counter / 1000:.1f} thousand')
        print(f'Reward is in [{reward_min:.2f}, {reward_max:.2f}], with avg {reward_mean:.2f} and std {reward_std:.2f}')
        print(
            f'Wealth is in [{wealth_min:.2f}, {wealth_max:.2f}], with avg {wealth_mean:.2f} and std {wealth_std:.2f}')

########################
# V. SOFT ACTOR-CRITIC #
########################

BATCH_MULTIPLIER = 1  # how much bigger "normal" batches are compared to "near-ruin" batches
# make sure (BATCH_MULTIPLIER + 1) divides training loop's batch_size


class SoftActorCritic:

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
                 entropy_min,
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
        self.entropy_min = entropy_min
        self.num_policy_samples = num_policy_samples

        self.tau = tau  # rate of approach of Q-target networks to true Qs (usually 0.005)

        # initialize TWO replay buffers
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size
        self.normal_batch_size = self.batch_size * BATCH_MULTIPLIER // (BATCH_MULTIPLIER + 1)
        self.ruin_batch_size = self.batch_size // (BATCH_MULTIPLIER + 1)

        self.normal_memory = ReplayBuffer(max_memory_size=self.max_memory_size,
                                          state_dimension=self.state_dimension,
                                          action_dimension=self.action_dimension,
                                          batch_size=self.normal_batch_size,
                                          replace=replace)

        self.ruin_memory = ReplayBuffer(max_memory_size=self.max_memory_size,
                                        state_dimension=self.state_dimension,
                                        action_dimension=self.action_dimension,
                                        batch_size=self.ruin_batch_size,
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

    # take tau-step toward parent model for target networks
    def update_target(self, target_model, model):
        with torch.no_grad():  # ensure the non-target model parameters aren't recruited to begin computational graphs
            for target_parameter, parameter in zip(target_model.parameters(), model.parameters()):
                target_parameter.mul_(1 - self.tau).add_(parameter, alpha=self.tau)

    @staticmethod  # haven't used this yet
    def noisy(state, std=0.1):
        noise = torch.normal(mean=0, std=std, size=state.size())
        return state + noise

    @staticmethod
    def clamp_action(action_batch):
        eps = 1e-3
        action_batch_clamped = action_batch.clamp(eps, 1 - eps)
        action_batch_safe = action_batch_clamped / action_batch_clamped.sum(dim=-1, keepdim=True)
        return action_batch_safe

    @staticmethod
    def concatenate_batches_to_tensor(np_batch_1, np_batch_2):
        torch_batch_1 = torch.from_numpy(np_batch_1)
        torch_batch_2 = torch.from_numpy(np_batch_2)

        return torch.concat((torch_batch_1, torch_batch_2), dim=0)

    @torch.no_grad()
    def print_policy_stats(self, state_batch):
        action_dist_batch = self.policy.forward(state_batch)
        action_batch = action_dist_batch.sample()
        log_probs = action_dist_batch.log_prob(action_batch)

        mean_action = action_batch.mean(dim=0).numpy()
        log_probs_min = log_probs.min().item()
        log_probs_max = log_probs.max().item()
        log_probs_avg = log_probs.mean().item()

        print(f'Policy update: mean action is {mean_action}, '
              f'and log prob is in [{log_probs_min:.3f},{log_probs_max:.3f}], avg {log_probs_avg:.3f}')

    @torch.no_grad()
    def print_critic_values(self, state_batch):
        actions = [[0.9, 0.1],
                   [0.7, 0.3],
                   [0.5, 0.5],
                   [0.3, 0.7],
                   [0.1, 0.9]]

        avg_Q_values = [0, 0, 0, 0, 0]
        std_Q_values = [0, 0, 0, 0, 0]
        batch_size = state_batch.size(0)

        for i, action in enumerate(actions):
            action_tensor = torch.tensor(action)
            action_batch = action_tensor.unsqueeze(0).expand(batch_size, -1)
            Q_batch = self.Q1.forward(state_batch, action_batch).squeeze(-1)

            std_mean = torch.std_mean(Q_batch.squeeze(-1))
            std_Q_values[i] = std_mean[0]
            avg_Q_values[i] = std_mean[1]

        print(f"Q update: values in {[f'{avg_Q_values[i]:.8f}' for i in range(5)]}")
        print(f"........with stdevs {[f'{std_Q_values[i]:.8f}' for i in range(5)]}")

    @torch.no_grad()
    def print_update(self, state_batch_n, state_batch_r):
        state_batch_n = torch.from_numpy(state_batch_n)
        state_batch_r = torch.from_numpy(state_batch_r)

        print('NORMAL regime:')
        self.normal_memory.print_stats()
        self.print_policy_stats(state_batch_n)
        self.print_critic_values(state_batch_n)
        print('')
        print('RUIN regime:')
        self.ruin_memory.print_stats()
        self.print_policy_stats(state_batch_r)
        self.print_critic_values(state_batch_r)
        print('')

    @torch.no_grad()
    def visualize_nr_distributions(self, num_samples, traj_id):
        state_batch_n, _, _, _, _ = self.normal_memory.get_batch()
        state_batch_r, _, _, _, _ = self.ruin_memory.get_batch()

        state_batch_n = torch.from_numpy(state_batch_n)
        state_batch_r = torch.from_numpy(state_batch_r)

        num_states = 10     # number of states to sample, for each regime

        states_n = state_batch_n[:num_states, ...]  # take the first ten to plot
        states_r = state_batch_r[:num_states, ...]
        dists_n = self.policy.forward(states_n)
        dists_r = self.policy.forward(states_r)

        title_string = f'Action distributions for 10 random states, after {traj_id} trajectories'
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(13, 7))
        axes = axes.ravel()
        fig.suptitle(title_string, fontsize=16)
        bins = 50

        action_1_samples_n = dists_n.sample(sample_shape=(num_samples,))[..., 0].flatten()        # (num_samples * 10)
        action_1_samples_r = dists_r.sample(sample_shape=(num_samples,))[..., 0].flatten()        # (num_samples * 10)

        axes[0].set_title(f'NORMAL action #1 distribution over {num_states} random states, after {traj_id} trajectories')
        axes[0].hist(action_1_samples_n, range=(0, 1), bins=bins, color='blue', alpha=0.3)

        axes[1].set_title(f'NEAR RUIN action #1 distribution over {num_states} random states, after {traj_id} trajectories')
        axes[1].hist(action_1_samples_r, range=(0, 1), bins=bins, color='red', alpha=0.3)

        plt.show()

    def update(self, beta):
        # if not enough in buffer, skip
        if self.normal_memory.memory_counter < self.normal_memory.batch_size:
            return
        if self.ruin_memory.memory_counter < self.ruin_memory.batch_size:
            return

        # extract np batches from buffers. Recall the order is SAS'R'D', not SARS'D'
        state_batch_n, action_batch_n, new_state_batch_n, new_reward_batch_n, done_batch_n = self.normal_memory.get_batch()
        state_batch_r, action_batch_r, new_state_batch_r, new_reward_batch_r, done_batch_r = self.ruin_memory.get_batch()

        # convert to tensors and concatenate
        state_batch = self.concatenate_batches_to_tensor(state_batch_n, state_batch_r)
        action_batch = self.concatenate_batches_to_tensor(action_batch_n, action_batch_r)
        new_state_batch = self.concatenate_batches_to_tensor(new_state_batch_n, new_state_batch_r)
        new_reward_batch = self.concatenate_batches_to_tensor(new_reward_batch_n, new_reward_batch_r)
        done_batch = self.concatenate_batches_to_tensor(done_batch_n, done_batch_r)

        # DIAGNOSTICS
        self.print_update(state_batch_n, state_batch_r)

        # I. Q UPDATE

        # 1) produce target future rewards values from batch of new states
        with torch.no_grad():
            new_action_distribution_batch = self.policy.forward(new_state_batch)

            # prepare for construction of target values
            min_of_target_values_batch = torch.zeros_like(new_reward_batch)
            nll_batch = torch.zeros_like(new_reward_batch)

            for _ in range(self.num_policy_samples):
                # take batch of samples
                new_action_batch = self.clamp_action(new_action_distribution_batch.sample())

                # compute Q values
                Q1_target_values_batch = self.Q1_target.forward(new_state_batch, new_action_batch).squeeze(-1)
                Q2_target_values_batch = self.Q2_target.forward(new_state_batch, new_action_batch).squeeze(-1)

                # update minimum; (B,)
                min_of_target_values_batch = min_of_target_values_batch \
                                             + torch.min(Q1_target_values_batch, Q2_target_values_batch)

                # compute nll
                nll_batch = nll_batch + new_action_distribution_batch.log_prob(new_action_batch)

            # normalize by number of policy samples
            min_of_target_values_batch = min_of_target_values_batch / self.num_policy_samples
            nll_batch = nll_batch / self.num_policy_samples

            # calculate target values
            target_future_rewards_batch = new_reward_batch \
                                          + self.gamma * (1 - done_batch) \
                                          * (min_of_target_values_batch + beta * nll_batch)

        # 2) use target values to update the critics Q1 and Q2
        # Q1:
        self.Q1.train()
        self.Q1_optimizer.zero_grad(set_to_none=True)
        Q1_value_batch = self.Q1.forward(state_batch, action_batch).squeeze(-1)
        Q1_loss = torch.mean((target_future_rewards_batch - Q1_value_batch) ** 2)
        # alternative loss: Q1_loss = F.smooth_l1_loss(target_future_rewards_batch, Q1_value_batch)
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
        Q2_loss = torch.mean((target_future_rewards_batch - Q2_value_batch) ** 2)
        # alternative loss: Q2_loss = F.smooth_l1_loss(target_future_rewards_batch, Q2_value_batch)
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

        min_of_Q_values_batch = torch.zeros_like(Q1_value_batch)
        nll_batch = torch.zeros_like(Q1_value_batch)

        # use updated Q-values to calculate minima by taking rsamples of current policy
        for _ in range(self.num_policy_samples):
            action_batch = self.clamp_action(action_distribution_batch.rsample())  # overwrite `action batch`
            Q1_value_batch = self.Q1.forward(state_batch, action_batch).squeeze(-1)
            Q2_value_batch = self.Q2.forward(state_batch, action_batch).squeeze(-1)
            min_of_Q_values_batch = min_of_Q_values_batch + torch.min(Q1_value_batch, Q2_value_batch)
            nll_batch = nll_batch - action_distribution_batch.log_prob(action_batch)

        expected_min_of_Q_values = torch.mean(min_of_Q_values_batch) / self.num_policy_samples
        expected_entropy = torch.mean(nll_batch) / self.num_policy_samples

        # 3) calculate policy loss, and take a step. IMPORTANT: NEGATIVE SIGNS!
        policy_value = expected_min_of_Q_values + beta * expected_entropy
        policy_loss = -policy_value

        # take detached views to track learning
        policy_value_copy = policy_value.detach().clone()
        expected_min_of_Q_values_copy = expected_min_of_Q_values.detach().clone()
        expected_entropy_copy = expected_entropy.detach().clone()

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

        return policy_value_copy, expected_min_of_Q_values_copy, expected_entropy_copy, Q1_loss_copy, Q2_loss_copy

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
