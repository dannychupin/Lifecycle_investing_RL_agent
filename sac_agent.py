import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

import numpy as np
from copy import deepcopy

# set seed for debugging
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


class ActorNetwork(nn.Module):
    """
        Actor: policy function p: R^{dim_state} -> Dist(R^{dim_action})

        State space is continuous, dim_state = dim_action + dim_ancilla + 2
        Action space is continuous, dim_action
        Reward is continuous, scalar

        Design choices:
        (1) An action is a set of weights, so takes values in the SIMPLEX in [0,1]^{dim_action}
        (2) alpha(s) := concentration(s) * mean(s) + alpha_min, for Dirichlet distribution
    """

    def __init__(self, state_dimension, action_dimension, hidden1_dimension, hidden2_dimension):

        super().__init__()
        self.alpha_min = 1e-3       # for numerical stability

        self.mlp_base = nn.Sequential(
            nn.Linear(in_features=state_dimension, out_features=hidden1_dimension),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden1_dimension, out_features=hidden2_dimension),
            nn.ReLU(inplace=True)
        )
        self.mean_head = nn.Linear(in_features=hidden2_dimension, out_features=action_dimension)
        self.concentration_head = nn.Linear(in_features=hidden2_dimension, out_features=1)

    def forward(self, state):
        h = self.mlp_base(state)

        # find `mean` direction in the action simplex (hence the softmax)
        mean_action = F.softmax(self.mean_head(h), dim=-1)

        # find `concentration` parameter
        concentration_action = F.sigmoid(self.concentration_head(h))
        # print(f'concentration is {concentration_action}')

        # define alpha, and the resulting Dirichlet distribution on the simplex in R^{action_dimension}
        alpha_of_state = mean_action * concentration_action + self.alpha_min
        pi_given_state = Dirichlet(alpha_of_state)

        return pi_given_state


class CriticNetwork(nn.Module):
    """
        Critic: approximates Q: R^{dim_state} x R^{dim_action} -> R^1
    """

    def __init__(self, state_dimension, action_dimension, hidden1_dimension, hidden2_dimension):

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=state_dimension + action_dimension, out_features=hidden1_dimension),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden1_dimension, out_features=hidden2_dimension),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden2_dimension, out_features=1)
        )

    def forward(self, state, action):
        state_and_action = torch.cat((state, action), dim=-1)
        q_value = self.mlp(state_and_action)

        return torch.tanh(q_value)    # unactivated


class ReplayBuffer:
    # lives in Numpy. Need to convert to Torch tensors at sample time

    def __init__(self, max_memory_size, state_dimension, action_dimension, batch_size):
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size

        self.memory_counter = 0

        # order for training will be SAS'R', not SARS': reward only comes after the NEW state
        self.state_memory = np.zeros((self.max_memory_size, state_dimension), dtype=np.float32)
        self.action_memory = np.zeros((self.max_memory_size, action_dimension), dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_memory_size, state_dimension), dtype=np.float32)
        self.reward_memory = np.zeros(self.max_memory_size, dtype=np.float32)

        # done is an int, either 0 (no) or 1 (yes)
        self.done_memory = np.zeros(self.max_memory_size, dtype=np.float32)

    def store_transition(self, state, action, new_state, reward, is_done: bool):
        # state, action, new_state are np.arrays. reward, done are single values

        # get position of first unoccupied memory slot
        index = self.memory_counter % self.max_memory_size

        # add to the memory
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward

        # is this transition the end of an episode? is_done = 1 if yes, = 0 if no
        self.done_memory[index] = int(is_done)

        # update memory counter!
        self.memory_counter += 1

    def get_batch(self):
        memory_lower_bound = min(self.max_memory_size, self.memory_counter)
        batch = np.random.choice(memory_lower_bound, self.batch_size, replace=False)

        # extract memory submatrix with just the rows in the list `batch`:
        state_batch = self.state_memory[batch]
        action_batch = self.action_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        reward_batch = self.reward_memory[batch]
        done_batch = self.done_memory[batch]

        # outputs are np arrays, not torch tensors
        # recall order is SAS'R', not SARS'
        return state_batch, action_batch, new_state_batch, reward_batch, done_batch


class SoftActorCritic:

    def __init__(self, state_dimension,
                 action_dimension,
                 hidden1_dimension,
                 hidden2_dimension,
                 tau,
                 lr_actor,
                 lr_critic,
                 lr_alpha,
                 H_target,
                 max_memory_size,
                 batch_size,
                 alpha_initial,
                 gamma,
                 checkpoint_file_name="SAC_model_weights/checkpoint.pt"):

        self.checkpoint_file_name = checkpoint_file_name
        self.action_dimension = action_dimension
        self.state_dimension = state_dimension
        self.gamma = gamma

        self.tau = tau      # `lr` for the two Q target networks = rate of approach to true Qs

        # initialize replay buffer
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(max_memory_size=self.max_memory_size,
                                   state_dimension=self.state_dimension,
                                   action_dimension=self.action_dimension,
                                   batch_size=self.batch_size)

        # actor. Outputs a distribution on action space
        self.policy = ActorNetwork(state_dimension=state_dimension,
                                   action_dimension=action_dimension,
                                   hidden1_dimension=hidden1_dimension,
                                   hidden2_dimension=hidden2_dimension)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)

        # Q1
        self.Q1 = CriticNetwork(state_dimension=state_dimension,
                                action_dimension=action_dimension,
                                hidden1_dimension=hidden1_dimension,
                                hidden2_dimension=hidden2_dimension)
        self.Q1_optimizer = torch.optim.Adam(self.Q1.parameters(), lr=lr_critic)

        # Q1 target: turn off grad
        self.Q1_target = deepcopy(self.Q1)
        self.Q1_target.requires_grad_(False)
        self.Q1_target.eval()

        # Q2
        self.Q2 = CriticNetwork(state_dimension=state_dimension,
                                action_dimension=action_dimension,
                                hidden1_dimension=hidden1_dimension,
                                hidden2_dimension=hidden2_dimension)
        self.Q2_optimizer = torch.optim.Adam(self.Q2.parameters(), lr=lr_critic)

        # Q2 target: turn off grad
        self.Q2_target = deepcopy(self.Q2)
        self.Q2_target.requires_grad_(False)
        self.Q2_target.eval()

        # temperature parameter
        self.alpha_initial = alpha_initial
        self.alpha = nn.Parameter(torch.tensor(self.alpha_initial))
        self.alpha_optimizer = torch.optim.Adam([self.alpha], lr=lr_alpha)
        self.H_target = H_target    # target/minimum entropy for policy

    @staticmethod
    def noisy(state, std=0.1):
        noise = torch.normal(mean=0, std=std, size=state.size())
        return state + noise

    # helper function for taking tau-step toward parent model for target networks
    def update_target(self, target_model, model):
        # ensure the non-target model parameters aren't recruited to begin computational graphs
        with torch.no_grad():
            for target_parameter, parameter in zip(target_model.parameters(), model.parameters()):
                target_parameter.mul_(1 - self.tau).add_(parameter, alpha=self.tau)
                # should be the case that target_parameter.requires_grad == False

    def update(self):
        # if not enough in buffer, skip
        if self.memory.memory_counter < self.memory.batch_size:
            return

        # extract np batch from buffer
        # recall the order is SAS'R'D', not SARS'D': reward is calculated from the new state
        state_batch, action_batch, new_state_batch, reward_batch, done_batch = self.memory.get_batch()

        state_batch = torch.from_numpy(state_batch)
        action_batch = torch.from_numpy(action_batch)
        new_state_batch = torch.from_numpy(new_state_batch)
        reward_batch = torch.from_numpy(reward_batch)
        done_batch = torch.from_numpy(done_batch)

        # I. Q UPDATE

        # 1) get new action distribution batch
        self.policy.eval()
        new_action_distribution_batch = self.policy.forward(new_state_batch)

        # IMPORTANT: for upcoming policy update, make sure it is rsample instead of sample
        new_action_batch = new_action_distribution_batch.rsample()
        new_action_log_prob_batch = new_action_distribution_batch.log_prob(new_action_batch)

        # 2) compute target values
        with torch.no_grad():
            Q1_target_values_batch = self.Q1_target.forward(new_state_batch, new_action_batch).squeeze()
            Q2_target_values_batch = self.Q2_target.forward(new_state_batch, new_action_batch).squeeze()
            min_of_target_values_batch = torch.min(Q1_target_values_batch, Q2_target_values_batch)

            # extract inflation and wealth, and compute discount factor
            inflation_batch = new_state_batch[:, -3]    # (B,)
            wealth_batch = new_state_batch[:, -2]       # (B,)
            discount_factor_batch = self.gamma * torch.reciprocal(1 + inflation_batch)

            # calculate target future value
            continuing_piece = reward_batch + min_of_target_values_batch - self.alpha * new_action_log_prob_batch
            terminal_piece = wealth_batch      # remove relu to penalize amount of debt

            target_future_rewards_batch = discount_factor_batch * ((1 - done_batch) * continuing_piece +
                                                                   done_batch * terminal_piece)

        # 2) use target values to update the critics Q1 and Q2
        # Q1:
        self.Q1.train()
        self.Q1_optimizer.zero_grad()

        Q1_value_batch = self.Q1.forward(state_batch, action_batch).squeeze()
        Q1_loss = torch.mean((target_future_rewards_batch - Q1_value_batch) ** 2)

        Q1_loss.backward()
        # nn.utils.clip_grad_norm_(self.Q1.parameters(), max_norm=10.0, norm_type=2)
        Q1_grad_norm = nn.utils.clip_grad_norm_(self.Q1.parameters(), max_norm=float('inf'), norm_type=2)
        self.Q1_optimizer.step()
        self.Q1.eval()

        # Q2:
        self.Q2.train()
        self.Q2_optimizer.zero_grad()

        Q2_value_batch = self.Q2.forward(state_batch, action_batch).squeeze()
        Q2_loss = torch.mean((target_future_rewards_batch - Q2_value_batch) ** 2)

        Q2_loss.backward()
        # nn.utils.clip_grad_norm_(self.Q2.parameters(), max_norm=10.0, norm_type=2)
        Q2_grad_norm = nn.utils.clip_grad_norm_(self.Q2.parameters(), max_norm=float('inf'), norm_type=2)
        self.Q2_optimizer.step()
        self.Q2.eval()

        # II. POLICY UPDATE

        # 1) zero out the grad on policy
        self.policy.train()
        self.policy_optimizer.zero_grad()

        # for speed: freeze parameters on Qs, so that grad only flows through their action arguments
        self.Q1.requires_grad_(False)
        self.Q2.requires_grad_(False)

        # 2) compute Q minimum. Use previously rsample()'d action batch and log probs
        Q1_value_batch = self.Q1.forward(state_batch, action_batch).squeeze()
        Q2_value_batch = self.Q2.forward(state_batch, action_batch).squeeze()
        min_of_Q_value_batch = torch.min(Q1_value_batch, Q2_value_batch)

        # 3) calculate loss, and take a step
        policy_loss = torch.mean(self.alpha * new_action_log_prob_batch - min_of_Q_value_batch)
        policy_loss.backward()
        policy_grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=float('inf'), norm_type=2)
        self.policy_optimizer.step()

        # 4) undo 1)
        self.policy.eval()
        self.Q1.requires_grad_(True)
        self.Q2.requires_grad_(True)

        # III. TEMPERATURE UPDATE

        # 1) calculate the gradient
        alpha_gradient = -(torch.mean(new_action_log_prob_batch) + self.H_target)

        # 2) manually set the gradient, but use optimizer to propagate it with the learning rate
        with torch.no_grad():   # I'm told to always wrap with no_grad when setting gradients
            self.alpha.grad = alpha_gradient

        self.alpha_optimizer.step()

        # IV. TARGET UPDATE

        self.update_target(self.Q1_target, self.Q1)
        self.update_target(self.Q2_target, self.Q2)

        return policy_grad_norm, Q1_grad_norm, Q2_grad_norm

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
            "alpha": self.alpha.detach(),
            "alpha_optimizer": self.alpha_optimizer.state_dict()
        }, self.checkpoint_file_name)

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_file_name)

        # model weights
        self.policy.load_state_dict(checkpoint["policy"])
        self.Q1.load_state_dict(checkpoint["Q1"])
        self.Q2.load_state_dict(checkpoint["Q2"])
        self.Q1_target.load_state_dict(checkpoint["Q1_target"])
        self.Q2_target.load_state_dict(checkpoint["Q2_target"])

        with torch.no_grad():
            self.alpha.copy_(checkpoint["alpha"])

        # optimizers
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.Q1_optimizer.load_state_dict(checkpoint["Q1_optimizer"])
        self.Q2_optimizer.load_state_dict(checkpoint["Q2_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
