"""
PPO Training for Retirement Portfolio Allocation
=================================================

Drop-in replacement for ac_train.py.  Uses the same block_bootstrap.SimulatedInvestor
environment interface.

Key changes:
  - PPO instead of off-policy Actor-Critic
  - Wealth normalized by STARTING_WEALTH so the network sees ~O(1) inputs
  - Collects a batch of trajectories, then does multiple PPO epochs on the batch
  - Cleaner metric tracking and plotting :)
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from block_bootstrap import SimulatedInvestor
from ppo_agent import PPOAgent

# reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ===========================================================================
# Configuration
# ===========================================================================

# --- assets & countries (same as original) ---

ALLOWED_COUNTRIES = [
    'Australia', 'Belgium', 'Canada', 'Switzerland', 'Germany',
    'Denmark', 'Spain', 'Finland', 'France', 'UK',
    'Ireland', 'Italy', 'Japan', 'Netherlands', 'Norway',
    'Portugal', 'Sweden', 'USA',
]

ALLOWED_ASSETS = [
    ['eq_tr', 'eq_capgain', 'eq_div_rtn'],     # stocks
    ['bond_tr', 'bond_rate'],                    # bonds
]

ALLOWED_MACROS = [['cpi']]   # inflation must be last

ACTION_DIM = len(ALLOWED_ASSETS)                       # 2 (stocks, bonds)
OBS_DIM    = len(ALLOWED_ASSETS) + len(ALLOWED_MACROS) - 1   # real returns (no inflation col)
STATE_DIM  = OBS_DIM + 1                               # + normalized wealth

# --- financial parameters ---

STARTING_WEALTH = 1_000_000.0
BETA            = 0.04
WITHDRAW        = BETA * STARTING_WEALTH               # 40,000 real $/year

# --- reward ---

REWARD_TYPE = 'simple'    # 'simple' (survive/ruin) or 'wealth' (delta W)
PENALTY     = 100         # penalty per remaining year on ruin

# --- PPO hyperparameters ---

HIDDEN_DIM      = 32
DIST_TYPE       = 'gaussian'     # 'gaussian' or 'dirichlet'
LR              = 3e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
PPO_EPOCHS      = 4              # gradient epochs per batch
MINIBATCH_SIZE  = 128
VALUE_COEF      = 0.5
MAX_GRAD_NORM   = 0.5

# --- training schedule ---

TRAJS_PER_BATCH = 40             # episodes collected before each PPO update
NUM_BATCHES     = 300            # total PPO updates
TOTAL_TRAJS     = TRAJS_PER_BATCH * NUM_BATCHES   # 4000 episodes total

# --- output ---

CHECKPOINT_PATH = 'PPO_weights/checkpoint.pt'
GRAPH_DIR       = 'PPO_graphs'


# ===========================================================================
# Helpers
# ===========================================================================

def nominal_to_real(observation):
    """Convert nominal observation (with inflation as last entry) to real returns."""
    inflation = observation[-1]
    return observation[:-1] - inflation


def make_state(real_obs, wealth):
    """Bundle real returns + normalized wealth into a state vector."""
    norm_wealth = wealth / STARTING_WEALTH    # <-- the key fix: O(1) scale
    return np.concatenate([real_obs, [norm_wealth]], dtype=np.float32)


def financial_update(real_obs, action, wealth_old):
    """Apply portfolio return and withdrawal.  Returns new wealth (not clipped)."""
    asset_returns = real_obs[:ACTION_DIM]
    portfolio_return = np.dot(asset_returns, action)
    return wealth_old * (1.0 + portfolio_return) - WITHDRAW


def compute_reward(wealth_new, wealth_old, time_remaining, reward_type, penalty):
    if reward_type == 'simple':
        if wealth_new > 0:
            return 1.0
        else:
            return -penalty * time_remaining

    elif reward_type == 'wealth':
        if wealth_new >= 0:
            return (wealth_new - wealth_old) / STARTING_WEALTH
        else:
            return 0.0


# ===========================================================================
# Trainer
# ===========================================================================

class Trainer:

    def __init__(self, reward_type=REWARD_TYPE, penalty=PENALTY):
        self.reward_type = reward_type
        self.penalty = penalty

        self.investor = SimulatedInvestor(
            countries=ALLOWED_COUNTRIES,
            asset_proxy_list=ALLOWED_ASSETS,
            macro_list=ALLOWED_MACROS,
            should_maximize_entropy=False,
        )

        self.agent = PPOAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            dist_type=DIST_TYPE,
            lr=LR,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_eps=CLIP_EPS,
            ppo_epochs=PPO_EPOCHS,
            minibatch_size=MINIBATCH_SIZE,
            value_coef=VALUE_COEF,
            max_grad_norm=MAX_GRAD_NORM,
        )

        # per-episode metrics  (one entry per episode)
        self.history = {
            'terminal_wealth': [],
            'ruin':            [],      # 1 if went broke, else 0
            'avg_stock_alloc': [],
            'avg_bond_alloc':  [],
            'episode_length':  [],
        }
        # per-batch metrics  (one entry per PPO update)
        self.train_stats = {
            'policy_loss': [],
            'value_loss':  [],
            'clip_frac':   [],
        }

    # ----- run one episode, storing transitions in the agent's buffer -----

    def _run_episode(self):
        """Roll out one episode. Returns (terminal_wealth, ruin, avg_actions)."""
        traj_len = self.investor.generate_time_after_retirement()
        if traj_len <= 1:
            return None   # skip degenerate episodes

        observations = self.investor.get_trajectory(traj_len)
        wealth = STARTING_WEALTH
        actions_taken = []

        for t in range(traj_len - 1):
            # state
            real_obs = nominal_to_real(observations[t])
            state = make_state(real_obs, wealth)

            # act
            action, log_prob, value = self.agent.act(state)
            actions_taken.append(action)

            # environment step
            real_obs_next = nominal_to_real(observations[t + 1])
            wealth_new = financial_update(real_obs_next, action, wealth)
            wealth_new = np.clip(wealth_new, 0.0, 10 * STARTING_WEALTH)

            # reward
            reward = compute_reward(
                wealth_new, wealth, traj_len - t,
                self.reward_type, self.penalty,
            )

            # done?
            is_done = (wealth_new == 0.0) or (t + 1 == traj_len - 1)

            # next state (only needed to store in buffer for the value baseline)
            state_next = make_state(real_obs_next, wealth_new)

            # store
            self.agent.buffer.store(state, action, log_prob, reward,
                                    float(is_done), value)
            wealth = wealth_new

            if wealth == 0.0:
                break

        actions_arr = np.array(actions_taken)
        ruin = 1 if wealth == 0.0 else 0
        return wealth, ruin, actions_arr.mean(axis=0)

    # ----- main training loop -----

    def train(self):
        start = time.time()
        print(f'PPO training: {NUM_BATCHES} batches x {TRAJS_PER_BATCH} episodes '
              f'= {TOTAL_TRAJS} episodes')
        print(f'Reward: {self.reward_type}  |  Penalty: {self.penalty}  |  '
              f'Dist: {DIST_TYPE}  |  LR: {LR}')
        print('-' * 60)

        for batch in range(NUM_BATCHES):
            # ---- collect ----
            batch_wealth = []
            batch_ruin = []
            batch_stock = []
            batch_bond = []

            for _ in range(TRAJS_PER_BATCH):
                result = self._run_episode()
                if result is None:
                    continue
                w, r, avg_a = result
                batch_wealth.append(w)
                batch_ruin.append(r)
                batch_stock.append(avg_a[0])
                batch_bond.append(avg_a[1])

            # ---- PPO update ----
            if len(self.agent.buffer) > 0:
                stats = self.agent.update()
                self.train_stats['policy_loss'].append(stats['policy_loss'])
                self.train_stats['value_loss'].append(stats['value_loss'])
                self.train_stats['clip_frac'].append(stats['clip_frac'])

            # ---- record ----
            self.history['terminal_wealth'].extend(batch_wealth)
            self.history['ruin'].extend(batch_ruin)
            self.history['avg_stock_alloc'].extend(batch_stock)
            self.history['avg_bond_alloc'].extend(batch_bond)
            self.history['episode_length'].extend([0] * len(batch_wealth))  # placeholder

            # ---- log ----
            if (batch + 1) % 10 == 0 or batch == 0:
                ruin_rate = np.mean(batch_ruin) if batch_ruin else 0
                avg_w = np.mean(batch_wealth) if batch_wealth else 0
                avg_s = np.mean(batch_stock) if batch_stock else 0
                elapsed = time.time() - start
                print(f'  batch {batch+1:4d}/{NUM_BATCHES}  '
                      f'ruin={ruin_rate:.2f}  '
                      f'wealth={avg_w:,.0f}  '
                      f'stock={avg_s:.2f}  '
                      f'ploss={stats["policy_loss"]:.4f}  '
                      f'vloss={stats["value_loss"]:.4f}  '
                      f'clip={stats["clip_frac"]:.2f}  '
                      f'[{elapsed:.0f}s]')

        elapsed = time.time() - start
        print('-' * 60)
        print(f'Done in {elapsed:.1f}s  ({elapsed/TOTAL_TRAJS:.3f}s/episode)')

        # save
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        self.agent.save(CHECKPOINT_PATH)
        print(f'Checkpoint saved to {CHECKPOINT_PATH}')

    # ----- plotting -----

    def plot(self, window=100, save=True, show=False):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex='col')
        fig.suptitle(
            f'PPO  |  {DIST_TYPE}  |  reward={self.reward_type}  |  '
            f'penalty={self.penalty}  |  lr={LR}',
            fontsize=13,
        )

        def rolling(data, w=window):
            return pd.Series(data).rolling(w, min_periods=1).mean()

        # --- stock allocation ---
        ax = axes[0, 0]
        s = rolling(self.history['avg_stock_alloc'])
        ax.plot(s, linewidth=0.8)
        ax.set_ylim(0, 1)
        ax.set_title('Stock allocation (rolling avg)')
        ax.grid(True, alpha=0.3)

        # --- bond allocation ---
        ax = axes[1, 0]
        b = rolling(self.history['avg_bond_alloc'])
        ax.plot(b, linewidth=0.8, color='tab:orange')
        ax.set_ylim(0, 1)
        ax.set_title('Bond allocation (rolling avg)')
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)

        # --- terminal wealth ---
        ax = axes[0, 1]
        ax.plot(rolling(self.history['terminal_wealth']), linewidth=0.8, color='tab:green')
        ax.set_title('Terminal wealth (rolling avg)')
        ax.grid(True, alpha=0.3)

        # --- ruin rate ---
        ax = axes[1, 1]
        ax.plot(rolling(self.history['ruin']), linewidth=0.8, color='tab:red')
        ax.set_ylim(0, max(0.25, max(self.history['ruin']) + 0.05))
        ax.set_title('Ruin probability (rolling avg)')
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)

        # --- policy loss ---
        ax = axes[0, 2]
        ax.plot(self.train_stats['policy_loss'], linewidth=0.8, color='tab:purple')
        ax.set_title('Policy loss (per PPO update)')
        ax.grid(True, alpha=0.3)

        # --- value loss ---
        ax = axes[1, 2]
        ax.plot(self.train_stats['value_loss'], linewidth=0.8, color='tab:brown')
        ax.set_title('Value loss (per PPO update)')
        ax.set_xlabel('PPO update')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        if save:
            os.makedirs(GRAPH_DIR, exist_ok=True)
            fname = (f'{DIST_TYPE}_reward={self.reward_type}_'
                     f'penalty={self.penalty}_lr={LR}.png')
            path = os.path.join(GRAPH_DIR, fname)
            plt.savefig(path, dpi=150)
            print(f'Plot saved to {path}')

        if show:
            plt.show()

        plt.close(fig)


# ===========================================================================
# Run
# ===========================================================================

if __name__ == '__main__':

    # single run with good defaults
    trainer = Trainer(reward_type=REWARD_TYPE, penalty=PENALTY)
    trainer.train()
    trainer.plot(window=100, save=True, show=False)

    # uncomment for a small hyperparameter sweep:
    # for penalty in [1, 5, 10]:
    #     for reward_type in ['simple', 'wealth']:
    #         trainer = Trainer(reward_type=reward_type, penalty=penalty)
    #         trainer.train()
    #         trainer.plot(save=True)
