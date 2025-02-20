# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gym
from gym import spaces
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA  # For Canonical Correlation Analysis
from collections import deque, Counter
from tqdm import tqdm
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, rankdata
import copy

# Uncomment the following line if using Weights & Biases for logging
# import wandb

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

####################################
# Detect Device
####################################

# Detect device and make it accessible globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

####################################
# Define Utility Functions
####################################

def print_log(str, logfile=None):
    """Prints a log message with timestamp."""
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

####################################
# Define Metrics
####################################

@torch.jit.script
def kl_div(p, q, ndims: int=1):
    # Compute KL divergence between p and q
    div = p * (torch.log(p) - torch.log(q))
    div[p == 0] = 0  # NaNs quick fix
    dims = [i for i in range(-1, -(ndims+1), -1)]
    div = div.sum(dims)
    return div

@torch.jit.script
def js_div(p, q, ndims: int=1):
    # Compute Jensen-Shannon divergence between p and q
    m = (p + q) / 2
    div = (kl_div(p, m, ndims) + kl_div(q, m, ndims)) / 2
    return div

####################################
# Define Statistical Tests
####################################

def run_permutation_test(all_data, n1, n2):
    """Runs a single permutation test."""
    np.random.shuffle(all_data)
    data_a = all_data[:n1]
    data_b = all_data[-n2:]
    return data_a.mean() - data_b.mean()

def run_test(test_id, data1, data2, alpha=0.05):
    """
    Compute statistical tests comparing data1 and data2 with confidence level alpha.
    Returns True if the null hypothesis is rejected.
    """
    data1 = data1.squeeze()
    data2 = data2.squeeze()
    n1 = data1.size
    n2 = data2.size

    if np.all(data1 == data2):
        return False

    if test_id == 't-test':
        _, p = ttest_ind(data1, data2, equal_var=True)
        return p < alpha

    elif test_id == "Welch t-test":
        _, p = ttest_ind(data1, data2, equal_var=False)
        return p < alpha

    elif test_id == 'Mann-Whitney':
        _, p = mannwhitneyu(data1, data2, alternative='two-sided')
        return p < alpha

    elif test_id == 'Wilcoxon':
        _, p = wilcoxon(data1, data2, correction=True, alternative='two-sided', zero_method="pratt")
        return p < alpha

    elif test_id == 'Ranked t-test':
        all_data = np.concatenate([data1.copy(), data2.copy()], axis=0)
        ranks = rankdata(all_data)
        ranks1 = ranks[: n1]
        ranks2 = ranks[n1:n1 + n2]
        assert ranks2.size == n2
        _, p = ttest_ind(ranks1, ranks2, equal_var=True)
        return p < alpha

    elif test_id == 'permutation':
        all_data = np.concatenate([data1.copy(), data2.copy()], axis=0)
        delta = np.abs(data1.mean() - data2.mean())
        num_samples = 1000
        estimates = []
        for _ in range(num_samples):
            estimates.append(run_permutation_test(all_data.copy(), n1, n2))
        estimates = np.abs(np.array(estimates))
        diff_count = len(np.where(estimates <= delta)[0])
        return (1.0 - (float(diff_count) / float(num_samples))) < alpha

    else:
        raise NotImplementedError(f"Test '{test_id}' is not implemented.")

####################################
# Define the Custom Gym Environment
####################################

class NavigationEnv(gym.Env):
    """
    Custom Navigation Environment using dSprites dataset.
    The agent's goal is to navigate to a target position while dealing with confounders.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_games=1, max_steps=100):
        super(NavigationEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        self.max_steps = max_steps
        self.current_step = 0

        # Load dSprites dataset
        self.dataset_path = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        if not os.path.exists(self.dataset_path):
            # Download dataset if it doesn't exist
            import urllib.request
            url = 'https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
            print("Downloading dSprites dataset...")
            urllib.request.urlretrieve(url, self.dataset_path)
            print("dSprites dataset downloaded.")

        dataset = np.load(self.dataset_path, allow_pickle=True, encoding='latin1')
        self.imgs = dataset['imgs'].reshape(-1, 64, 64, 1).astype(np.float32)
        metadata = dataset['metadata'][()]
        self.latent_sizes = metadata['latents_sizes']  # [1, 3, 6, 40, 32, 32]
        self.latent_bases = np.concatenate(
            (self.latent_sizes[::-1].cumprod()[::-1][1:], np.array([1]))
        )
        self.latent_names = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY']
        self.latent_dim = len(self.latent_names)
        self.num_games = num_games

        # Initialize variables
        self.current_latents = np.zeros((self.num_games, self.latent_dim))
        self.reset_all_latents()

        print(f'Dataset loaded. Time: {datetime.datetime.now()}, '
              f'Datapoints: {len(self.imgs)}, Latent Dimensions: {self.latent_dim}')

    def sample_latent(self):
        """
        Sample a random latent vector.
        """
        latent = np.zeros(self.latent_dim)
        latent[0] = 0  # Color is always 0 in dSprites dataset

        latent[1] = np.random.randint(0, self.latent_sizes[1])   # Shape
        latent[2] = np.random.randint(0, self.latent_sizes[2])   # Scale
        latent[3] = np.random.randint(0, self.latent_sizes[3])   # Orientation
        latent[4] = np.random.randint(0, self.latent_sizes[4])   # posX
        latent[5] = np.random.randint(0, self.latent_sizes[5])   # posY
        return latent

    def sample_all_latents(self):
        """
        Sample latents for all game instances.
        """
        latents = np.zeros((self.num_games, self.latent_dim))
        for idx in range(self.num_games):
            latents[idx] = self.sample_latent()
        return latents

    def latents_to_index(self, latents):
        """
        Convert latent vectors to dataset indices.
        """
        return np.dot(latents, self.latent_bases).astype(int)

    def step(self, action):
        """
        Execute one action step in the environment.
        """
        index = 0  # Single game instance
        # Implement action logic affecting the latent variables (position)
        if action == 0:  # Up
            if self.current_latents[index, 5] < self.latent_sizes[5] - 1:
                self.current_latents[index, 5] += 1
        elif action == 1:  # Down
            if self.current_latents[index, 5] > 0:
                self.current_latents[index, 5] -= 1
        elif action == 2:  # Left
            if self.current_latents[index, 4] > 0:
                self.current_latents[index, 4] -= 1
        elif action == 3:  # Right
            if self.current_latents[index, 4] < self.latent_sizes[4] - 1:
                self.current_latents[index, 4] += 1
        else:
            raise ValueError(f"Invalid action: {action}")

        # Define reward logic
        target_pos = np.array([self.latent_sizes[4]//2, self.latent_sizes[5]//2])
        new_pos = self.current_latents[index, 4:6]
        distance = np.linalg.norm(new_pos - target_pos)
        reward = -1  # Step penalty

        # Check if target is reached
        done = False
        if distance == 0:
            reward += 100  # Large reward for reaching the target
            done = True

        # Increment step counter
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        # Get observation
        obs = self.current_latents[index].copy()

        # Compute state index
        posX = int(obs[4])
        posY = int(obs[5])
        s = posX * self.latent_sizes[5] + posY

        info = {'s': s}

        return obs, reward, done, info

    def reset(self):
        """
        Reset the environment to an initial state.
        """
        self.current_latents = self.sample_all_latents()
        self.current_step = 0
        state = self.current_latents[0].copy()

        # Compute state index
        posX = int(state[4])
        posY = int(state[5])
        s = posX * self.latent_sizes[5] + posY

        info = {'s': s}

        return state, info

    def render(self, mode='human'):
        """
        Render the environment.
        """
        pass  # Implement rendering if needed

    def reset_all_latents(self):
        """
        Reset all latents in the environment.
        """
        self.current_latents = self.sample_all_latents()
        self.current_step = 0

####################################
# Confounded Environment Wrapper
####################################

class ConfounderEnvWrapper(gym.Wrapper):
    """
    Gym environment wrapper that adds hidden confounders influencing both state transitions and rewards.
    """
    def __init__(self, env, noise_std=0.05):
        super(ConfounderEnvWrapper, self).__init__(env)
        self.noise_std = noise_std
        self.last_info = {}
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        """
        Step with confounders affecting the environment.
        """
        obs, reward, done, info = self.env.step(action)
        # Add noise to state and reward
        modified_state = obs + np.random.normal(0, self.noise_std, size=obs.shape).astype(np.float32)
        modified_reward = reward + np.random.normal(0, self.noise_std)
        self.last_info = info
        return modified_state, modified_reward, done, info

    def reset(self):
        """
        Reset the environment state.
        """
        state, info = self.env.reset()
        modified_state = state + np.random.normal(0, self.noise_std, size=state.shape).astype(np.float32)
        self.last_info = info
        return modified_state, info

####################################
# Define Policies
####################################

class Policy:
    def __init__(self):
        pass

    def reset(self):
        pass

    def action(self, env):
        raise NotImplementedError

class UniformPolicy(Policy):
    """
    Interventional Policy: Selects actions uniformly at random.
    """
    def __init__(self, a_nvals):
        super().__init__()
        self.a_nvals = a_nvals

    def reset(self):
        pass

    def action(self, env):
        return random.randint(0, self.a_nvals - 1)

class ExpertPolicy(Policy):
    """
    Observational Policy: Chooses actions based on the hidden state.
    """
    def __init__(self, target_pos, env_latent_sizes):
        super().__init__()
        self.target_pos = np.array(target_pos)
        self.latent_sizes = env_latent_sizes

    def reset(self):
        pass

    def action(self, env):
        # Retrieve current position from the latest info
        if hasattr(env, 'last_info') and 's' in env.last_info:
            s = env.last_info['s']
            posX = s // self.latent_sizes[5]
            posY = s % self.latent_sizes[5]
            current_pos = np.array([posX, posY])
        else:
            current_pos = np.array([self.latent_sizes[4]//2, self.latent_sizes[5]//2])

        delta = self.target_pos - current_pos
        actions = []

        if delta[0] > 0:
            actions.append(3)  # Right
        elif delta[0] < 0:
            actions.append(2)  # Left

        if delta[1] > 0:
            actions.append(0)  # Up
        elif delta[1] < 0:
            actions.append(1)  # Down

        if not actions:
            return random.randint(0, env.action_space.n - 1)

        return random.choice(actions)

####################################
# Define Augmented Models
####################################

class AugmentedModel(nn.Module):
    """ Augmented Model Base Class """
    def __init__(self):
        super(AugmentedModel, self).__init__()

    def log_q_s(self, s=None):
        raise NotImplementedError

    def log_q_snext_sa(self, a, s=None, snext=None):
        raise NotImplementedError

    def log_q_o_s(self, o, s=None):
        raise NotImplementedError

    def log_q_r_s(self, r, s=None):
        raise NotImplementedError

    def log_q_d_s(self, d, s=None):
        raise NotImplementedError

    def log_q_a_s(self, a, s=None):
        raise NotImplementedError

    def log_q_s_h(self, regime, loq_q_sprev_hprev, a, o, r, d, with_done=False):
        assert (loq_q_sprev_hprev is None) == (a is None)

        if regime == 1:
            # Interventional regime: actions are uniform, no dependence on state
            return self.log_q_s(o, r, d)

        # Implement the log probability computation for observational regime
        # This is a placeholder and should be replaced with actual implementation
        return self.log_q_s(o, r, d)

    def log_prob_joint(self, regime, episode, states, with_done=False):
        """ Compute joint log probability. Placeholder implementation. """
        # This should be replaced with actual computation
        return torch.tensor(0.0)

    def log_prob(self, regime: torch.Tensor, episode: list, with_done: bool=False, return_log_q_s_h: bool=False):
        """ Compute log probability of an episode. Placeholder implementation. """
        # This should be replaced with actual computation
        return torch.tensor(0.0)

    def sample_states(self, regime: torch.Tensor, episode: list, with_done: bool=False):
        """ Sample states from the model. Placeholder implementation. """
        # This should be replaced with actual sampling
        return [torch.zeros(self.s_nvals) for _ in episode]

    def loss_nll(self, regime: torch.Tensor, episode: list, with_done: bool=False):
        """ Compute Negative Log-Likelihood loss. Placeholder implementation. """
        # This should be replaced with actual computation
        return torch.tensor(0.0)

    def loss_em(self, regime: torch.Tensor, episode: list, with_done: bool=False):
        """ Compute Expectation-Maximization loss. Placeholder implementation. """
        # This should be replaced with actual computation
        return torch.tensor(0.0)

class TabularAugmentedModel(AugmentedModel):
    """ Learnable Augmented Model using tabular probability distribution parameters. """

    def __init__(self, s_nvals, a_nvals, r_nvals):
        super(TabularAugmentedModel, self).__init__()
        self.s_nvals = s_nvals
        self.a_nvals = a_nvals
        self.r_nvals = r_nvals

        # Initialize model parameters with logits
        self.params_s_sa = nn.Parameter(torch.randn(s_nvals, a_nvals, s_nvals))
        self.params_r_s = nn.Parameter(torch.randn(s_nvals, r_nvals))
        self.params_a_s = nn.Parameter(torch.randn(s_nvals, a_nvals))

    def log_q_snext_sa(self, a):
        """
        Log probability of next state given current state and action.
        a: action index (int)
        Returns: Tensor of shape (s_nvals, s_nvals)
        """
        return F.log_softmax(self.params_s_sa[:, a, :], dim=-1)

    def log_q_r_s(self):
        """
        Log probability of reward given state.
        Returns: Tensor of shape (s_nvals, r_nvals)
        """
        return F.log_softmax(self.params_r_s, dim=-1)

    def log_q_a_s(self):
        """
        Log probability of action given state.
        Returns: Tensor of shape (s_nvals, a_nvals)
        """
        return F.log_softmax(self.params_a_s, dim=-1)

    def loss_nll(self, data):
        """
        Compute Negative Log-Likelihood loss.
        data: list of tuples (s, a, r, s_next)
        """
        loss = 0
        for (s, a, r, s_next) in data:
            if s >= self.s_nvals or s_next >= self.s_nvals:
                continue  # Skip invalid indices
            log_p_snext = self.log_q_snext_sa(a)[s, s_next]
            log_p_r = self.log_q_r_s()[s, r]
            log_p_a = self.log_q_a_s()[s, a]
            loss -= (log_p_snext + log_p_r + log_p_a)
        return loss / len(data) if len(data) > 0 else torch.tensor(0.0, requires_grad=True)

    def train_model(self, optimizer, data, epochs=10):
        """
        Train the augmented model using NLL loss.
        data: list of tuples (s, a, r, s_next)
        """
        self.train()
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            loss = self.loss_nll(data)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}')

####################################
# Define PPO Agents
####################################

class BaselineActorCriticPPO(nn.Module):
    """
    Baseline Actor-Critic model for PPO, without using any latent variables.
    Designed for continuous state inputs.
    """
    def __init__(self, state_dim, action_dim):
        super(BaselineActorCriticPPO, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)  # Process continuous state input

        # Actor network
        self.actor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """
        Select action based on state.
        """
        state_features = F.relu(self.fc1(state))
        action_probs = self.actor(state_features)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist

    def evaluate(self, state, action):
        """
        Evaluate action log probabilities, state values, and entropy.
        """
        state_features = F.relu(self.fc1(state))
        action_probs = self.actor(state_features)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        entropy = dist.entropy()
        state_value = self.critic(state_features)
        return action_logprobs, state_value.squeeze(-1), entropy, dist

class BaselinePPOAgent:
    """
    Baseline PPO Agent without augmented model integration.
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2):
        self.policy = BaselineActorCriticPPO(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = BaselineActorCriticPPO(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.mse_loss = nn.MSELoss()

    def select_action(self, state, memory):
        """
        Select action using the baseline policy and store in memory.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # Convert state to tensor
        action, log_prob, dist = self.policy_old.act(state_tensor)
        memory.states.append(state_tensor)
        memory.actions.append(torch.tensor([action]).to(device))
        memory.logprobs.append(log_prob)
        return action

    def update_policy(self, memory):
        """
        Update the PPO policy using the stored memory.
        """
        # Convert lists to tensors
        old_states = torch.cat(memory.states).to(device)  # Shape: (batch_size, state_dim)
        old_actions = torch.cat(memory.actions).to(device)  # Shape: (batch_size, 1)
        old_logprobs = torch.stack(memory.logprobs).to(device)
        rewards = memory.rewards
        is_terminals = memory.is_terminals

        # Compute discounted rewards
        discounted_rewards = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(is_terminals)):
            if done:
                G = 0
            G = reward + self.gamma * G
            discounted_rewards.insert(0, G)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)
        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        # Compute advantages
        with torch.no_grad():
            # Correctly compute state values by passing through the network
            state_features = F.relu(self.policy.fc1(old_states))
            state_values = self.policy.critic(state_features).squeeze(-1)
        advantages = discounted_rewards - state_values

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, entropy, dist = self.policy.evaluate(old_states, old_actions.squeeze(-1))

            # Finding the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages.detach()
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.detach()

            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2).mean() + \
                   0.5 * self.mse_loss(state_values, discounted_rewards) - \
                   0.01 * entropy.mean()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear memory
        memory.clear()

        # Log to console
        print(f"Baseline PPO Update - Loss: {loss.item():.4f}, Entropy: {entropy.mean().item():.4f}")

        # Return the loss for logging if needed
        return loss.item()

class AugmentedActorCriticPPO(nn.Module):
    """
    Actor-Critic network for PPO, incorporating the augmented model's features.
    Designed for discrete state indices using embedding layers.
    """
    def __init__(self, state_dim, augmented_model, action_dim):
        super(AugmentedActorCriticPPO, self).__init__()
        self.augmented_model = augmented_model  # TabularAugmentedModel

        # State embedding
        self.embedding = nn.Embedding(state_dim, 64)

        # Augmented features: Using log probabilities of actions given states
        self.augmented_feature_dim = augmented_model.a_nvals  # Example feature size

        # Combine state embedding and augmented features
        combined_dim = 64 + self.augmented_feature_dim

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """
        Generate an action using the current policy.
        """
        state_embedded = self.embedding(state)  # Shape: (batch_size, embed_dim)
        # Get augmented features from the augmented model
        with torch.no_grad():
            log_q_a_s = self.augmented_model.log_q_a_s()  # Shape: (s_nvals, a_nvals)
            augmented_features = log_q_a_s[state]  # Shape: (batch_size, a_nvals)
        combined = torch.cat([state_embedded, augmented_features], dim=-1)  # Shape: (batch_size, combined_dim)
        action_probs = self.actor(combined)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist

    def evaluate(self, state, action):
        """
        Evaluate action log probabilities, state values, and entropy for PPO training.
        """
        state_embedded = self.embedding(state)  # Shape: (batch_size, embed_dim)
        with torch.no_grad():
            log_q_a_s = self.augmented_model.log_q_a_s()  # Shape: (s_nvals, a_nvals)
            augmented_features = log_q_a_s[state]  # Shape: (batch_size, a_nvals)
        combined = torch.cat([state_embedded, augmented_features], dim=-1)  # Shape: (batch_size, combined_dim)
        action_probs = self.actor(combined)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        entropy = dist.entropy()
        state_value = self.critic(combined)
        return action_logprobs, state_value.squeeze(-1), entropy, dist

class AugmentedPPOAgent:
    """
    PPO Agent that utilizes the augmented model for enhanced policy learning.
    """
    def __init__(self, state_dim, augmented_model, action_dim, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2):
        self.policy = AugmentedActorCriticPPO(state_dim, augmented_model, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = AugmentedActorCriticPPO(state_dim, augmented_model, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.mse_loss = nn.MSELoss()

    def select_action(self, state, memory):
        """
        Select action using the augmented PPO policy and store in memory.
        """
        state_tensor = torch.LongTensor([state]).to(device)  # State is an integer index
        action, log_prob, dist = self.policy_old.act(state_tensor)
        memory.states.append(state_tensor)
        memory.actions.append(torch.tensor([action]).to(device))
        memory.logprobs.append(log_prob)
        return action

    def update_policy(self, memory):
        """
        Update the PPO policy using the stored memory.
        """
        # Convert lists to tensors
        old_states = torch.cat(memory.states).to(device)  # Shape: (batch_size, )
        old_actions = torch.cat(memory.actions).to(device)  # Shape: (batch_size, 1)
        old_logprobs = torch.stack(memory.logprobs).to(device)
        rewards = memory.rewards
        is_terminals = memory.is_terminals

        # Compute discounted rewards
        discounted_rewards = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(is_terminals)):
            if done:
                G = 0
            G = reward + self.gamma * G
            discounted_rewards.insert(0, G)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)
        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        # Compute advantages
        with torch.no_grad():
            # Correctly compute state values by using the evaluate method
            _, state_values, _, _ = self.policy.evaluate(old_states, old_actions.squeeze(-1))
        advantages = discounted_rewards - state_values

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, entropy, dist = self.policy.evaluate(old_states, old_actions.squeeze(-1))

            # Finding the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages.detach()
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.detach()

            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2).mean() + \
                   0.5 * self.mse_loss(state_values, discounted_rewards) - \
                   0.01 * entropy.mean()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear memory
        memory.clear()

        # Log to console
        print(f"Augmented PPO Update - Loss: {loss.item():.4f}, Entropy: {entropy.mean().item():.4f}")

        # Return the loss for logging if needed
        return loss.item()

####################################
# Define Memory for PPO
####################################

class Memory:
    """
    Memory for storing trajectories for PPO.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

####################################
# Define Learning and Evaluation Functions
####################################

class DatasetCustom(torch.utils.data.Dataset):
    """ Custom Dataset class for handling (regime, episode) tuples. """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_augmented_model(model, optimizer, env, policy_obs, policy_int, n_obs_steps, n_int_steps, latent_sizes, epochs=100, device='cpu'):
    """
    Train the augmented model using both observational and interventional data.
    Collects a specified number of transitions per epoch.
    """
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        data = []

        # Generate observational transitions
        transitions_collected = 0
        while transitions_collected < n_obs_steps:
            for episode in episode_generator(env, policy_obs, max_steps=env.max_steps):
                for transition in episode:
                    if transitions_collected >= n_obs_steps:
                        break
                    s, a, r, s_next = transition
                    # Compute state indices with floor and clamping
                    posX = int(np.clip(np.floor(s[4]), 0, latent_sizes[4]-1))
                    posY = int(np.clip(np.floor(s[5]), 0, latent_sizes[5]-1))
                    s_idx = posX * latent_sizes[5] + posY

                    posX_next = int(np.clip(np.floor(s_next[4]), 0, latent_sizes[4]-1))
                    posY_next = int(np.clip(np.floor(s_next[5]), 0, latent_sizes[5]-1))
                    s_next_idx = posX_next * latent_sizes[5] + posY_next

                    # Map reward to discrete category
                    r_cat = 1 if r > 0 else 0
                    r_cat = min(max(r_cat, 0), model.r_nvals - 1)  # Ensure within bounds

                    data.append((s_idx, a, r_cat, s_next_idx))
                    transitions_collected += 1

        # Generate interventional transitions
        transitions_collected = 0
        while transitions_collected < n_int_steps:
            for episode in episode_generator(env, policy_int, max_steps=env.max_steps):
                for transition in episode:
                    if transitions_collected >= n_int_steps:
                        break
                    s, a, r, s_next = transition
                    # Compute state indices with floor and clamping
                    posX = int(np.clip(np.floor(s[4]), 0, latent_sizes[4]-1))
                    posY = int(np.clip(np.floor(s[5]), 0, latent_sizes[5]-1))
                    s_idx = posX * latent_sizes[5] + posY

                    posX_next = int(np.clip(np.floor(s_next[4]), 0, latent_sizes[4]-1))
                    posY_next = int(np.clip(np.floor(s_next[5]), 0, latent_sizes[5]-1))
                    s_next_idx = posX_next * latent_sizes[5] + posY_next

                    # Map reward to discrete category
                    r_cat = 1 if r > 0 else 0
                    r_cat = min(max(r_cat, 0), model.r_nvals - 1)  # Ensure within bounds

                    data.append((s_idx, a, r_cat, s_next_idx))
                    transitions_collected += 1

        # Shuffle data
        random.shuffle(data)

        # Train for one epoch
        optimizer.zero_grad()
        loss = model.loss_nll(data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}/{epochs}, Loss: {total_loss:.4f}')

def episode_generator(env, policy, max_steps=100):
    """
    Generator that yields episodes to avoid storing all episodes in memory.
    """
    for _ in range(1):  # Single episode
        episode = []
        state, info = env.reset()
        policy.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = policy.action(env)
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward, next_state))
            state = next_state
            env.last_info = info  # Fixed the NameError by assigning 'info' instead of 'next_info'
            steps += 1
        yield episode

def evaluate_agent(agent, env, episodes=100, max_steps=200, device='cpu'):
    """
    Evaluate the PPO agent by running multiple episodes and computing the average reward.
    """
    agent.policy.eval()
    total_rewards = []
    with torch.no_grad():
        for episode_num in range(1, episodes + 1):
            state, info = env.reset()
            done = False
            steps = 0
            episode_reward = 0
            while not done and steps < max_steps:
                if isinstance(agent, AugmentedPPOAgent):
                    # For AugmentedPPOAgent, pass 's' as state
                    action = agent.select_action(info['s'], Memory())
                else:
                    # For BaselinePPOAgent, pass 'obs' as state
                    action = agent.select_action(state, Memory())
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state
                steps += 1
            total_rewards.append(episode_reward)
            if episode_num % 10 == 0:
                print(f'Episode {episode_num}: Total Reward: {episode_reward:.2f}')
    average_reward = np.mean(total_rewards)
    print(f'Average Reward over {episodes} episodes: {average_reward:.2f}')
    return average_reward

####################################
# Define Causal Analysis Functions
####################################

def get_best_for_data(X, Y, regs):
    """
    Select the best model based on mean squared error.
    """
    x_train, x_test, y_train, y_test = X, X, Y, Y
    val_errs = []
    models = []
    for reg in regs:
        model = copy.deepcopy(reg)
        model.fit(x_train, y_train)
        val_errs.append(np.mean((y_test - model.predict(x_test))**2))
        models.append(copy.deepcopy(model))
    min_ind = val_errs.index(min(val_errs))
    return copy.deepcopy(models[min_ind])

def run_method(X_rct, Y_rct, T_rct, X_obs, Y_obs, T_obs):
    """
    Run the causal analysis method using causal forests and linear regression.
    """
    # Placeholder: Implement causal forest using scikit-learn or other libraries
    # This is a simplified version and may need adjustments based on actual requirements

    # Step 1: Estimate ^w with causal forest as family Q from observational dataset
    # For demonstration, we'll use a RandomForestRegressor as a proxy
    from sklearn.ensemble import RandomForestRegressor

    cf_model = RandomForestRegressor(n_estimators=200, random_state=0)
    cf_model.fit(X_obs, Y_obs)

    # Evaluate on interventional dataset
    omega_int_pred = cf_model.predict(X_rct)

    # Step 2: Compute cate_int_est
    cate_int_est = 2 * np.multiply(Y_rct, 2*T_rct-1).ravel()
    assert(cate_int_est.shape == omega_int_pred.shape)

    # Step 3: Compute theta * x = ^tau - ^w
    eta_est = cate_int_est - omega_int_pred
    assert(len(eta_est.shape) == 1)
    
    # Step 4: Linear regression to find theta
    from sklearn.linear_model import Ridge, Lasso, LinearRegression

    linear_regs = [LinearRegression()]
    linear_regs.extend([Lasso(alpha=alph) for alph in [1e-5,1e-3,1e-1,1,1e+1,1e+3,1e+5]])
    linear_regs.extend([Ridge(alpha=alph) for alph in [1e-5,1e-3,1e-1,1,1e+1,1e+3,1e+5]])

    best_eta_est_linear = get_best_for_data(X_rct.reshape(-1,1), eta_est, regs=linear_regs)

    return copy.deepcopy(best_eta_est_linear), eta_est, omega_int_pred

def empiricalJS(model_q, model_p, policy, max_length=1, n_iter=500):
    """
    Compute empirical Jensen-Shannon divergence between two models.
    """
    settings_mp = model_p.get_settings()
    env_p = NavigationEnv(num_games=1, max_steps=max_length)  # Placeholder for PomdpEnv
    confounded_env_p = ConfounderEnvWrapper(env_p, noise_std=0.05)
    
    settings_mq = model_q.get_settings()
    env_q = NavigationEnv(num_games=1, max_steps=max_length)  # Placeholder for PomdpEnv
    confounded_env_q = ConfounderEnvWrapper(env_q, noise_std=0.05)

    n_iter = n_iter
    loss_q, loss_p = 0, 0
    
    # E x~q(x) [log(q(x)) - log(q(x) + p(x))]
    for _ in range(n_iter):
        ep = rollout(confounded_env_q, policy)
        ep = [torch.tensor(t).unsqueeze(0).to(device) for t in ep]
        regime = torch.tensor(1.).unsqueeze(0).to(device)
        
        loss_q += model_q.log_prob(regime, ep)[0]
        loss_q -= torch.logsumexp(torch.cat([model_q.log_prob(regime, ep), 
                                             model_p.log_prob(regime, ep)]).unsqueeze(0), 1)[0]

    # E x~p(x) [log(p(x)) - log(q(x) + p(x))]
    for _ in range(n_iter):
        ep = rollout(confounded_env_p, policy)
        ep = [torch.tensor(t).unsqueeze(0).to(device) for t in ep]
        regime = torch.tensor(1.).unsqueeze(0).to(device)
        
        loss_p += model_p.log_prob(regime, ep)[0]
        loss_p -= torch.logsumexp(torch.cat([model_q.log_prob(regime, ep), 
                                             model_p.log_prob(regime, ep)]).unsqueeze(0), 1)[0]
        
    return torch.log(torch.tensor(2.)) + (loss_p + loss_q)/(2*n_iter)

def cross_entropy_empirical(model_q, data_p, batch_size, with_done=False):
    """
    Compute cross-entropy between model_q and data_p.
    """
    device = next(model_q.parameters()).device

    dataloader_p = torch.utils.data.DataLoader(DatasetCustom(data_p), batch_size=batch_size)

    ce = 0

    for batch in dataloader_p:
        regime, episode = batch
        regime = regime.to(device)
        episode = [tensor.to(device) for tensor in episode] 

        log_prob_q = model_q.log_prob(regime, episode, with_done=with_done)

        ce += -log_prob_q.sum(dim=0)

    ce /= len(data_p)

    return ce

def kl_div_empirical(model_p, model_q, data_p, batch_size, with_done=False):
    """
    Compute KL divergence between model_p and model_q on data_p.
    """
    assert next(model_q.parameters()).device == next(model_p.parameters()).device

    device = next(model_p.parameters()).device

    # Build DataLoaders
    dataloader_p = torch.utils.data.DataLoader(DatasetCustom(data_p), batch_size=batch_size)

    # KL(p|q) = E x~p(x) [log(p(x)) - log(q(x))]
    kl_p_q = 0

    for batch in dataloader_p:
        regime, episode = batch
        regime = regime.to(device)
        episode = [tensor.to(device) for tensor in episode]

        log_prob_q = model_q.log_prob(regime, episode, with_done=with_done)
        log_prob_p = model_p.log_prob(regime, episode, with_done=with_done)

        kl_p_q += (log_prob_p - log_prob_q).sum(dim=0)

    kl_p_q /= len(data_p)

    return kl_p_q

def js_div_empirical(model_q, model_p, data_q, data_p, batch_size, with_done=False):
    """
    Compute Jensen-Shannon divergence between model_q and model_p on data_q and data_p.
    """
    assert next(model_q.parameters()).device == next(model_p.parameters()).device

    device = next(model_p.parameters()).device

    # Build DataLoaders
    dataloader_q = torch.utils.data.DataLoader(DatasetCustom(data_q), batch_size=batch_size)
    dataloader_p = torch.utils.data.DataLoader(DatasetCustom(data_p), batch_size=batch_size)

    # KL(p|m) = E x~p(x) [log(p(x)) - log(q(x) + p(x)) + log(2)]
    kl_p_m = 0

    for batch in dataloader_p:
        regime, episode = batch
        regime = regime.to(device)
        episode = [tensor.to(device) for tensor in episode]

        log_prob_q = model_q.log_prob(regime, episode, with_done=with_done)
        log_prob_p = model_p.log_prob(regime, episode, with_done=with_done)
        log_prob_m = torch.logsumexp(torch.stack([log_prob_q, log_prob_p], dim=0), dim=0)  # log(q(x) + p(x))

        kl_p_m += (log_prob_p - log_prob_m).sum(dim=0)

    kl_p_m /= len(data_p)
    kl_p_m += torch.log(torch.tensor(2.0).to(device))

    # KL(q|m) = E x~q(x) [log(q(x)) - log(q(x) + p(x)) + log(2)]
    kl_q_m = 0

    for batch in dataloader_q:
        regime, episode = batch
        regime = regime.to(device)
        episode = [tensor.to(device) for tensor in episode]

        log_prob_q = model_q.log_prob(regime, episode, with_done=with_done)
        log_prob_p = model_p.log_prob(regime, episode, with_done=with_done)
        log_prob_m = torch.logsumexp(torch.stack([log_prob_q, log_prob_p], dim=0), dim=0)  # log(q(x) + p(x))

        kl_q_m += (log_prob_q - log_prob_m).sum(dim=0)

    kl_q_m /= len(data_q)
    kl_q_m += torch.log(torch.tensor(2.0).to(device))

    # JS(p|q) = (KL(p|m) + KL(q|m)) / 2
    return (kl_q_m + kl_p_m) / 2

####################################
# Define Causal Analysis Functions
####################################

def construct_dataset(env, policy, n_samples, regime):
    """
    Construct a dataset (of n samples) by collecting rollouts using a given 
    policy in a given environment.
    """
    data = []
    for _ in range(n_samples):
        policy.reset()
        episode = rollout(env, policy)
        data.append((regime, episode))
    return data

def rollout(env, policy):
    """ Perform rollout of the game and return episodes in a list """
    episode = []
    o, info = env.reset()
    r, done = 0.0, False
    episode += [o, r, done]

    while not done:
        action = policy.action(env)
        o, r, done, info = env.step(action)
        episode += [action, o, r, done]
    return episode

####################################
# Main Execution
####################################

def main():
    # Define training parameters
    training_params = {
        "num_data_steps": 100000,       # Total number of data steps for augmented model
        "dualvae_epochs": 100,           # Augmented model training epochs
        "dualvae_batch_size": 128,       # Augmented model batch size (not used in current code)
        "dualvae_learning_rate": 1e-3,   # Augmented model learning rate
        "ppo_max_episodes": 20000,        # RL agent training episodes (reduced for demonstration)
        "ppo_max_timesteps": 200,         # Max timesteps per episode
        "ppo_update_timestep": 4000,      # Timesteps per PPO update
        "ppo_lr": 3e-4,                    # RL agent learning rate
        "ppo_gamma": 0.99,                 # Discount factor
        "ppo_K_epochs": 10,                # PPO epochs
        "ppo_eps_clip": 0.2,               # PPO epsilon clip
        "ppo_causal_weight": 0.1,          # PPO causal weight (not used in current code)
        "dualvae_loss_weights": [1.0, 1.0, 1.0, 0.1, 0.1, 0.001, 0.001],  # Dual VAE loss weights
        "causal_var_indices": [2, 3],      # posX and posY indices within s_x
        "moving_average_window": 50        # Moving average window for logging
    }

    # Define environment parameters
    latent_sizes = [1, 3, 6, 40, 32, 32]  # From dSprites dataset
    s_nvals = latent_sizes[4] * latent_sizes[5]  # 32 * 32 = 1024
    a_nvals = 4
    r_nvals = 2

    # Create the Navigation environment and wrap it with ConfounderEnvWrapper
    env = NavigationEnv(num_games=1, max_steps=training_params["ppo_max_timesteps"])
    confounded_env = ConfounderEnvWrapper(env, noise_std=0.05)

    # Define policies
    target_position = [env.latent_sizes[4] // 2, env.latent_sizes[5] // 2]
    expert_policy = ExpertPolicy(target_pos=target_position, env_latent_sizes=env.latent_sizes)
    uniform_policy = UniformPolicy(a_nvals=a_nvals)

    # Initialize the augmented model
    model = TabularAugmentedModel(s_nvals, a_nvals, r_nvals).to(device)

    # Define optimizer for the augmented model
    optimizer_model = optim.Adam(model.parameters(), lr=training_params["dualvae_learning_rate"])

    # Calculate the number of observational and interventional data steps per epoch
    # To distribute num_data_steps across all epochs
    epochs = training_params["dualvae_epochs"]
    num_data_steps = training_params["num_data_steps"]
    n_obs_steps = num_data_steps // (2 * epochs)  # Half for observational
    n_int_steps = num_data_steps // (2 * epochs)  # Half for interventional

    print("Training Augmented Model with Backdoor Adjustment...")
    train_augmented_model(
        model, optimizer_model, confounded_env, expert_policy, uniform_policy,
        n_obs_steps=n_obs_steps, n_int_steps=n_int_steps, latent_sizes=env.latent_sizes,
        epochs=training_params["dualvae_epochs"], device=device
    )

    # Initialize the PPO agents
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize the Baseline PPO Agent
    ppo_agent_baseline = BaselinePPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=training_params["ppo_lr"],
        gamma=training_params["ppo_gamma"],
        K_epochs=training_params["ppo_K_epochs"],
        eps_clip=training_params["ppo_eps_clip"]
    )

    # Initialize the Augmented PPO Agent
    ppo_agent_augmented = AugmentedPPOAgent(
        state_dim=s_nvals,  # AugmentedPPOAgent expects state as integer index
        augmented_model=model,
        action_dim=action_dim,
        lr=training_params["ppo_lr"],
        gamma=training_params["ppo_gamma"],
        K_epochs=training_params["ppo_K_epochs"],
        eps_clip=training_params["ppo_eps_clip"]
    )

    # Initialize Memory
    memory_baseline = Memory()
    memory_augmented = Memory()

    # Initialize counters
    timestep_baseline = 0
    timestep_augmented = 0

    # Training loop for Baseline PPO Agent
    print("\nTraining Baseline PPO Agent...")
    reward_history_baseline = []
    moving_average_window = training_params["moving_average_window"]

    for episode in tqdm(range(1, training_params["ppo_max_episodes"] + 1), desc="Baseline PPO Training"):
        state, info = confounded_env.reset()
        done = False
        episode_reward = 0
        for t in range(training_params["ppo_max_timesteps"]):
            action = ppo_agent_baseline.select_action(state, memory_baseline)
            next_state, reward, done, info = confounded_env.step(action)

            # Save reward and is_terminal flag
            memory_baseline.rewards.append(reward)
            memory_baseline.is_terminals.append(done)

            state = next_state
            episode_reward += reward
            timestep_baseline += 1

            # Update PPO Agent every ppo_update_timestep
            if timestep_baseline % training_params["ppo_update_timestep"] == 0:
                ppo_agent_baseline.update_policy(memory_baseline)

        # Logging
        reward_history_baseline.append(episode_reward)
        if len(reward_history_baseline) > moving_average_window:
            reward_history_baseline.pop(0)
        avg_reward = np.mean(reward_history_baseline)

        # Print progress
        if episode % 100 == 0 or episode == 1:
            print(f'Episode {episode}/{training_params["ppo_max_episodes"]}, '
                  f'Episode Reward: {episode_reward:.2f}, '
                  f'Average Reward (last {moving_average_window}): {avg_reward:.2f}')

    # Training loop for Augmented PPO Agent
    print("\nTraining Augmented PPO Agent...")
    reward_history_augmented = []

    for episode in tqdm(range(1, training_params["ppo_max_episodes"] + 1), desc="Augmented PPO Training"):
        state, info = confounded_env.reset()
        done = False
        episode_reward = 0
        while not done:
            # For AugmentedPPOAgent, pass 's' as state
            s = info['s']
            action = ppo_agent_augmented.select_action(s, memory_augmented)
            next_state, reward, done, info = confounded_env.step(action)

            # Save reward and is_terminal flag
            memory_augmented.rewards.append(reward)
            memory_augmented.is_terminals.append(done)

            state = next_state
            episode_reward += reward
            timestep_augmented += 1

            # Update PPO Agent every ppo_update_timestep
            if timestep_augmented % training_params["ppo_update_timestep"] == 0:
                ppo_agent_augmented.update_policy(memory_augmented)

        # Logging
        reward_history_augmented.append(episode_reward)
        if len(reward_history_augmented) > moving_average_window:
            reward_history_augmented.pop(0)
        avg_reward = np.mean(reward_history_augmented)

        # Print progress
        if episode % 100 == 0 or episode == 1:
            print(f'Episode {episode}/{training_params["ppo_max_episodes"]}, '
                  f'Episode Reward: {episode_reward:.2f}, '
                  f'Average Reward (last {moving_average_window}): {avg_reward:.2f}')

    # Evaluate the PPO agents
    print("\nEvaluating Baseline PPO Agent...")
    average_reward_baseline = evaluate_agent(
        ppo_agent_baseline, confounded_env,
        episodes=100,
        max_steps=training_params["ppo_max_timesteps"],
        device=device
    )

    print("\nEvaluating Augmented PPO Agent...")
    average_reward_augmented = evaluate_agent(
        ppo_agent_augmented, confounded_env,
        episodes=100,
        max_steps=training_params["ppo_max_timesteps"],
        device=device
    )

    # Plot training rewards
    plt.figure(figsize=(12, 6))
    plt.plot(reward_history_baseline, label='Baseline PPO')
    plt.plot(reward_history_augmented, label='Augmented PPO')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'PPO Agents Training Rewards (Average over Last {moving_average_window} Episodes)')
    plt.legend()
    plt.grid(True)
    plt.show()

####################################
# Entry Point
####################################

if __name__ == "__main__":
    main()