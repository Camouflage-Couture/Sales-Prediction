# --------------------------- Importing Necessary Libraries --------------------------- #

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cross_decomposition import CCA
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import gym
from gym import spaces
import os
import time
import wandb  # For experiment tracking
import pandas as pd
import urllib.request

# --------------------------- Reproducibility and Device Configuration --------------------------- #

torch.manual_seed(42)
np.random.seed(42)
random_seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------- Custom Gym Environment Definition --------------------------- #

class NavigationEnv(gym.Env):
    """
    Custom Navigation Environment using dSprites dataset.
    The agent's goal is to navigate to a target position while dealing with confounders.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_games=1, max_steps=100):
        super(NavigationEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        self.max_steps = max_steps
        self.current_step = 0

        self.dataset_path = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        if not os.path.exists(self.dataset_path):
            print(f"dSprites dataset not found at {self.dataset_path}. Downloading...")
            self.download_dsprites()
        dataset = np.load(self.dataset_path, allow_pickle=True, encoding='latin1')
        self.imgs = dataset['imgs'].reshape(-1, 64, 64, 1).astype(np.float32)
        metadata = dataset['metadata'][()]
        self.latent_sizes = metadata['latents_sizes']
        self.latent_bases = np.concatenate(
            (metadata['latents_sizes'][::-1].cumprod()[::-1][1:], np.array([1]))
        )
        self.latent_names = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY']
        self.latent_dim = len(self.latent_names)
        self.num_games = num_games

        self.current_latents = np.zeros((self.num_games, self.latent_dim))
        self.reset_all_latents()

        print(f'Dataset loaded. Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, '
              f'Datapoints: {len(self.imgs)}, Latent Dimensions: {self.latent_dim}, '
              f'Latent Bases: {self.latent_bases}')

    def download_dsprites(self):
        url = 'https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true'
        urllib.request.urlretrieve(url, self.dataset_path)
        print("dSprites dataset downloaded successfully.")

    def sample_latent(self):
        latent = np.zeros(self.latent_dim)
        latent[0] = 0  # color is fixed for simplicity
        latent[1] = np.random.randint(0, 3)  # shape
        latent[2] = np.random.randint(0, 6)  # scale
        latent[3] = np.random.randint(0, 40)  # orientation
        latent[4] = np.random.randint(0, 32)  # posX
        latent[5] = np.random.randint(0, 32)  # posY
        return latent

    def sample_all_latents(self):
        latents = np.zeros((self.num_games, self.latent_dim))
        for idx in range(self.num_games):
            latents[idx] = self.sample_latent()
        return latents

    def latents_to_index(self, latents):
        return np.dot(latents, self.latent_bases).astype(int)

    def get_observation(self, index):
        idx = self.latents_to_index(self.current_latents[index])
        idx = np.clip(idx, 0, len(self.imgs) - 1)
        return self.imgs[idx].copy()

    def step(self, action):
        index = 0  # Single game
        if action == 0:  # Up
            if self.current_latents[index, 5] < 31:
                self.current_latents[index, 5] += 1
        elif action == 1:  # Down
            if self.current_latents[index, 5] > 0:
                self.current_latents[index, 5] -= 1
        elif action == 2:  # Left
            if self.current_latents[index, 4] > 0:
                self.current_latents[index, 4] -= 1
        elif action == 3:  # Right
            if self.current_latents[index, 4] < 31:
                self.current_latents[index, 4] += 1
        else:
            raise ValueError(f"Invalid action: {action}")

        target_pos = np.array([16, 16])
        current_pos = self.current_latents[index, 4:6]
        prev_distance = np.linalg.norm(current_pos - target_pos)
        new_pos = self.current_latents[index, 4:6]
        new_distance = np.linalg.norm(new_pos - target_pos)
        distance_reduction = prev_distance - new_distance
        step_penalty = -0.2
        boundary_penalty = 0.0
        if (action == 0 and self.current_latents[index, 5] >= 31) or \
           (action == 1 and self.current_latents[index, 5] <= 0) or \
           (action == 2 and self.current_latents[index, 4] <= 0) or \
           (action == 3 and self.current_latents[index, 4] >= 31):
            boundary_penalty = -1.0

        reward = distance_reduction + step_penalty + boundary_penalty
        done = False
        if new_distance == 0:
            reward += 20
            done = True
            self.current_latents[index] = self.sample_latent()
        else:
            done = False

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        obs = self.current_latents[index].copy()
        return obs, reward, done, {}

    def reset(self):
        self.current_latents = self.sample_all_latents()
        self.current_step = 0
        state = self.current_latents[0].copy()
        return state

    def render(self, mode='human'):
        pass

    def reset_all_latents(self):
        self.current_latents = self.sample_all_latents()
        self.current_step = 0

class ConfounderEnvWrapper(gym.Wrapper):
    def __init__(self, env, confounder_freqs=None, confounder_amplitudes=None, noise_std=0.1):
        super(ConfounderEnvWrapper, self).__init__(env)
        self.confounder_freqs = confounder_freqs if confounder_freqs is not None else [0.5, 0.3, 0.2, 0.4]
        self.confounder_amplitudes = confounder_amplitudes if confounder_amplitudes is not None else [0.1, 0.1, 0.15, 0.15]  # Low confounding
        self.noise_std = noise_std
        self.time_step = 0

    def step(self, action):
        confounder = sum(
            amp * np.sin(freq * self.time_step) + amp * np.cos(freq * self.time_step)
            for freq, amp in zip(self.confounder_freqs, self.confounder_amplitudes)
        )
        state, reward, done, info = self.env.step(action)
        mod_state = state + confounder + np.random.normal(0, self.noise_std, size=state.shape)
        mod_reward = reward + confounder * 0.2
        self.time_step += 1
        return mod_state, mod_reward, done, info

    def reset(self):
        self.time_step = 0
        state = self.env.reset()
        confounder = sum(
            amp * np.sin(freq * self.time_step) + amp * np.cos(freq * self.time_step)
            for freq, amp in zip(self.confounder_freqs, self.confounder_amplitudes)
        )
        mod_state = state + confounder + np.random.normal(0, self.noise_std, size=state.shape)
        return mod_state

# --------------------------- Neural Network Components --------------------------- #

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.ReLU(), batch_norm=False):
        super(MLPNetwork, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            if batch_norm and i < len(dims) - 1:
                layers.append(nn.BatchNorm1d(dims[i]))
            if i < len(dims) - 1:
                layers.append(activation)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[256, 256], batch_norm=False, activation=nn.ReLU()):
        super(Encoder, self).__init__()
        self.encoder = MLPNetwork(input_dim, latent_dim, hidden_dims, batch_norm=batch_norm, activation=activation)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z)
        return mu, logvar

class StructuralEquation(nn.Module):
    def __init__(self, s_dim, z_dim, hidden_dims=[256, 256], activation=nn.ReLU()):
        super(StructuralEquation, self).__init__()
        self.model = MLPNetwork(s_dim, z_dim, hidden_dims, activation=activation)

    def forward(self, s):
        return self.model(s)

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256], batch_norm=False, activation=nn.ReLU()):
        super(Decoder, self).__init__()
        self.decoder = MLPNetwork(input_dim, output_dim, hidden_dims, batch_norm=batch_norm, activation=activation)

    def forward(self, z):
        return self.decoder(z)

class VAEOutput:
    def __init__(self, mean, logvar, sample, recon=None):
        self.mean = mean
        self.logvar = logvar
        self.sample = sample
        self.recon = recon

# --------------------------- DualVAE Model --------------------------- #

class DualVAEModel(nn.Module):
    def __init__(self, state_dim, action_dim, reward_dim=1, z_dim=4, s_dim=4, hidden_dims=[256, 256], activation=nn.ReLU(), batch_norm=False):
        super(DualVAEModel, self).__init__()
        self.z_dim = z_dim
        self.s_dim = s_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim

        input_dim_x = state_dim + action_dim
        input_dim_y = reward_dim + state_dim

        self.encoder_s = Encoder(input_dim_x + input_dim_y, s_dim, hidden_dims, batch_norm=batch_norm, activation=activation)
        self.structural_eq_zx = StructuralEquation(s_dim, z_dim, hidden_dims, activation=activation)
        self.structural_eq_zy = StructuralEquation(s_dim, z_dim, hidden_dims, activation=activation)
        self.encoder_residual_x = Encoder(input_dim_x, z_dim, hidden_dims, batch_norm=batch_norm, activation=activation)
        self.encoder_residual_y = Encoder(input_dim_y, z_dim, hidden_dims, batch_norm=batch_norm, activation=activation)
        self.inference_encoder_zx = Encoder(state_dim, z_dim, hidden_dims, batch_norm=batch_norm, activation=activation)
        self.decoder_X = Decoder(z_dim + s_dim, input_dim_x, hidden_dims, batch_norm=batch_norm, activation=activation)
        self.decoder_Y = Decoder(z_dim + s_dim, input_dim_y, hidden_dims, batch_norm=batch_norm, activation=activation)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) + 1e-7
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y, use_scm=False):
        xy = torch.cat([x, y], dim=1)
        s_mu, s_logvar = self.encoder_s(xy)
        s = self.reparameterize(s_mu, s_logvar)

        z_x_mu_structural = self.structural_eq_zx(s)
        z_y_mu_structural = self.structural_eq_zy(s)

        z_x_res_mu, z_x_res_logvar = self.encoder_residual_x(x)
        z_y_res_mu, z_y_res_logvar = self.encoder_residual_y(y)

        z_x_mu = z_x_mu_structural + z_x_res_mu
        z_y_mu = z_y_mu_structural + z_y_res_mu

        z_x_logvar = z_x_res_logvar
        z_y_logvar = z_y_res_logvar

        z_x = self.reparameterize(z_x_mu, z_x_logvar)
        z_y = self.reparameterize(z_y_mu, z_y_logvar)

        recon_X = self.decoder_X(torch.cat([z_x, s], dim=1))
        recon_Y = self.decoder_Y(torch.cat([z_y, s], dim=1))

        z_x_infer_mu, z_x_infer_logvar = self.inference_encoder_zx(x[:, :self.state_dim])
        z_x_infer = self.reparameterize(z_x_infer_mu, z_x_infer_logvar)

        mean = {"s": s_mu, "z_x": z_x_mu, "z_y": z_y_mu, "z_x_infer": z_x_infer_mu}
        logvar = {"s": s_logvar, "z_x": z_x_logvar, "z_y": z_y_logvar, "z_x_infer": z_x_infer_logvar}
        sample = {"s": s, "z_x": z_x, "z_y": z_y, "z_x_infer": z_x_infer}
        recon = {"x": recon_X, "y": recon_Y}
        output = VAEOutput(mean, logvar, sample, recon)
        return output

    def infer_z_x(self, state):
        z_x_mu, _ = self.inference_encoder_zx(state)
        return z_x_mu

# --------------------------- PPO Actor-Critic Models --------------------------- #

class PPOActorCritic(nn.Module):
    def __init__(self, z_dim, action_dim):
        super(PPOActorCritic, self).__init__()
        input_dim = z_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.actor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, z_x):
        x = z_x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        features = self.relu2(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value.squeeze(-1), features

class PPOActorCriticBaseline(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOActorCriticBaseline, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.actor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        x = state
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        features = self.relu2(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value.squeeze(-1), features

class PPOActorCriticBaselineCausal(nn.Module):
    def __init__(self, causal_dim, action_dim):
        super(PPOActorCriticBaselineCausal, self).__init__()
        input_dim = causal_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.actor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, causal_vars):
        x = causal_vars
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        features = self.relu2(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value.squeeze(-1), features

# --------------------------- PPO Agent Classes --------------------------- #

class CausalPPOAgent:
    def __init__(self, z_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2):
        self.policy = PPOActorCritic(z_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = PPOActorCritic(z_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.mse_loss = nn.MSELoss()

    def select_action(self, z_x):
        z_x = torch.FloatTensor(z_x).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs, state_value, _ = self.policy_old(z_x)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        return action.item(), action_logprob.item(), state_value.item()

    def update_policy(self, memory):
        old_z_x = torch.FloatTensor(memory['z_x']).to(device)
        old_actions = torch.LongTensor(memory['actions']).to(device)
        old_logprobs = torch.FloatTensor(memory['logprobs']).to(device)
        old_state_values = torch.FloatTensor(memory['state_values']).to(device)
        rewards = memory['rewards']
        is_terminals = memory['is_terminals']

        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.FloatTensor(returns).to(device)

        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        advantages = returns - old_state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for _ in range(self.K_epochs):
            action_probs, state_values, _ = self.policy(old_z_x)
            dist = torch.distributions.Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy()

            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * self.mse_loss(state_values.squeeze(), returns)
            loss = policy_loss + value_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

class BaselinePPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2):
        self.policy = PPOActorCriticBaseline(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = PPOActorCriticBaseline(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs, state_value, _ = self.policy_old(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        return action.item(), action_logprob.item(), state_value.item()

    def update_policy(self, memory):
        old_states = torch.FloatTensor(memory['states']).to(device)
        old_actions = torch.LongTensor(memory['actions']).to(device)
        old_logprobs = torch.FloatTensor(memory['logprobs']).to(device)
        old_state_values = torch.FloatTensor(memory['state_values']).to(device)
        rewards = memory['rewards']
        is_terminals = memory['is_terminals']

        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.FloatTensor(returns).to(device)

        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        advantages = returns - old_state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for _ in range(self.K_epochs):
            action_probs, state_values, _ = self.policy(old_states)
            dist = torch.distributions.Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy()
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * self.mse_loss(state_values.squeeze(), returns)
            loss = policy_loss + value_loss - 0.01 * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())

class BaselineCausalPPOAgent:
    def __init__(self, causal_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2):
        self.policy = PPOActorCriticBaselineCausal(causal_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = PPOActorCriticBaselineCausal(causal_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.mse_loss = nn.MSELoss()

    def select_action(self, causal_vars):
        causal_vars = torch.FloatTensor(causal_vars).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs, state_value, _ = self.policy_old(causal_vars)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        return action.item(), action_logprob.item(), state_value.item()

    def update_policy(self, memory):
        old_causal = torch.FloatTensor(np.array(memory['causal_vars'])).to(device)
        old_actions = torch.LongTensor(memory['actions']).to(device)
        old_logprobs = torch.FloatTensor(memory['logprobs']).to(device)
        old_state_values = torch.FloatTensor(memory['state_values']).to(device)
        rewards = memory['rewards']
        is_terminals = memory['is_terminals']

        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.FloatTensor(returns).to(device)

        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        advantages = returns - old_state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for _ in range(self.K_epochs):
            action_probs, state_values, _ = self.policy(old_causal)
            dist = torch.distributions.Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy()

            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * self.mse_loss(state_values.squeeze(), returns)
            loss = policy_loss + value_loss - 0.01 * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())

# --------------------------- Memory Classes --------------------------- #

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = []

class MemoryCausal:
    def __init__(self):
        self.causal_vars = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = []

# --------------------------- Loss Functions --------------------------- #

def orthogonality_loss(x, y):
    cosine_similarity = torch.nn.functional.cosine_similarity(x, y, dim=1)
    ortho_loss = torch.mean((cosine_similarity)**2)
    return ortho_loss

def kl_divergence_loss(mean, logvar):
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()

def conditional_independence_loss(z_x, z_y, s):
    # Residuals after removing the influence of s
    z_x_residual = z_x - s
    z_y_residual = z_y - s
    corr = torch.mean(z_x_residual * z_y_residual, dim=0)
    ci_loss = torch.sum(corr ** 2)
    return ci_loss

# --------------------------- DualVAE Loss Function --------------------------- #

def dualvae_loss(x, y, output, weights, use_scm=False):
    # If use_scm=False, we skip ci_loss
    recon_X_loss = nn.functional.mse_loss(output.recon["x"], x, reduction='sum')
    recon_Y_loss = nn.functional.mse_loss(output.recon["y"], y, reduction='sum')
    kld_s = kl_divergence_loss(output.mean["s"], output.logvar["s"])
    kld_zx = kl_divergence_loss(output.mean["z_x"], output.logvar["z_x"])
    kld_zy = kl_divergence_loss(output.mean["z_y"], output.logvar["z_y"])
    kld_total = kld_s + kld_zx + kld_zy

    ortho_loss_zx_s = orthogonality_loss(output.sample["z_x"], output.sample["s"])
    ortho_loss_zy_s = orthogonality_loss(output.sample["z_y"], output.sample["s"])
    
    ci_loss = torch.tensor(0.0, device=device)
    if use_scm:
        ci_loss = conditional_independence_loss(output.sample["z_x"], output.sample["z_y"], output.sample["s"])

    inference_zx_loss = nn.functional.mse_loss(output.sample["z_x_infer"], output.sample["z_x"], reduction='sum')

    total_loss = (weights[0]*recon_X_loss +
                  weights[1]*recon_Y_loss +
                  weights[2]*kld_total +
                  weights[3]*ortho_loss_zx_s +
                  weights[4]*ortho_loss_zy_s +
                  weights[5]*ci_loss +  # Only added if use_scm=True
                  weights[6]*inference_zx_loss)

    loss_dict = {
        "total_loss": total_loss,
        "recon_X_loss": recon_X_loss,
        "recon_Y_loss": recon_Y_loss,
        "kld_total": kld_total,
        "ortho_loss_zx_s": ortho_loss_zx_s,
        "ortho_loss_zy_s": ortho_loss_zy_s,
        "ci_loss": ci_loss,
        "inference_zx_loss": inference_zx_loss
    }
    return loss_dict

# --------------------------- Training DualVAE --------------------------- #

def train_dualvae(model, weights, dataset, device, use_scm=False, num_epochs=100, batch_size=128, learning_rate=1e-3, run_id=1):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    training_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(x_batch, y_batch, use_scm=use_scm)
            loss_dict = dualvae_loss(x_batch, y_batch, output, weights, use_scm=use_scm)
            loss = loss_dict["total_loss"]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader.dataset)
        training_losses.append(avg_loss)

        wandb.log({
            f"DualVAE_{'SCM' if use_scm else 'NoSCM'}/Epoch": epoch + 1,
            f"DualVAE_{'SCM' if use_scm else 'NoSCM'}/Total Loss": avg_loss,
            f"DualVAE_{'SCM' if use_scm else 'NoSCM'}/Reconstruction X Loss": loss_dict["recon_X_loss"].item() / len(dataloader.dataset),
            f"DualVAE_{'SCM' if use_scm else 'NoSCM'}/Reconstruction Y Loss": loss_dict["recon_Y_loss"].item() / len(dataloader.dataset),
            f"DualVAE_{'SCM' if use_scm else 'NoSCM'}/KL Divergence": loss_dict["kld_total"].item() / len(dataloader.dataset),
            f"DualVAE_{'SCM' if use_scm else 'NoSCM'}/Orthogonality ZX-S": loss_dict["ortho_loss_zx_s"].item() / len(dataloader.dataset),
            f"DualVAE_{'SCM' if use_scm else 'NoSCM'}/Orthogonality ZY-S": loss_dict["ortho_loss_zy_s"].item() / len(dataloader.dataset),
            f"DualVAE_{'SCM' if use_scm else 'NoSCM'}/Conditional Independence Loss": loss_dict["ci_loss"].item() / len(dataloader.dataset) if use_scm else 0.0,
            f"DualVAE_{'SCM' if use_scm else 'NoSCM'}/Inference ZX Loss": loss_dict["inference_zx_loss"].item() / len(dataloader.dataset)
        })

        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Run {run_id} ({'SCM' if use_scm else 'NoSCM'}): Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

    plt.figure(figsize=(10, 5))
    sns.set(style="whitegrid")
    plt.plot(range(1, num_epochs + 1), training_losses, label=f'DualVAE_{"SCM" if use_scm else "NoSCM"} Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'DualVAE Training Progress (SCM={use_scm})')
    plt.legend()
    plt.tight_layout()
    plot_path = f"DualVAE_Training_Run_{run_id}_{'SCM' if use_scm else 'NoSCM'}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    wandb.log({f"DualVAE/Training Progress ({'SCM' if use_scm else 'NoSCM'})": wandb.Image(plot_path)})

    return model

# --------------------------- Causal Index Computation --------------------------- #

def compute_causal_index(model, dataloader, causal_var_indices):
    model.eval()
    latent_s = []
    causal_data = []
    with torch.no_grad():
        for batch in dataloader:
            x_batch = batch[0].to(device)
            y_batch = batch[1].to(device)
            output = model(x_batch, y_batch, use_scm=True)
            s_mu = output.mean["s"].cpu().numpy()
            latent_s.append(s_mu)
            states = x_batch[:, causal_var_indices].cpu().numpy()
            causal_vars = states  # Assuming causal_var_indices are correctly selected
            causal_data.append(causal_vars)
    latent_s = np.concatenate(latent_s, axis=0)
    causal_data = np.concatenate(causal_data, axis=0)

    cca = CCA(n_components=min(latent_s.shape[1], causal_data.shape[1]))
    cca.fit(latent_s, causal_data)
    X_c, Y_c = cca.transform(latent_s, causal_data)
    correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0,1] for i in range(cca.n_components)]
    mean_correlation = np.mean(np.abs(correlations))
    causal_index = np.clip(mean_correlation, 0, 1)
    print(f"Causal Index: {causal_index:.4f}")
    return causal_index

# --------------------------- Performance Comparison Plot --------------------------- #

def plot_performance_comparison(rewards_no_scm, rewards_scm, run_id):
    plt.figure(figsize=(10,5))
    sns.set(style="whitegrid")
    plt.plot(rewards_no_scm, label='No SCM', color='red')
    plt.plot(rewards_scm, label='With SCM', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Comparison of PPO Performance: DualVAE with vs. without SCM')
    plt.legend()
    plt.tight_layout()
    plot_path = f"Comparison_SCM_vs_NoSCM_Performance_Run_{run_id}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    wandb.log({"Performance Comparison": wandb.Image(plot_path)})
    print(f"Performance Comparison plot saved: {plot_path}")

# --------------------------- Main Experiment Function --------------------------- #

def run_experiment(run_id, config, confounder_level, confounder_amplitudes, use_scm):
    wandb.init(project="SCM_DualVAE_PPO_Experiment",
               config=config,
               name=f"Run_{run_id}_Confounder_{confounder_level}_SCM_{use_scm}",
               reinit=True)
    
    # Create environment with specified confounder amplitudes
    base_env = NavigationEnv(num_games=1, max_steps=config["ppo_max_timesteps_causal"])  # Updated key
    env = ConfounderEnvWrapper(base_env,
                               confounder_amplitudes=confounder_amplitudes)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Phase 1: Data Collection for DualVAE
    num_data_steps = config["num_data_steps"]
    x_data = []
    y_data = []
    state = env.reset()
    for _ in tqdm(range(num_data_steps), desc=f"Run {run_id}: Data Collection for DualVAE", leave=False):
        # Random action to collect diverse data
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        x = np.concatenate([state, np.array([action])])
        y = np.concatenate([np.array([reward]), next_state])
        x_data.append(x)
        y_data.append(y)
        state = next_state
        if done:
            state = env.reset()
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Phase 2: Train DualVAE
    dualvae = DualVAEModel(state_dim=state_dim, action_dim=1, reward_dim=1, z_dim=4, s_dim=4, hidden_dims=[256, 256], batch_norm=False).to(device)
    dualvae_loss_weights = config["dualvae_loss_weights"]
    dataset = TensorDataset(torch.FloatTensor(x_data), torch.FloatTensor(y_data))
    dataloader = DataLoader(dataset, batch_size=config["dualvae_batch_size"], shuffle=True)
    dualvae = train_dualvae(dualvae, dualvae_loss_weights, dataset, device,
                            use_scm=use_scm,
                            num_epochs=config["dualvae_epochs"],
                            batch_size=config["dualvae_batch_size"],
                            learning_rate=config["dualvae_learning_rate"],
                            run_id=run_id)
    for param in dualvae.parameters():
        param.requires_grad = False

    # Phase 3: Compute Causal Index
    causal_index = compute_causal_index(dualvae, dataloader, config["causal_var_indices"])
    wandb.log({"Causal Index": causal_index})

    # Phase 4: PPO Training with DualVAE
    ppo_agent_causal = CausalPPOAgent(z_dim=4, action_dim=action_dim,
                                      lr=config["ppo_lr_causal"],
                                      gamma=config["ppo_gamma_causal"],
                                      K_epochs=config["ppo_K_epochs_causal"],
                                      eps_clip=config["ppo_eps_clip_causal"])
    max_episodes_causal = config["ppo_max_episodes_causal"]
    max_timesteps_causal = config["ppo_max_timesteps_causal"]
    update_timestep_causal = config["ppo_update_timestep_causal"]
    timestep_causal = 0
    memory_causal = {'z_x': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': [], 'state_values': []}
    rewards_causal = []

    for episode in tqdm(range(1, max_episodes_causal + 1), desc=f"Run {run_id}: PPO Training ({'With SCM' if use_scm else 'No SCM'})", leave=False):
        state = env.reset()
        ep_reward = 0
        for t in range(max_timesteps_causal):
            timestep_causal += 1
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            dualvae.eval()
            with torch.no_grad():
                z_x_mu = dualvae.infer_z_x(state_tensor)
                z_x = z_x_mu.cpu().numpy()[0]
            action, log_prob, state_value = ppo_agent_causal.select_action(z_x)
            next_state, reward, done, _ = env.step(action)
            memory_causal['z_x'].append(z_x)
            memory_causal['actions'].append(action)
            memory_causal['logprobs'].append(log_prob)
            memory_causal['state_values'].append(state_value)
            memory_causal['rewards'].append(reward)
            memory_causal['is_terminals'].append(done)
            ep_reward += reward
            state = next_state
            if timestep_causal % update_timestep_causal == 0:
                ppo_agent_causal.update_policy(memory_causal)
                memory_causal = {'z_x': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': [], 'state_values': []}
            if done:
                break
        rewards_causal.append(ep_reward)
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_causal[-100:])
            wandb.log({f"PPO/{'With SCM' if use_scm else 'No SCM'}/Average Reward (Last 100)": avg_reward})
            print(f"Run {run_id}: Episode {episode}, Average Reward (Last 100): {avg_reward:.2f}")

    # Phase 5: Logging and Plotting
    run_data = {
        f"PPO/{'With SCM' if use_scm else 'No SCM'}/Rewards": rewards_causal
    }
    wandb.log(run_data)

    # Note: Plotting is moved to the main function for correct comparison between runs

    # Phase 6: Compute and Log ATE (Average Treatment Effect)
    def estimate_ate(agent, env, dualvae_model, num_episodes=100, agent_type='causal_dualvae'):
        rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            ep_reward = 0
            done = False
            while not done:
                if agent_type == 'causal_dualvae':
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    dualvae_model.eval()
                    with torch.no_grad():
                        z_x_mu = dualvae_model.infer_z_x(state_tensor)
                        z_x = z_x_mu.cpu().numpy()[0]
                    action, log_prob, state_value = agent.select_action(z_x)
                else:
                    action, log_prob, state_value = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                state = next_state
            rewards.append(ep_reward)
        avg_reward = np.mean(rewards)
        return avg_reward

    num_episodes_ate = config.get("num_episodes_ate", 50)
    avg_reward_causal = estimate_ate(ppo_agent_causal, env, dualvae, num_episodes=num_episodes_ate, agent_type='causal_dualvae')
    wandb.log({
        f"PPO/{'With SCM' if use_scm else 'No SCM'}/Average Reward (ATE)": avg_reward_causal
    })
    print(f"Run {run_id}: {'With SCM' if use_scm else 'No SCM'}, Average Reward (ATE): {avg_reward_causal:.2f}")

    wandb.finish()

    return rewards_causal

# --------------------------- Main Function --------------------------- #

def main():
    config = {
        "num_data_steps": 10000,
        "dualvae_epochs": 50,
        "dualvae_batch_size": 128,
        "dualvae_learning_rate": 1e-3,
        "ppo_max_episodes_causal": 2000,
        "ppo_max_timesteps_causal": 200,
        "ppo_update_timestep_causal": 4000,
        "ppo_lr_causal": 3e-4,
        "ppo_gamma_causal": 0.99,
        "ppo_K_epochs_causal": 10,
        "ppo_eps_clip_causal": 0.2,
        "dualvae_loss_weights": [1.0, 1.0, 0.001, 0.1, 0.1, 0.1, 1.0],
        "causal_var_indices": [4, 5],  # posX and posY
        "moving_avg_window": 100,
        "num_episodes_ate": 50
    }

    # Low confounding scenario
    confounder_level = "Low"
    confounder_amplitudes = [0.1, 0.1, 0.15, 0.15]  # Low confounding amplitudes

    num_runs = 5
    rewards_runs_with_scm = []
    rewards_runs_no_scm = []

    for run_id in range(1, num_runs + 1):
        print(f"\n========== Starting Run {run_id} for Confounder Level: {confounder_level} ==========\n")
        
        # Run with SCM (Structural Equations)
        rewards_scm = run_experiment(run_id, config, confounder_level, confounder_amplitudes, use_scm=True)
        rewards_runs_with_scm.append(rewards_scm)
        
        # Run without SCM
        rewards_no_scm = run_experiment(run_id, config, confounder_level, confounder_amplitudes, use_scm=False)
        rewards_runs_no_scm.append(rewards_no_scm)
        
        print(f"========== Run {run_id} for Confounder Level: {confounder_level} Completed ==========\n")

    # Aggregate and Compare Results
    rewards_runs_with_scm = np.array(rewards_runs_with_scm)
    rewards_runs_no_scm = np.array(rewards_runs_no_scm)

    # Calculate mean and std for the last 100 episodes
    mean_reward_scm = np.mean(rewards_runs_with_scm[:, -100:])
    std_reward_scm = np.std(rewards_runs_with_scm[:, -100:])

    mean_reward_no_scm = np.mean(rewards_runs_no_scm[:, -100:])
    std_reward_no_scm = np.std(rewards_runs_no_scm[:, -100:])

    print(f"===== Performance Summary for Low Confounding =====")
    print(f"With SCM: Mean Reward (Last 100 Episodes) = {mean_reward_scm:.2f} ± {std_reward_scm:.2f}")
    print(f"No SCM: Mean Reward (Last 100 Episodes) = {mean_reward_no_scm:.2f} ± {std_reward_no_scm:.2f}")

    wandb.log({
        "Performance/With SCM Mean Reward": mean_reward_scm,
        "Performance/With SCM Std Reward": std_reward_scm,
        "Performance/No SCM Mean Reward": mean_reward_no_scm,
        "Performance/No SCM Std Reward": std_reward_no_scm
    })

    # Plot aggregated performance
    plt.figure(figsize=(10,5))
    sns.set(style="whitegrid")
    plt.boxplot([rewards_runs_no_scm.flatten(), rewards_runs_with_scm.flatten()], labels=['No SCM', 'With SCM'])
    plt.ylabel('Reward')
    plt.title('Performance Comparison: DualVAE with vs. without SCM (Low Confounding)')
    plt.tight_layout()
    plot_path = "Final_Performance_Comparison_Low_Confounding.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    wandb.log({"Final Performance Comparison": wandb.Image(plot_path)})
    print(f"Final Performance Comparison plot saved: {plot_path}")

    # Optionally, plot the mean rewards over runs
    episodes = np.arange(1, config["ppo_max_episodes_causal"] + 1)
    mean_rewards_no_scm = np.mean(rewards_runs_no_scm, axis=0)
    mean_rewards_scm = np.mean(rewards_runs_with_scm, axis=0)

    plt.figure(figsize=(10,5))
    sns.set(style="whitegrid")
    plt.plot(mean_rewards_no_scm, label='No SCM', color='red')
    plt.plot(mean_rewards_scm, label='With SCM', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Average PPO Performance: DualVAE with vs. without SCM (Low Confounding)')
    plt.legend()
    plt.tight_layout()
    plot_path_avg = "Average_PPO_Performance_Low_Confounding.png"
    plt.savefig(plot_path_avg, dpi=300, bbox_inches='tight')
    plt.close()
    wandb.log({"Average PPO Performance": wandb.Image(plot_path_avg)})
    print(f"Average PPO Performance plot saved: {plot_path_avg}")

    wandb.finish()
    print("All runs completed and results logged.")

# --------------------------- Execute Main Function --------------------------- #

if __name__ == "__main__":
    main()
