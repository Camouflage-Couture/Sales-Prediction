# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gym
from gym import spaces
import os
import matplotlib.pyplot as plt
import datetime
import time

# --------------------------- Seed Setting for Reproducibility --------------------------- #
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(0)

# --------------------------- Device Detection --------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# --------------------------- Custom Gym Environment Definition --------------------------- #

class NavigationEnv(gym.Env):
    """
    Custom Navigation Environment using dSprites dataset.
    The agent's goal is to navigate to a target position while dealing with confounders.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_games=1, max_steps=100, grid_size=32):
        super(NavigationEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        # Match dSprites original dimensions
        self.latent_sizes = [1, 3, 6, 40, 32, 32]  # Original dSprites dimensions
        self.grid_size = grid_size  # Store grid_size as instance variable
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(6,),  # [color, shape, scale, orientation, posX, posY]
            dtype=np.float32
        )
        self.max_steps = max_steps
        self.current_step = 0

        # Load dSprites dataset
        self.dataset_path = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"dSprites dataset not found at {self.dataset_path}. Please download it from https://github.com/deepmind/dsprites-dataset.")

        dataset = np.load(self.dataset_path, allow_pickle=True, encoding='latin1')
        self.imgs = dataset['imgs'].reshape(-1, 64, 64, 1).astype(np.float32)
        metadata = dataset['metadata'][()]
        self.latent_sizes = list(metadata['latents_sizes'])  # Convert to list for mutability
        self.latent_names = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY']
        self.latent_dim = len(self.latent_names)  # 6 dimensions
        self.num_games = num_games

        # Align posX and posY latent sizes with grid_size
        original_posX_size = self.latent_sizes[4]
        original_posY_size = self.latent_sizes[5]
        self.latent_sizes[4] = self.grid_size  # posX
        self.latent_sizes[5] = self.grid_size  # posY

        # Initialize variables
        self.current_latents = np.zeros((self.num_games, self.latent_dim), dtype=int)
        self.reset_all_latents()

        print(f'Dataset loaded. Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, '
              f'Datapoints: {len(self.imgs)}, Latent Dimensions: {self.latent_dim}, '
              f'Latent Sizes: {self.latent_sizes}')

    def sample_latent(self):
        """
        Sample a random latent vector.
        """
        latent = np.zeros(self.latent_dim, dtype=int)
        latent[0] = 0  # Color is always 0 in dSprites dataset

        latent[1] = np.random.randint(0, self.latent_sizes[1])   # Shape: 0=square, 1=ellipse, 2=heart
        latent[2] = np.random.randint(0, self.latent_sizes[2])   # Scale: 6 values
        latent[3] = np.random.randint(0, self.latent_sizes[3])   # Orientation: 40 values
        latent[4] = np.random.randint(0, self.latent_sizes[4])   # posX: grid_size values (0-15)
        latent[5] = np.random.randint(0, self.latent_sizes[5])   # posY: grid_size values (0-15)
        return latent

    def sample_all_latents(self):
        """
        Sample latents for all game instances.
        """
        latents = np.zeros((self.num_games, self.latent_dim), dtype=int)
        for idx in range(self.num_games):
            latents[idx] = self.sample_latent()
        return latents

    def latents_to_index(self, latents):
        """
        Convert latent vectors to state indices based only on posX and posY.
        """
        posX = latents[4]
        posY = latents[5]
        return posX * self.grid_size + posY

    def get_observation(self, index):
        """Return full latent state vector normalized to [0,1]."""
        normalized_latents = np.zeros(6, dtype=np.float32)
        for i in range(6):
            normalized_latents[i] = self.current_latents[index, i] / (self.latent_sizes[i] - 1)
        return normalized_latents

    def step(self, action):
        index = 0  # Single game instance
        # Store previous position
        prev_pos = self.current_latents[index, 4:6].copy()
        
        # Update position based on action
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

        # Simple reward structure
        target_pos = np.array([self.latent_sizes[4] // 2, self.latent_sizes[5] // 2])
        current_pos = self.current_latents[index, 4:6]
        
        # Basic distance-based reward
        reward = -np.linalg.norm(current_pos - target_pos) / (self.latent_sizes[4] + self.latent_sizes[5])
        
        # Terminal condition
        done = np.array_equal(current_pos, target_pos)
        if done:
            reward = 1.0  # Fixed reward for reaching target
        
        # Max steps condition
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        
        return self.get_observation(index), reward, done, {}

    def reset(self):
        """
        Reset the environment to an initial state.
        """
        self.current_latents = self.sample_all_latents()
        self.current_step = 0  # Reset step counter
        s = self.latents_to_index(self.current_latents[0])
        s = int(s)  # Ensure s is a Python int
        return s

    def render(self, mode='human'):
        """
        Render the environment. (Optional)
        """
        # For simplicity, we'll skip rendering. Implement as needed.
        pass

    def reset_all_latents(self):
        """
        Reset all latents in the environment.
        """
        self.current_latents = self.sample_all_latents()
        self.current_step = 0

# --------------------------- Confounded Environment Wrapper Definition --------------------------- #

class ConfounderEnvWrapper(gym.Wrapper):
    """
    Gym environment wrapper that adds hidden confounders influencing both state transitions and rewards.
    """
    def __init__(self, env, confounder_freqs=None, confounder_amplitudes=None, noise_std=0.05):
        super(ConfounderEnvWrapper, self).__init__(env)
        self.confounder_freqs = confounder_freqs if confounder_freqs is not None else [0.2, 0.15, 0.1, 0.25]
        self.confounder_amplitudes = confounder_amplitudes if confounder_amplitudes is not None else [0.05, 0.03, 0.02, 0.015]
        self.noise_std = noise_std
        self.time_step = 0

    def step(self, action):
        """
        Step with confounders affecting the environment.
        """
        # Get the original state, reward, done, info
        state, reward, done, info = self.env.step(action)
        
        # Apply confounders and noise to the state
        # Since state is a discrete index, we need to map it back to latents
        latents = self.env.current_latents[0].copy()
        
        # Add confounders and noise to latents
        confounder = sum(
            amp * np.sin(freq * self.time_step) + amp * np.cos(freq * self.time_step)
            for freq, amp in zip(self.confounder_freqs, self.confounder_amplitudes)
        )
        # Add noise
        latents = latents + confounder + np.random.normal(0, self.noise_std, size=latents.shape)
        # Round and clamp to ensure latents remain within valid ranges
        latents = np.round(latents).astype(int)
        latents[0] = 0  # Color is always 0 in dSprites
        
        # Clamp each latent dimension (only posX and posY are used for state index)
        latents[1] = np.clip(latents[1], 0, self.env.latent_sizes[1] - 1)  # Shape
        latents[2] = np.clip(latents[2], 0, self.env.latent_sizes[2] - 1)  # Scale
        latents[3] = np.clip(latents[3], 0, self.env.latent_sizes[3] - 1)  # Orientation
        latents[4] = np.clip(latents[4], 0, self.env.latent_sizes[4] - 1)  # posX
        latents[5] = np.clip(latents[5], 0, self.env.latent_sizes[5] - 1)  # posY
        
        # Update the environment's latents
        self.env.current_latents[0] = latents
        
        # Map back to state index
        s = self.env.latents_to_index(latents)
        s = int(s)  # Ensure s is a Python int
        
        # Modify reward if needed
        modified_reward = reward  # Adjust based on confounders if necessary
        
        # Increment time step
        self.time_step += 1
        
        return s, modified_reward, done, info

    def reset(self):
        """
        Reset the environment state with confounders reset.
        """
        self.time_step = 0
        state = self.env.reset()
        return state

# --------------------------- Define Policies --------------------------- #

class Policy:
    def reset(self):
        pass

    def action(self, o, r, d, **info):
        raise NotImplementedError

class UniformPolicy(Policy):
    """
    Interventional Policy: Selects actions uniformly at random.
    """
    def __init__(self, a_nvals):
        super().__init__()
        self.a_nvals = a_nvals

    def action(self, o, r, d, **info):
        return random.randint(0, self.a_nvals - 1)

class ExpertPolicy(Policy):
    """
    Observational Policy: Chooses actions based on the true state.
    """
    def __init__(self, target_pos, grid_size):
        super().__init__()
        self.target_pos = target_pos
        self.grid_size = grid_size

    def action(self, o, r, d, **info):
        # Retrieve true position from state index
        s = o  # 'o' is the state index
        posX = s // self.grid_size
        posY = s % self.grid_size
        current_pos = np.array([posX, posY])

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
            return random.randint(0, 3)  # Random action if already at target

        return random.choice(actions)

# --------------------------- Define the Tabular Augmented Model --------------------------- #

class TabularAugmentedModel(nn.Module):
    def __init__(self, s_nvals, a_nvals, o_nvals, r_nvals):
        super(TabularAugmentedModel, self).__init__()
        self.s_nvals = s_nvals
        self.a_nvals = a_nvals
        self.o_nvals = o_nvals
        self.r_nvals = r_nvals

        # Initialize parameters (logits)
        self.params_s = nn.Parameter(torch.randn(s_nvals))  # Initial state distribution
        self.params_s_sa = nn.Parameter(torch.randn(s_nvals, a_nvals, s_nvals))  # Transition probabilities
        self.params_o_s = nn.Parameter(torch.randn(s_nvals, o_nvals))  # Observation probabilities
        self.params_r_s = nn.Parameter(torch.randn(s_nvals, r_nvals))  # Reward probabilities
        self.params_a_s = nn.Parameter(torch.randn(s_nvals, a_nvals))  # Action probabilities

    def log_q_s(self):
        """
        Log probability of initial state.
        """
        return F.log_softmax(self.params_s, dim=-1)

    def log_q_snext_sa(self):
        """
        Log probability of next state given current state and action.
        """
        return F.log_softmax(self.params_s_sa, dim=-1)

    def log_q_o_s(self):
        """
        Log probability of observation given state.
        """
        return F.log_softmax(self.params_o_s, dim=-1)

    def log_q_r_s(self):
        """
        Log probability of reward given state.
        """
        return F.log_softmax(self.params_r_s, dim=-1)

    def log_q_a_s(self):
        """
        Log probability of action given state.
        """
        return F.log_softmax(self.params_a_s, dim=-1)

    def loss_nll(self, regime, episodes):
        """
        Negative Log-Likelihood loss.
        regime: Tensor indicating regime (0=observational, 1=interventional)
        episodes: List of episodes, where each episode is a list containing states, rewards, dones, and actions
                  Format: [s0, r0, d0, a0, s1, r1, d1, a1, ..., sn, rn, dn, an]
        """
        device = self.params_s.device  # Get device from model parameters
        loss_terms = []
        batch_size = len(episodes)

        for i in range(batch_size):
            episode = episodes[i]
            regime_i = regime[i]
            # Unpack episode into sequences
            num_transitions = (len(episode) - 1) // 4  # Each transition has s, r, d, a

            for t in range(num_transitions):
                s = episode[4 * t]
                r = episode[4 * t + 1]
                d = episode[4 * t + 2]
                a = episode[4 * t + 3]

                # Map reward to category
                r_cat = 1 if r > 0 else 0

                # Validate indices
                if s >= self.s_nvals or a >= self.a_nvals or r_cat >= self.r_nvals:
                    continue  # Skip invalid indices

                if not d:
                    s_next = episode[4 * t + 4]
                    if s_next >= self.s_nvals:
                        continue  # Skip invalid s_next
                    log_p_snext_sa = self.log_q_snext_sa()[s, a, s_next]
                else:
                    log_p_snext_sa = torch.tensor(0.0, device=device)

                # Log probabilities
                log_p_a_s = self.log_q_a_s()[s, a]
                log_p_r_s = self.log_q_r_s()[s, r_cat]

                # Since observation o = s, log_p_o_s = 0 (deterministic)
                log_p_o_s = torch.tensor(0.0, device=device)

                # Sum log probabilities
                log_prob = log_p_a_s + log_p_r_s + log_p_snext_sa + log_p_o_s

                # Accumulate negative log likelihood
                loss_terms.append(-log_prob)

        # Sum all loss terms
        if loss_terms:
            # Stack all loss terms into a single tensor
            total_loss = torch.stack(loss_terms).sum() / batch_size
        else:
            # Create a zero tensor that requires grad to prevent issues during backprop
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss

# --------------------------- Training the Augmented Model --------------------------- #

def train_model(model, train_data, valid_data, n_epochs=5, batch_size=16, lr=5e-2):
    """
    Train the augmented model using Negative Log-Likelihood loss.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    valid_losses = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0
        random.shuffle(train_data)
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            regime, episodes = zip(*batch)
            regime = torch.tensor(regime, dtype=torch.float32, device=device)
            loss = model.loss_nll(regime, episodes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / (len(train_data) / batch_size)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for i in range(0, len(valid_data), batch_size):
                batch = valid_data[i:i + batch_size]
                regime, episodes = zip(*batch)
                regime = torch.tensor(regime, dtype=torch.float32, device=device)
                loss = model.loss_nll(regime, episodes)
                total_valid_loss += loss.item()
        avg_valid_loss = total_valid_loss / (len(valid_data) / batch_size)
        valid_losses.append(avg_valid_loss)

        print(f'Epoch {epoch}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}')

    return train_losses, valid_losses

# --------------------------- Define the Actor-Critic Agent --------------------------- #

class ActorCritic(nn.Module):
    def __init__(self, s_nvals, a_nvals, hidden_size=128):
        super(ActorCritic, self).__init__()
        self.s_nvals = s_nvals
        self.a_nvals = a_nvals

        self.actor = nn.Sequential(
            nn.Linear(s_nvals, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, a_nvals),
            nn.LogSoftmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(s_nvals, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        """
        Forward pass through the network.
        """
        log_probs = self.actor(state)
        value = self.critic(state)
        return log_probs, value

def run_actor_critic(env, agent, n_epochs=1000, max_steps_per_episode=100, gamma=0.99, lr=1e-3):
    """
    Train the Actor-Critic agent.
    """
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    reward_history = []
    for epoch in range(1, n_epochs + 1):
        state = env.reset()
        done = False
        total_reward = 0
        log_probs = []
        values = []
        rewards = []
        for t in range(max_steps_per_episode):
            # One-hot encode the state
            state_tensor = F.one_hot(torch.tensor(state, dtype=torch.long), num_classes=agent.s_nvals).float().unsqueeze(0).to(device)
            log_prob, value = agent(state_tensor)
            action_probs = torch.exp(log_prob)
            action = torch.multinomial(action_probs, num_samples=1).item()

            next_state, reward, done, info = env.step(action)
            total_reward += reward

            log_probs.append(log_prob[0, action])
            values.append(value.squeeze(0))
            rewards.append(reward)

            if done:
                break
            state = next_state

        # Compute returns and advantages
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        advantage = returns - values.detach()

        # Compute loss
        actor_loss = - (log_probs * advantage).mean()
        critic_loss = F.mse_loss(values.squeeze(-1), returns)
        loss = actor_loss + critic_loss

        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        reward_history.append(total_reward)

        if epoch % 100 == 0 or epoch == 1:
            avg_reward = np.mean(reward_history[-100:])
            print(f'Epoch {epoch}/{n_epochs}, Total Reward: {total_reward:.2f}, Avg Reward (last 100): {avg_reward:.2f}')

    return reward_history

# --------------------------- Define Utility Functions --------------------------- #

def construct_dataset(env, policy, n_samples, regime):
    """
    Collect episodes using the given policy.
    Each episode is a list: [s0, r0, d0, a0, s1, r1, d1, a1, ..., sn, rn, dn, an]
    """
    data = []
    for _ in range(n_samples):
        policy.reset()
        episode = []
        state = env.reset()
        done = False
        while not done:
            action = policy.action(state, None, None, s=state)
            next_state, reward, done, info = env.step(action)
            # Map reward to category
            r_cat = 1 if reward > 0 else 0
            # Append to episode: s, r_cat, d, a
            episode.extend([state, r_cat, done, action])
            state = next_state
        data.append((regime, episode))
    return data

# --------------------------- Main Execution --------------------------- #

def main():
    # Define parameters
    grid_size = 32  # Reduced from 32 to 16 for testing
    s_nvals = grid_size * grid_size  # 256
    a_nvals = 4
    o_nvals = s_nvals  # Observations are state indices
    r_nvals = 2  # Reward can be 0 or 1

    # Create the environment
    env = NavigationEnv(num_games=1, max_steps=100, grid_size=grid_size)
    confounded_env = ConfounderEnvWrapper(env)

    # Define policies
    target_pos = np.array([grid_size // 2, grid_size // 2])
    expert_policy = ExpertPolicy(target_pos=target_pos, grid_size=grid_size)
    uniform_policy = UniformPolicy(a_nvals=a_nvals)

    # Generate datasets
    print("Collecting observational data...")
    obs_data = construct_dataset(confounded_env, expert_policy, n_samples=1000, regime=0.0)
    print("Collecting interventional data...")
    int_data = construct_dataset(confounded_env, uniform_policy, n_samples=1000, regime=1.0)
    train_data = obs_data + int_data
    random.shuffle(train_data)
    valid_data = train_data  # For simplicity, use the same data as validation

    # Initialize the augmented model
    model = TabularAugmentedModel(s_nvals, a_nvals, o_nvals, r_nvals).to(device)

    # Train the model
    print("Training the augmented model...")
    train_losses, valid_losses = train_model(model, train_data, valid_data, n_epochs=10, batch_size=16, lr=1e-2)

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Augmented Model Training Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Initialize the Actor-Critic agent
    agent = ActorCritic(s_nvals, a_nvals).to(device)

    # Train the agent for longer (1000 epochs)
    print("Training the RL agent...")
    reward_history = run_actor_critic(confounded_env, agent, n_epochs=10000, max_steps_per_episode=100, gamma=0.99, lr=1e-3)

    # Plot reward history
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(reward_history) + 1), reward_history)
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.title('Actor-Critic Training Rewards')
    plt.grid(True)
    plt.show()

    # Evaluate the agent
    print("Evaluating the RL agent...")
    total_rewards = []
    for episode_num in range(1, 101):
        state = confounded_env.reset()
        done = False
        total_reward = 0
        while not done:
            # One-hot encode the state with correct dtype
            state_tensor = F.one_hot(torch.tensor(state, dtype=torch.long), num_classes=s_nvals).float().unsqueeze(0).to(device)
            log_prob, value = agent(state_tensor)
            action_probs = torch.exp(log_prob)
            action = torch.multinomial(action_probs, num_samples=1).item()
            next_state, reward, done, info = confounded_env.step(action)
            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
        if episode_num % 10 == 0:
            print(f'Episode {episode_num}: Total Reward: {total_reward:.2f}')
    average_reward = np.mean(total_rewards)
    print(f'\nAverage Reward over 100 episodes: {average_reward:.2f}')

    # Plot evaluation rewards
    plt.figure(figsize=(10, 5))
    plt.hist(total_rewards, bins=20, edgecolor='k', alpha=0.7)
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Rewards over 100 Evaluation Episodes')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
