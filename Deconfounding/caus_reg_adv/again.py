# --------------------------- Import Necessary Libraries --------------------------- #
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cross_decomposition import CCA
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import gym
from gym import spaces
import os
import time
import wandb  # For experiment tracking

# --------------------------- Reproducibility and Device Configuration --------------------------- #

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
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
        self.action_space = spaces.Discrete(4)  # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)  # [color, shape, scale, orientation, posX, posY]
        self.max_steps = max_steps
        self.current_step = 0

        # Load dSprites dataset
        self.dataset_path = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"dSprites dataset not found at {self.dataset_path}. Please download it from https://github.com/deepmind/dsprites-dataset.")

        dataset = np.load(self.dataset_path, allow_pickle=True, encoding='latin1')
        self.imgs = dataset['imgs'].reshape(-1, 64, 64, 1).astype(np.float32)
        metadata = dataset['metadata'][()]
        self.latent_sizes = metadata['latents_sizes']  # [1, 3, 6, 40, 32, 32]
        self.latent_bases = np.concatenate(
            (metadata['latents_sizes'][::-1].cumprod()[::-1][1:], np.array([1]))
        )
        self.latent_names = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY']
        self.latent_dim = len(self.latent_names)  # 6 dimensions
        self.num_games = num_games

        # Initialize variables
        self.current_latents = np.zeros((self.num_games, self.latent_dim))
        self.reset_all_latents()

        print(f'Dataset loaded. Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, '
              f'Datapoints: {len(self.imgs)}, Latent Dimensions: {self.latent_dim}, '
              f'Latent Bases: {self.latent_bases}')

    def sample_latent(self):
        """
        Sample a random latent vector.
        """
        latent = np.zeros(self.latent_dim)
        latent[0] = 0  # Color is always 0 in dSprites dataset

        latent[1] = np.random.randint(0, 3)   # Shape: 0=square, 1=ellipse, 2=heart
        latent[2] = np.random.randint(0, 6)   # Scale: 6 values
        latent[3] = np.random.randint(0, 40)  # Orientation: 40 values
        latent[4] = np.random.randint(0, 32)  # posX: 32 values
        latent[5] = np.random.randint(0, 32)  # posY: 32 values
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

    def get_observation(self, index):
        """
        Retrieve observation (image) based on latent index.
        """
        idx = self.latents_to_index(self.current_latents[index])
        idx = np.clip(idx, 0, len(self.imgs) - 1)
        return self.imgs[idx].copy()

    def step(self, action):
        """
        Execute one action step in the environment.
        """
        index = 0  # Single game instance
        # Implement action logic affecting the latent variables (position)
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

        # Define enhanced reward logic
        target_pos = np.array([16, 16])  # Target position at center
        current_pos = self.current_latents[index, 4:6]
        prev_distance = np.linalg.norm(current_pos - target_pos)

        # Update position after action
        new_pos = self.current_latents[index, 4:6]
        new_distance = np.linalg.norm(new_pos - target_pos)

        # Calculate distance reduction
        distance_reduction = prev_distance - new_distance

        # Step penalty to encourage faster completion
        step_penalty = -0.1  # Adjust as needed

        # Boundary penalty
        boundary_penalty = 0.0
        if (action == 0 and self.current_latents[index, 5] >= 31) or \
           (action == 1 and self.current_latents[index, 5] <= 0) or \
           (action == 2 and self.current_latents[index, 4] <= 0) or \
           (action == 3 and self.current_latents[index, 4] >= 31):
            boundary_penalty = -0.5  # Penalty for invalid action

        # Total reward
        reward = distance_reduction + step_penalty + boundary_penalty

        # Additional reward for reaching the target
        done = False
        if new_distance == 0:
            reward += 10
            done = True
            self.current_latents[index] = self.sample_latent()
        else:
            done = False

        # Increment step counter
        self.current_step += 1

        # Check max steps
        if self.current_step >= self.max_steps:
            done = True

        # Get observation
        obs = self.current_latents[index].copy()

        return obs, reward, done, {}

    def reset(self):
        """
        Reset the environment to an initial state.
        """
        self.current_latents = self.sample_all_latents()
        self.current_step = 0  # Reset step counter
        state = self.current_latents[0].copy()
        return state

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
    def __init__(self, env, confounding_level='medium', noise_std=0.05):
        super(ConfounderEnvWrapper, self).__init__(env)
        self.confounding_level = confounding_level
        self.noise_std = noise_std
        self.time_step = 0

        # Define confounder amplitudes based on confounding level
        if confounding_level == 'low':
            self.confounder_amplitudes = [0.01, 0.005, 0.003, 0.002]
        elif confounding_level == 'medium':
            self.confounder_amplitudes = [0.05, 0.03, 0.02, 0.015]
        elif confounding_level == 'high':
            self.confounder_amplitudes = [0.07, 0.05, 0.03, 0.02]
        else:
            raise ValueError(f"Invalid confounding level: {confounding_level}")

        # Confounder frequencies remain the same
        self.confounder_freqs = [0.2, 0.15, 0.1, 0.25]

    def step(self, action):
        """
        Step with confounders affecting the environment.
        """
        confounder = sum(
            amp * np.sin(freq * self.time_step) + amp * np.cos(freq * self.time_step)
            for freq, amp in zip(self.confounder_freqs, self.confounder_amplitudes)
        )

        state, reward, done, info = self.env.step(action)
        modified_state = state + confounder + np.random.normal(0, self.noise_std, size=state.shape)
        modified_reward = reward + confounder * 0.1  # Simplified modification
        self.time_step += 1
        return modified_state, modified_reward, done, info

    def reset(self):
        """
        Reset the environment state with confounders reset.
        """
        self.time_step = 0
        state = self.env.reset()
        confounder = sum(
            amp * np.sin(freq * self.time_step) + amp * np.cos(freq * self.time_step)
            for freq, amp in zip(self.confounder_freqs, self.confounder_amplitudes)
        )
        modified_state = state + confounder + np.random.normal(0, self.noise_std, size=state.shape)
        return modified_state

# --------------------------- Utility Functions --------------------------- #

def normalize_magnitude(x, y):
    """
    Normalize two values based on their magnitudes.
    """
    denominator = x * x + y * y + 1e-7
    x_norm = x * x / denominator
    y_norm = y * y / denominator
    return x_norm, y_norm

def orthogonality_loss(x, y):
    """
    Orthogonality loss to ensure latent variables are orthogonal.
    """
    cosine_similarity = torch.nn.functional.cosine_similarity(x, y, dim=1)
    ortho_loss = torch.mean((cosine_similarity)**2)
    return ortho_loss

def kl_divergence_loss(mean, logvar):
    """
    Computes the KL divergence loss.
    """
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()

def zero_out_latents(input_dict, zero_vars=None):
    """
    Set specified latent variables in the input_dict to zero.
    """
    if zero_vars is None:
        return input_dict

    assert all(var_name in input_dict for var_name in zero_vars), "All zero_vars must exist in the input_dict."

    result_dict = {}
    for key, value in input_dict.items():
        if key in zero_vars:
            zero_tensor = torch.zeros_like(value, device=value.device)
            result_dict[key] = zero_tensor
        else:
            result_dict[key] = value

    return result_dict

# --------------------------- DualVAE Model Definitions --------------------------- #

class MLPNetwork(nn.Module):
    """
    Simple Multi-Layer Perceptron.
    """
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
    """
    Encoder part of the VAE.
    """
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

class Decoder(nn.Module):
    """
    Decoder part of the VAE.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256], batch_norm=False, activation=nn.ReLU()):
        super(Decoder, self).__init__()
        self.decoder = MLPNetwork(input_dim, output_dim, hidden_dims, batch_norm=batch_norm, activation=activation)

    def forward(self, z):
        return self.decoder(z)

class VAEOutput:
    """
    Container for VAE outputs.
    """
    def __init__(self, mean, logvar, sample, recon=None):
        self.mean = mean
        self.logvar = logvar
        self.sample = sample
        self.recon = recon

    def set_latent_zero(self, zero_vars):
        """
        Zero-out specified latent variables.
        """
        self.mean = zero_out_latents(self.mean, zero_vars)
        self.logvar = zero_out_latents(self.logvar, zero_vars)
        self.sample = zero_out_latents(self.sample, zero_vars)

    def clone(self):
        """
        Deep copy the VAEOutput.
        """
        if self.recon is None:
            return VAEOutput(
                {k: v.clone() for k, v in self.mean.items()},
                {k: v.clone() for k, v in self.logvar.items()},
                {k: v.clone() for k, v in self.sample.items()}
            )
        else:
            return VAEOutput(
                {k: v.clone() for k, v in self.mean.items()},
                {k: v.clone() for k, v in self.logvar.items()},
                {k: v.clone() for k, v in self.sample.items()},
                {k: v.clone() for k, v in self.recon.items()}
            )

class DualVAEModel(nn.Module):
    """
    DualVAE model that decomposes variables into shared and private latent representations.
    """
    def __init__(self, x_dim, y_dim, z_x_dim=4, z_y_dim=4, s_dim=4, hidden_dims=[256, 256], activation=nn.ReLU(), batch_norm=False):
        super(DualVAEModel, self).__init__()
        self.z_x_dim = z_x_dim
        self.z_y_dim = z_y_dim
        self.s_dim = s_dim

        # Encoder and decoder for X
        self.encoder_X = Encoder(x_dim, z_x_dim + s_dim, hidden_dims, batch_norm=batch_norm, activation=activation)
        self.decoder_X = Decoder(z_x_dim + s_dim, x_dim, hidden_dims, batch_norm=batch_norm, activation=activation)

        # Encoder and decoder for Y
        self.encoder_Y = Encoder(y_dim, z_y_dim + s_dim, hidden_dims, batch_norm=batch_norm, activation=activation)
        self.decoder_Y = Decoder(z_y_dim + s_dim, y_dim, hidden_dims, batch_norm=batch_norm, activation=activation)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar) + 1e-7
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y, zero_vars=None):
        """
        Forward pass through DualVAE.
        """
        # Encode X
        mu_X, logvar_X = self.encoder_X(x)
        z_x_mean, s_x_mean = torch.split(mu_X, [self.z_x_dim, self.s_dim], dim=1)
        z_x_logvar, s_x_logvar = torch.split(logvar_X, [self.z_x_dim, self.s_dim], dim=1)

        # Encode Y
        mu_Y, logvar_Y = self.encoder_Y(y)
        z_y_mean, s_y_mean = torch.split(mu_Y, [self.z_y_dim, self.s_dim], dim=1)
        z_y_logvar, s_y_logvar = torch.split(logvar_Y, [self.z_y_dim, self.s_dim], dim=1)

        # Mean and logvar dictionaries
        mean = {
            "z_x": z_x_mean,
            "z_y": z_y_mean,
            "s_x": s_x_mean,
            "s_y": s_y_mean
        }
        logvar = {
            "z_x": z_x_logvar,
            "z_y": z_y_logvar,
            "s_x": s_x_logvar,
            "s_y": s_y_logvar
        }

        # Reparameterize to get samples
        sample = {key: self.reparameterize(mean[key], logvar[key]) for key in mean.keys()}
        output = VAEOutput(mean, logvar, sample)

        # Zero out specified latents if needed (for ablation studies or controlled experiments)
        # Example: zero_vars = ['z_x', 'z_y']
        output.set_latent_zero(zero_vars)

        # Decode X
        recon_X = self.decoder_X(torch.cat([output.sample["z_x"], output.sample["s_x"]], dim=1))
        recon_X_shared = self.decoder_X(torch.cat([torch.zeros_like(output.sample["z_x"]), output.sample["s_x"]], dim=1))
        recon_X_zx = self.decoder_X(torch.cat([output.sample["z_x"], torch.zeros_like(output.sample["s_x"])], dim=1))

        # Decode Y
        recon_Y = self.decoder_Y(torch.cat([output.sample["z_y"], output.sample["s_y"]], dim=1))
        recon_Y_shared = self.decoder_Y(torch.cat([torch.zeros_like(output.sample["z_y"]), output.sample["s_y"]], dim=1))
        recon_Y_zy = self.decoder_Y(torch.cat([output.sample["z_y"], torch.zeros_like(output.sample["s_y"])], dim=1))

        # Store reconstructions
        output.recon = {
            "x": recon_X,
            "y": recon_Y,
            "x_shared": recon_X_shared,
            "x_zx": recon_X_zx,
            "y_shared": recon_Y_shared,
            "y_zy": recon_Y_zy
        }

        return output

# --------------------------- Causal Index Computation --------------------------- #

def compute_causal_index(model, dataloader):
    """
    Computes the causal index based on the latent representations from DualVAE.
    Utilizes Canonical Correlation Analysis (CCA) to measure the correlation between original data and latent variables.
    """
    model.eval()
    latent_reps = []
    original_data = []
    with torch.no_grad():
        for batch in dataloader:
            data = batch[0].to(device)
            output = model(data, data)  # Assuming y = x for simplicity
            # Extract shared latents: s_x
            s_x = output.sample["s_x"].cpu().numpy()
            latent_reps.append(s_x)
            original_data.append(data.cpu().numpy())
    latent_reps = np.concatenate(latent_reps, axis=0)
    original_data = np.concatenate(original_data, axis=0)

    # Perform CCA
    cca = CCA(n_components=1)
    cca.fit(original_data, latent_reps)
    X_c, Y_c = cca.transform(original_data, latent_reps)

    # Compute correlation
    correlation = np.corrcoef(X_c[:,0], Y_c[:,0])[0,1]
    correlation = np.clip(correlation, -1, 1)
    print(f"CCA Correlation (Causal Index): {correlation:.4f}")

    # Use the correlation as the causal index
    causal_index = correlation  # Scalar value
    return causal_index

# --------------------------- DualVAE Training and Evaluation Functions --------------------------- #

def dualvae_loss(x, y, output, weights):
    """
    Computes the DualVAE loss components.
    """
    # Reconstruction losses
    recon_X_loss = nn.functional.mse_loss(output.recon["x"], x, reduction='sum')
    recon_Y_loss = nn.functional.mse_loss(output.recon["y"], y, reduction='sum')
    recon_X_shared_loss = nn.functional.mse_loss(output.recon["x_shared"], x, reduction='sum')

    # Orthogonality losses
    ortho_loss_1 = orthogonality_loss(output.sample["z_x"], output.sample["z_y"])
    ortho_loss_2 = orthogonality_loss(output.sample["z_x"], output.sample["s_x"])
    ortho_loss_3 = orthogonality_loss(output.sample["z_y"], output.sample["s_y"])
    orthogonality_total = ortho_loss_1 + ortho_loss_2 + ortho_loss_3

    # Equivariance losses
    equiv_loss1 = nn.functional.mse_loss(output.sample["s_x"], output.sample["s_y"], reduction='sum')
    equiv_loss2 = nn.functional.mse_loss(output.recon["x_shared"], output.recon["y_shared"], reduction='sum')
    equiv_total = equiv_loss1 + equiv_loss2

    # KL Divergence and L1 Loss
    kld_total = 0
    l1_total = 0
    for key in output.mean.keys():
        kld_total += kl_divergence_loss(output.mean[key], output.logvar[key])
        l1_total += torch.mean(torch.abs(output.sample[key]))

    # Total loss with weights
    total_loss = (weights[0] * recon_X_loss +
                  weights[1] * recon_Y_loss +
                  weights[2] * recon_X_shared_loss +
                  weights[3] * orthogonality_total +
                  weights[4] * equiv_total +
                  weights[5] * kld_total +
                  weights[6] * l1_total)

    loss_dict = {
        "total_loss": total_loss,
        "recon_X_loss": recon_X_loss,
        "recon_Y_loss": recon_Y_loss,
        "recon_X_shared_loss": recon_X_shared_loss,
        "orthogonality_total": orthogonality_total,
        "equiv_total": equiv_total,
        "kld_total": kld_total,
        "l1_total": l1_total
    }
    return loss_dict

def train_dualvae(model, weights, x_data, y_data, device, num_epochs=100, batch_size=128, learning_rate=1e-3):
    """
    Trains the DualVAE model and logs metrics to wandb.
    """
    dataset = TensorDataset(x_data.float(), y_data.float())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # For plotting loss
    total_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        recon_x_total = 0
        recon_y_total = 0
        kld_total = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(x_batch, y_batch)
            loss_dict = dualvae_loss(x_batch, y_batch, output, weights)
            loss = loss_dict["total_loss"]
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            recon_x_total += loss_dict["recon_X_loss"].item()
            recon_y_total += loss_dict["recon_Y_loss"].item()
            kld_total += loss_dict["kld_total"].item()

        avg_loss = epoch_loss / len(dataloader.dataset)
        avg_recon_x = recon_x_total / len(dataloader.dataset)
        avg_recon_y = recon_y_total / len(dataloader.dataset)
        avg_kld = kld_total / len(dataloader.dataset)

        total_losses.append(avg_loss)

        # Log metrics to wandb
        wandb.log({
            "DualVAE/Epoch": epoch + 1,
            "DualVAE/Total Loss": avg_loss,
            "DualVAE/Reconstruction X Loss": avg_recon_x,
            "DualVAE/Reconstruction Y Loss": avg_recon_y,
            "DualVAE/KL Divergence": avg_kld
        })

        # Scheduler step
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Recon_X: {avg_recon_x:.6f}, Recon_Y: {avg_recon_y:.6f}, KL: {avg_kld:.6f}")

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), total_losses, label='DualVAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DualVAE Training Progress')
    plt.legend()
    plt.grid(True)
    # Save the plot to a file
    plot_filename = "dualvae_training_progress.png"
    plt.savefig(plot_filename)
    plt.close()
    # Log the plot to wandb
    wandb.log({"DualVAE/Training Progress": wandb.Image(plot_filename)})

    return model

def evaluate_dualvae(model, weights, x, y, device):
    """
    Evaluates the DualVAE model on given data.
    """
    model.eval().to(device)
    with torch.no_grad():
        output = model(x, y)
        loss_dict = dualvae_loss(x, y, output, weights)
        total_loss = loss_dict["total_loss"].item()
        recon_x_loss = loss_dict["recon_X_loss"].item()
        recon_y_loss = loss_dict["recon_Y_loss"].item()
    return output, total_loss, recon_x_loss, recon_y_loss

# --------------------------- PPO Agent Definitions --------------------------- #

# Baseline Actor-Critic Model for PPO
class BaselineActorCriticPPO(nn.Module):
    """
    Baseline Actor-Critic model for PPO, without using any latent variables.
    """
    def __init__(self, state_dim, action_dim):
        super(BaselineActorCriticPPO, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """
        Select action based on state.
        """
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist

    def evaluate(self, state, action):
        """
        Evaluate action log probabilities, state values, and entropy.
        """
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, state_value.squeeze(-1), entropy, dist

class BaselinePPOAgent:
    """
    Baseline PPO Agent without causal regularization.
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

    def select_action(self, state):
        """
        Select action using the baseline policy.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            action, log_prob, _ = self.policy_old.act(state)
        return action, log_prob.item()

    def update_policy(self, memory):
        """
        Update the PPO policy using the stored memory.
        """
        old_states = torch.FloatTensor(memory['states']).to(device)
        old_actions = torch.LongTensor(memory['actions']).to(device)
        old_logprobs = torch.FloatTensor(memory['logprobs']).to(device)
        rewards = memory['rewards']
        is_terminals = memory['is_terminals']

        # Compute returns and advantages
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.FloatTensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            logprobs, state_values, entropy, dist = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean() + \
                   0.5 * self.mse_loss(state_values, returns) - \
                   0.01 * entropy.mean()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

# Causal PPO Agent with Latent Variables
class CausalActorCriticPPO(nn.Module):
    """
    Actor-Critic model for PPO that incorporates shared latent variables directly.
    """
    def __init__(self, state_dim, latent_dim, action_dim):
        super(CausalActorCriticPPO, self).__init__()
        # State processing stream
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
        )
        # Shared latent variables processing
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
        )
        # Combined layers
        self.actor = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, latent):
        """
        Generate an action using the current policy.
        """
        state_encoded = self.state_encoder(state)
        latent_encoded = self.latent_encoder(latent)
        combined = torch.cat([state_encoded, latent_encoded], dim=-1)
        action_probs = self.actor(combined)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist

    def evaluate(self, state, latent, action):
        """
        Evaluate the action log probabilities, state values, and entropy for PPO training.
        """
        state_encoded = self.state_encoder(state)
        latent_encoded = self.latent_encoder(latent)
        combined = torch.cat([state_encoded, latent_encoded], dim=-1)
        action_probs = self.actor(combined)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        entropy = dist.entropy()
        state_value = self.critic(combined)
        return action_logprobs, state_value.squeeze(-1), entropy, dist

class CausalPPOAgent:
    """
    PPO Agent that incorporates shared latent variables directly into the policy network.
    """
    def __init__(self, state_dim, latent_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2):
        self.policy = CausalActorCriticPPO(state_dim, latent_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = CausalActorCriticPPO(state_dim, latent_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.mse_loss = nn.MSELoss()

    def select_action(self, state, latent):
        """
        Select action using the current policy.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        latent = torch.FloatTensor(latent).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob, _ = self.policy_old.act(state, latent)
        return action, log_prob.item()

    def update_policy(self, memory):
        """
        Update the PPO policy using the stored memory.
        """
        old_states = torch.FloatTensor(memory['states']).to(device)
        old_latents = torch.FloatTensor(memory['latents']).to(device)
        old_actions = torch.LongTensor(memory['actions']).to(device)
        old_logprobs = torch.FloatTensor(memory['logprobs']).to(device)
        rewards = memory['rewards']
        is_terminals = memory['is_terminals']

        # Compute returns and advantages
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.FloatTensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            logprobs, state_values, entropy, dist = self.policy.evaluate(old_states, old_latents, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # PPO Loss
            loss = -torch.min(surr1, surr2).mean() + \
                   0.5 * self.mse_loss(state_values, returns) - \
                   0.01 * entropy.mean()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

# Causal PPO Agent with Frontdoor Adjustment
class PPOActorCriticFrontdoor(nn.Module):
    """
    Actor-Critic network for PPO with Frontdoor Adjustment.
    """
    def __init__(self, state_dim, latent_dim, action_dim):
        super(PPOActorCriticFrontdoor, self).__init__()
        # State processing stream
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
        )
        # Mediator processing stream
        self.mediator_encoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
        )
        # Action embedding
        self.action_embedding = nn.Embedding(action_dim, 64)

        # Combined layers
        self.actor = nn.Sequential(
            nn.Linear(256 + 128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(256 + 128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, mediator, action_prev):
        """
        Generate an action using the current policy.
        """
        state_encoded = self.state_encoder(state)
        mediator_encoded = self.mediator_encoder(mediator)
        action_embedded = self.action_embedding(action_prev)
        combined = torch.cat([state_encoded, mediator_encoded, action_embedded], dim=-1)
        action_probs = self.actor(combined)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist

    def evaluate(self, state, mediator, action_prev, action):
        """
        Evaluate the action log probabilities, state values, and entropy for PPO training.
        """
        state_encoded = self.state_encoder(state)
        mediator_encoded = self.mediator_encoder(mediator)
        action_embedded = self.action_embedding(action_prev)
        combined = torch.cat([state_encoded, mediator_encoded, action_embedded], dim=-1)
        action_probs = self.actor(combined)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        entropy = dist.entropy()
        state_value = self.critic(combined)
        return action_logprobs, state_value.squeeze(-1), entropy, dist

class FrontdoorPPOAgent:
    """
    PPO Agent that incorporates the frontdoor adjustment to deconfound policy learning.
    """
    def __init__(self, state_dim, latent_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2):
        self.policy = PPOActorCriticFrontdoor(state_dim, latent_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = PPOActorCriticFrontdoor(state_dim, latent_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.mse_loss = nn.MSELoss()

    def select_action(self, state, mediator, action_prev):
        """
        Select action using the current policy.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mediator = torch.FloatTensor(mediator).unsqueeze(0).to(device)
        action_prev = torch.LongTensor([action_prev]).to(device)
        with torch.no_grad():
            action, log_prob, _ = self.policy_old.act(state, mediator, action_prev)
        return action, log_prob.item()

    def update_policy(self, memory):
        """
        Update the PPO policy using the stored memory.
        """
        old_states = torch.FloatTensor(memory['states']).to(device)
        old_mediators = torch.FloatTensor(memory['mediators']).to(device)
        old_actions_prev = torch.LongTensor(memory['actions_prev']).to(device)
        old_actions = torch.LongTensor(memory['actions']).to(device)
        old_logprobs = torch.FloatTensor(memory['logprobs']).to(device)
        rewards = memory['rewards']
        is_terminals = memory['is_terminals']

        # Compute returns and advantages
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.FloatTensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            logprobs, state_values, entropy, dist = self.policy.evaluate(old_states, old_mediators, old_actions_prev, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # PPO Loss
            loss = -torch.min(surr1, surr2).mean() + \
                   0.5 * self.mse_loss(state_values, returns) - \
                   0.01 * entropy.mean()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

# --------------------------- Main Training Loop with Baseline and Causal Comparison --------------------------- #

def main():
    # Initialize wandb project
    wandb.init(project="Frontdoor_PPO_Confounding_Levels_Three_Agents", config={
        "num_data_steps": 100000,
        "dualvae_epochs": 100,
        "dualvae_batch_size": 128,
        "dualvae_learning_rate": 1e-3,
        "ppo_max_episodes": 10000,
        "ppo_max_timesteps": 200,
        "ppo_update_timestep": 4000,
        "ppo_lr": 3e-4,
        "ppo_gamma": 0.99,
        "ppo_K_epochs": 10,
        "ppo_eps_clip": 0.2,
        "dualvae_loss_weights": [1.0, 1.0, 1.0, 0.1, 0.1, 0.001, 0.001],
        "confounding_levels": ["low", "medium", "high"],
    })

    # For storing results across confounding levels
    results = {}

    # Loop over different confounding levels
    for confounding_level in wandb.config.confounding_levels:
        print(f"\n{'='*20} Confounding Level: {confounding_level.upper()} {'='*20}\n")
        wandb.log({"Confounding Level": confounding_level})

        # Initialize environment with confounders
        base_env = NavigationEnv(num_games=1, max_steps=200)
        env = ConfounderEnvWrapper(base_env, confounding_level=confounding_level)
        state_dim = env.observation_space.shape[0]  # 6
        action_dim = env.action_space.n  # 4

        # Initialize DualVAE model
        dualvae = DualVAEModel(x_dim=state_dim, y_dim=state_dim, z_x_dim=4, z_y_dim=4, s_dim=4, hidden_dims=[256, 256], batch_norm=False).to(device)
        # Initialize weights for loss components: [recon_X, recon_Y, recon_X_shared, ortho, equiv, kld, l1]
        dualvae_loss_weights = wandb.config.dualvae_loss_weights

        # Data collection for DualVAE training
        print("\nCollecting data for DualVAE training...")
        num_data_steps = wandb.config.num_data_steps  # Number of steps for data collection
        states = []
        state = env.reset()
        for _ in tqdm(range(num_data_steps), desc="Data Collection"):
            action = env.action_space.sample()
            next_state, _, done, _ = env.step(action)
            states.append(state)
            state = next_state
            if done:
                state = env.reset()
        states = np.array(states)
        dataset = TensorDataset(torch.FloatTensor(states), torch.FloatTensor(states))
        dataloader = DataLoader(dataset, batch_size=wandb.config.dualvae_batch_size, shuffle=True)

        # Train DualVAE
        print("\nTraining DualVAE...")
        dualvae = train_dualvae(dualvae, dualvae_loss_weights, dataset.tensors[0], dataset.tensors[1], device,
                                num_epochs=wandb.config.dualvae_epochs,
                                batch_size=wandb.config.dualvae_batch_size,
                                learning_rate=wandb.config.dualvae_learning_rate)
        print("DualVAE training completed.\n")

        # Freeze DualVAE parameters to prevent updates during PPO training
        for param in dualvae.parameters():
            param.requires_grad = False

        # Compute causal index
        print("Computing causal index based on DualVAE latent representations...")
        causal_index = compute_causal_index(dualvae, dataloader)
        wandb.log({"Causal Index": causal_index, "Confounding Level": confounding_level})
        print(f"Causal Index: {causal_index:.4f}\n")

        # Initialize PPO agents
        ppo_agent_baseline = BaselinePPOAgent(state_dim, action_dim,
                                             lr=wandb.config.ppo_lr,
                                             gamma=wandb.config.ppo_gamma,
                                             K_epochs=wandb.config.ppo_K_epochs,
                                             eps_clip=wandb.config.ppo_eps_clip)  # Baseline PPO
        ppo_agent_causal = CausalPPOAgent(state_dim, latent_dim=4, action_dim=action_dim,
                                         lr=wandb.config.ppo_lr,
                                         gamma=wandb.config.ppo_gamma,
                                         K_epochs=wandb.config.ppo_K_epochs,
                                         eps_clip=wandb.config.ppo_eps_clip)  # Causal PPO with Latent Variables
        ppo_agent_frontdoor = FrontdoorPPOAgent(state_dim, latent_dim=4, action_dim=action_dim,
                                                lr=wandb.config.ppo_lr,
                                                gamma=wandb.config.ppo_gamma,
                                                K_epochs=wandb.config.ppo_K_epochs,
                                                eps_clip=wandb.config.ppo_eps_clip)  # Causal PPO with Frontdoor Adjustment

        # Training parameters
        max_episodes = wandb.config.ppo_max_episodes  # Number of episodes for training
        max_timesteps = wandb.config.ppo_max_timesteps    # Max timesteps per episode
        update_timestep = wandb.config.ppo_update_timestep  # Update PPO agent every 'update_timestep' timesteps
        timestep = 0

        # Initialize memory for PPO agents
        memory_baseline = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
        memory_causal = {'states': [], 'latents': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
        memory_frontdoor = {'states': [], 'mediators': [], 'actions_prev': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}

        # Initialize rewards tracking
        rewards_baseline = []
        rewards_causal = []
        rewards_frontdoor = []

        # --------------------------- Training Baseline PPO Agent --------------------------- #

        print("Starting training of Baseline PPO agent...")
        timestep = 0  # Reset timestep for each agent
        for episode in tqdm(range(1, max_episodes + 1), desc=f"Baseline PPO Training ({confounding_level})"):
            state = env.reset()
            ep_reward = 0
            for t in range(max_timesteps):
                timestep += 1
                # Select action using Baseline PPO agent
                action, log_prob = ppo_agent_baseline.select_action(state)
                next_state, reward, done, _ = env.step(action)
                # Save data in memory
                memory_baseline['states'].append(state)
                memory_baseline['actions'].append(action)
                memory_baseline['logprobs'].append(log_prob)
                memory_baseline['rewards'].append(reward)
                memory_baseline['is_terminals'].append(done)
                state = next_state
                ep_reward += reward
                # Update PPO agent
                if timestep % update_timestep == 0:
                    ppo_agent_baseline.update_policy(memory_baseline)
                    memory_baseline = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
                if done:
                    break
            rewards_baseline.append(ep_reward)
            # Logging every 50 episodes
            if episode % 50 == 0 or episode == 1:
                avg_reward = np.mean(rewards_baseline[-50:])
                wandb.log({f"Baseline PPO/Average Reward ({confounding_level})": avg_reward, "Episode": episode})
                print(f"Baseline PPO Agent - Episode {episode}, Average Reward (last 50): {avg_reward:.2f}")
        print("\nTraining of Baseline PPO agent completed.\n")

        # --------------------------- Training Causal PPO Agent with Latent Variables --------------------------- #

        print("Starting training of Causal PPO agent with Latent Variables...")
        timestep = 0  # Reset timestep for each agent
        for episode in tqdm(range(1, max_episodes + 1), desc=f"Causal PPO Training ({confounding_level})"):
            state = env.reset()
            ep_reward = 0
            for t in range(max_timesteps):
                timestep += 1
                # Obtain shared latent representation from DualVAE
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                dualvae.eval()
                with torch.no_grad():
                    output = dualvae(state_tensor, state_tensor)  # Assuming y = x for simplicity
                    latent = output.sample["s_x"].cpu().numpy()[0]  # 4-dimensional shared latent

                # Select action using Causal PPO agent
                action, log_prob = ppo_agent_causal.select_action(state, latent)
                next_state, reward, done, _ = env.step(action)
                # Save data in memory
                memory_causal['states'].append(state)
                memory_causal['latents'].append(latent)
                memory_causal['actions'].append(action)
                memory_causal['logprobs'].append(log_prob)
                memory_causal['rewards'].append(reward)
                memory_causal['is_terminals'].append(done)
                state = next_state
                ep_reward += reward
                # Update PPO agent
                if timestep % update_timestep == 0:
                    ppo_agent_causal.update_policy(memory_causal)
                    memory_causal = {'states': [], 'latents': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
                if done:
                    break
            rewards_causal.append(ep_reward)
            # Logging every 50 episodes
            if episode % 50 == 0 or episode == 1:
                avg_reward = np.mean(rewards_causal[-50:])
                wandb.log({f"Causal PPO/Average Reward ({confounding_level})": avg_reward, "Episode": episode})
                print(f"Causal PPO Agent - Episode {episode}, Average Reward (last 50): {avg_reward:.2f}")
        print("\nTraining of Causal PPO agent completed.\n")

        # --------------------------- Training Causal PPO Agent with Frontdoor Adjustment --------------------------- #

        print("Starting training of Causal PPO agent with Frontdoor Adjustment...")
        timestep = 0  # Reset timestep for each agent
        for episode in tqdm(range(1, max_episodes + 1), desc=f"Frontdoor PPO Training ({confounding_level})"):
            state = env.reset()
            action_prev = 0  # Initialize previous action
            ep_reward = 0
            for t in range(max_timesteps):
                timestep += 1
                # Obtain mediator representation from DualVAE (shared latents)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                dualvae.eval()
                with torch.no_grad():
                    output = dualvae(state_tensor, state_tensor)  # Assuming y = x for simplicity
                    mediator = output.sample["s_x"].cpu().numpy()[0]  # 4-dimensional mediator

                # Select action using Frontdoor PPO agent
                action, log_prob = ppo_agent_frontdoor.select_action(state, mediator, action_prev)
                next_state, reward, done, _ = env.step(action)
                # Save data in memory
                memory_frontdoor['states'].append(state)
                memory_frontdoor['mediators'].append(mediator)
                memory_frontdoor['actions_prev'].append(action_prev)
                memory_frontdoor['actions'].append(action)
                memory_frontdoor['logprobs'].append(log_prob)
                memory_frontdoor['rewards'].append(reward)
                memory_frontdoor['is_terminals'].append(done)
                action_prev = action  # Update previous action
                state = next_state
                ep_reward += reward
                # Update PPO agent
                if timestep % update_timestep == 0:
                    ppo_agent_frontdoor.update_policy(memory_frontdoor)
                    memory_frontdoor = {'states': [], 'mediators': [], 'actions_prev': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
                if done:
                    break
            rewards_frontdoor.append(ep_reward)
            # Logging every 50 episodes
            if episode % 50 == 0 or episode == 1:
                avg_reward = np.mean(rewards_frontdoor[-50:])
                wandb.log({f"Frontdoor PPO/Average Reward ({confounding_level})": avg_reward, "Episode": episode})
                print(f"Frontdoor PPO Agent - Episode {episode}, Average Reward (last 50): {avg_reward:.2f}")
        print("\nTraining of Frontdoor PPO agent completed.\n")

        # Store results for this confounding level
        results[confounding_level] = {
            'rewards_baseline': rewards_baseline,
            'rewards_causal': rewards_causal,
            'rewards_frontdoor': rewards_frontdoor
        }

    # --------------------------- Performance Visualization Across Confounding Levels --------------------------- #

    # Plotting Total Rewards for Each Confounding Level
    for confounding_level in wandb.config.confounding_levels:
        rewards_baseline = results[confounding_level]['rewards_baseline']
        rewards_causal = results[confounding_level]['rewards_causal']
        rewards_frontdoor = results[confounding_level]['rewards_frontdoor']

        plt.figure(figsize=(12, 6))
        plt.plot(rewards_baseline, label=f'Baseline PPO ({confounding_level.capitalize()})')
        plt.plot(rewards_causal, label=f'Causal PPO ({confounding_level.capitalize()})')
        plt.plot(rewards_frontdoor, label=f'Frontdoor PPO ({confounding_level.capitalize()})')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'PPO Performance Comparison at {confounding_level.capitalize()} Confounding Level')
        plt.legend()
        plt.grid(True)
        # Save the plot to a file
        plot_filename = f"ppo_performance_comparison_{confounding_level}.png"
        plt.savefig(plot_filename)
        plt.close()
        # Log the plot to wandb
        wandb.log({f"PPO/Performance Comparison ({confounding_level})": wandb.Image(plot_filename)})

        # Plotting Moving Averages
        window_size = 50
        if len(rewards_baseline) >= window_size and len(rewards_causal) >= window_size and len(rewards_frontdoor) >= window_size:
            moving_avg_baseline = np.convolve(rewards_baseline, np.ones(window_size)/window_size, mode='valid')
            moving_avg_causal = np.convolve(rewards_causal, np.ones(window_size)/window_size, mode='valid')
            moving_avg_frontdoor = np.convolve(rewards_frontdoor, np.ones(window_size)/window_size, mode='valid')

            plt.figure(figsize=(12, 6))
            plt.plot(moving_avg_baseline, label=f'Baseline PPO (Moving Avg {window_size})')
            plt.plot(moving_avg_causal, label=f'Causal PPO (Moving Avg {window_size})')
            plt.plot(moving_avg_frontdoor, label=f'Frontdoor PPO (Moving Avg {window_size})')
            plt.xlabel('Episode')
            plt.ylabel('Moving Average Total Reward')
            plt.title(f'PPO Performance Moving Average at {confounding_level.capitalize()} Confounding Level')
            plt.legend()
            plt.grid(True)
            # Save the plot to a file
            plot_filename = f"ppo_moving_average_comparison_{confounding_level}.png"
            plt.savefig(plot_filename)
            plt.close()
            # Log the plot to wandb
            wandb.log({f"PPO/Moving Average Comparison ({confounding_level})": wandb.Image(plot_filename)})
        else:
            print("Not enough episodes to compute moving average.")

    # --------------------------- Finish Training --------------------------- #

    print("Training completed successfully across all confounding levels.")
    wandb.finish()

# --------------------------- Execution Entry Point --------------------------- #

if __name__ == '__main__':
    main()