# --------------------------- Importing Necessary Libraries --------------------------- #

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cross_decomposition import CCA as sklearn_CCA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gym
from gym import spaces
import os
import time
import wandb  # For experiment tracking
from scipy.stats import pearsonr, ttest_ind, kendalltau, spearmanr
from scipy.signal import correlate
from sklearn.metrics import normalized_mutual_info_score, roc_curve, auc, precision_recall_curve
import pandas as pd

# --------------------------- Reproducibility and Device Configuration --------------------------- #

torch.manual_seed(42)
np.random.seed(42)

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
    The confounder is a hidden variable that changes according to a Markov process.
    """
    def __init__(self, env, confounder_states=None, confounder_transition_matrix=None):
        super(ConfounderEnvWrapper, self).__init__(env)
        # Define confounder states (hidden modes)
        self.confounder_states = confounder_states if confounder_states is not None else [
            {"state_multiplier": np.ones(env.observation_space.shape, dtype=np.float32), "reward_bias": 0.0},
            {"state_multiplier": np.full(env.observation_space.shape, 0.95, dtype=np.float32), "reward_bias": -0.2},
            {"state_multiplier": np.full(env.observation_space.shape, 1.05, dtype=np.float32), "reward_bias": 0.2},
        ]
        self.num_states = len(self.confounder_states)
        # Define transition probabilities between confounder states
        self.confounder_transition_matrix = confounder_transition_matrix if confounder_transition_matrix is not None else np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ])
        # Ensure the transition matrix is valid
        assert self.confounder_transition_matrix.shape == (self.num_states, self.num_states), "Transition matrix shape mismatch."
        assert np.allclose(self.confounder_transition_matrix.sum(axis=1), 1), "Rows of transition matrix must sum to 1."
        # Initialize the current confounder state
        self.current_state = np.random.choice(self.num_states)

    def step(self, action):
        """
        Step with confounders affecting the environment.
        """
        # Transition to next confounder state based on transition probabilities
        self.current_state = np.random.choice(
            self.num_states,
            p=self.confounder_transition_matrix[self.current_state]
        )
        confounder = self.confounder_states[self.current_state]

        state, reward, done, info = self.env.step(action)
        # Apply confounder effect on state and reward
        modified_state = state * confounder["state_multiplier"]
        modified_reward = reward + confounder["reward_bias"]
        return modified_state, modified_reward, done, info

    def reset(self):
        """
        Reset the environment state with confounders reset.
        """
        # Randomly choose a starting confounder state
        self.current_state = np.random.choice(self.num_states)
        confounder = self.confounder_states[self.current_state]
        state = self.env.reset()
        modified_state = state * confounder["state_multiplier"]
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
    if x.size(1) != y.size(1):
        raise ValueError(f"Dimension mismatch: x has dim {x.size(1)}, y has dim {y.size(1)}")
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

# --------------------------- Additional Utility Functions --------------------------- #

def Normalize(x, y):
    """
    Alternative normalization function.
    """
    x0 = x * x / (x * x + y * y)
    y0 = y * y / (x * x + y * y)
    return x0, y0

def cal_kld_loss(mean, logvar):
    """
    Alternative KL divergence calculation.
    """
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()

class MinMaxNormalize:
    """
    Minmax normalization for a numpy array to (-1, 1)
    """
    def __init__(self, x, dim=0, dtype="float64"):
        self.dim = dim
        self.dtype = dtype
        x = x.astype(self.dtype)
        self.min = np.min(x, axis=dim, keepdims=True)
        self.max = np.max(x, axis=dim, keepdims=True)

    def normalize(self, x):
        x = x.astype(self.dtype)
        return 2 * (x - self.min) / (self.max - self.min) - 1

    def denormalize(self, x):
        x = x.astype(self.dtype)
        return (x + 1) / 2 * (self.max - self.min) + self.min

def plot_score_matrix(mat, labels=None, annot=True, fontsize=8, fmt='.3f', linewidths=1, tight_layout=True,
                      cmap='YlGnBu', tick_bin=1, ticklabel_rotation=0, ax=None, figsize=(6, 5),
                      diag_line=False, **kwargs):
    """
    Plots a heatmap score matrix.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mat, annot=annot, annot_kws={'size': fontsize}, fmt=fmt, linewidths=linewidths, cmap=cmap, ax=ax, **kwargs)

    if labels is None:
        labels = np.arange(0, len(mat), tick_bin)

    ticks = np.arange(0+.5, len(mat)+.5, tick_bin)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    if len(ticks) == len(labels):
        ax.set_xticklabels(labels, rotation=ticklabel_rotation, fontsize=fontsize)
        ax.set_yticklabels(labels, rotation=ticklabel_rotation, fontsize=fontsize)

    if diag_line:
        ax.plot([0, len(mat)], [0, len(mat)], c='gray')

    if tight_layout:
        plt.tight_layout()

    return ax

def plot_annot_square(idx, ax=None, **kwargs):
    """
    Plots annotated squares on the heatmap.
    """
    ys, xs = idx
    if ax is None:
        fig, ax = plt.subplots()

    for x, y in zip(xs, ys):
        ax.plot([x, x, x+1, x+1, x], [y, y+1, y+1, y, y], **kwargs)
    return ax

def find_differences(A, B):
    """
    Finds differences between two matrices.
    """
    differences = (A != B)
    rows, cols = np.where(differences)
    pos = [rows, cols]
    return pos

def delay_embedding2(data, embedding_dim, lag):
    """
    Performs delay embedding on the data.
    """
    data = np.array(data)
    n = data.shape[0]
    m = n - (embedding_dim - 1) * lag

    embedded_data = np.zeros((m, embedding_dim))

    for i in range(m):
        for j in range(embedding_dim):
            embedded_data[i, j] = data[i + embedding_dim - 1 - (j) * lag]
    embedded_data = embedded_data[:, ::-1]
    return embedded_data

def generate_embedd_data(data, embedding_dim, time_delay):
    """
    Generates embedded data using delay embedding.
    """
    X_dict0 = {}; X_dict1 = {}
    n_dataset0 = TensorDataset(); n_dataset1 = TensorDataset()
    for i in range(data.shape[1]):
        x_delay = delay_embedding2(data[:,i], embedding_dim, time_delay)
        x_delay0 = x_delay[:-2, :]; x_delay1 = x_delay[2:, :]  # t-1 and t
        normalizer_x0 = MinMaxNormalize(x_delay0)
        normalizer_x1 = MinMaxNormalize(x_delay1)
        n_x0 = normalizer_x0.normalize(x_delay0)
        n_x1 = normalizer_x1.normalize(x_delay1)
        n_x0 = torch.from_numpy(n_x0)
        n_x1 = torch.from_numpy(n_x1)
        X_dict0[f"X{i}"] = n_x0.float()
        X_dict1[f"X{i}"] = n_x1.float()
        n_dataset0.tensors += (X_dict0[f"X{i}"],)
        n_dataset1.tensors += (X_dict1[f"X{i}"],)

    return X_dict0, X_dict1, n_dataset0, n_dataset1

def logistic_3_system(rx, ry, rw, noise, betaxy, betaxz, betayx, betayz, num_steps):
    """
    Simulates a 3-variable logistic system.
    """
    Y = np.empty(num_steps)
    X = np.empty(num_steps)
    Z = np.empty(num_steps)

    X[0] = 0.4
    Y[0] = 0.4
    Z[0] = 0.4
    data0 = np.zeros((num_steps, 3))
    for j in range(1, num_steps):
        X[j] = X[j-1] * (rx - rx * X[j-1] - betaxy * Y[j-1] - betaxz * Z[j-1]) + np.random.normal(0, noise)
        Y[j] = Y[j-1] * (ry - ry * Y[j-1] - betayx * X[j-1] - betayz * Z[j-1]) + np.random.normal(0, noise)
        Z[j] = Z[j-1] * (rw - rw * Z[j-1]) + np.random.normal(0, noise)

    data0[:,0] = X; data0[:,1] = Y; data0[:,2] = Z
    return data0

def logistic_8_system(noise, beta, num_steps):
    """
    Simulates an 8-variable logistic system.
    """
    r = np.array([3.9, 3.5, 3.62, 3.75, 3.65, 3.72, 3.57, 3.68])
    data0 = np.zeros((num_steps, 8))
   
    X1 = np.empty(num_steps); X2 = np.empty(num_steps); X3 = np.empty(num_steps); X4 = np.empty(num_steps)
    X5 = np.empty(num_steps); X6 = np.empty(num_steps); X7 = np.empty(num_steps); X8 = np.empty(num_steps)

    X1[0] = 0.4; X2[0] = 0.4; X3[0] = 0.4; X4[0] = 0.4; X5[0] = 0.4; X6[0] = 0.4; X7[0] = 0.4; X8[0] = 0.4
    for j in range(1, num_steps):
        X1[j] = X1[j-1] * (r[0] - r[0] * X1[j-1]) + np.random.normal(0, noise)
        X2[j] = X2[j-1] * (r[1] - r[1] * X2[j-1]) + np.random.normal(0, noise)
        X3[j] = X3[j-1] * (r[2] - r[2] * X3[j-1] - beta * X1[j-1] - beta * X2[j-1]) + np.random.normal(0, noise)
        X4[j] = X4[j-1] * (r[3] - r[3] * X4[j-1] - beta * X2[j-1]) + np.random.normal(0, noise)
        X5[j] = X5[j-1] * (r[4] - r[4] * X5[j-1] - beta * X3[j-1]) + np.random.normal(0, noise)
        X6[j] = X6[j-1] * (r[5] - r[5] * X6[j-1] - beta * X3[j-1]) + np.random.normal(0, noise)
        X7[j] = X7[j-1] * (r[6] - r[6] * X7[j-1] - beta * X6[j-1]) + np.random.normal(0, noise)
        X8[j] = X8[j-1] * (r[7] - r[7] * X8[j-1] - beta * X6[j-1]) + np.random.normal(0, noise)
    data0[:,0] = X1; data0[:,1] = X2; data0[:,2] = X3; data0[:,3] = X4
    data0[:,4] = X5; data0[:,5] = X6; data0[:,6] = X7; data0[:,7] = X8
    return data0

def edges_to_mat(edges, num_nodes):
    """
    Converts edge list to adjacency matrix.
    """
    mat = np.zeros((num_nodes, num_nodes), dtype=int)
    for edge in edges:
        i, j = edge
        mat[i, j] = 1
    return mat

def GRN_Dream4_data(n_nodes=10, net_num=10):
    """
    Loads GRN Dream4 data.
    """
    GRN_Net = {}; GRN_data = {}
    for j in range(net_num):
        truth_path = f'./data/gene/net{j+1}_truth.tsv'
        expression_path = f'./data/gene/net{j+1}_expression.npy'
        if not os.path.exists(truth_path) or not os.path.exists(expression_path):
            continue
        Net_posi = pd.read_csv(truth_path, delimiter=r'\s+', header=None)
        data_posi = np.load(expression_path)
        edges_truth = Net_posi[[0, 1]].values.astype(int)
        gold_mat = edges_to_mat(edges_truth - 1, n_nodes)
        GRN_Net[f"Net{j}"] = torch.from_numpy(gold_mat)
        GRN_data[f"Net{j}"] = data_posi
    return GRN_Net, GRN_data

def Confounder(Net):
    """
    Introduces confounders into the network.
    """
    N = Net.shape[1]
    Updata_Net = Net.clone()
    Net_confd = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            if i != j:
                for k in range(N):
                    if k != j:
                        if Net[i, j] == 1:
                            if Net[i, k] == 1: 
                                Net_confd[j, k] = i
                                if Net[j, k] != 1:
                                    Updata_Net[j, k] = 2   
                                if Net[k, j] == 0:
                                    Updata_Net[k, j] = 2                               
    return Updata_Net, Net_confd 

def confound_CCA(data, Net_ground, Net_confd, out_s, Z0):
    """
    Computes CCA for confounders.
    """
    N = Net_ground.shape[1]
    CCA_matrix = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            if i != j:   
                if Net_ground[i, j] == 2:
                    index = torch.round(Net_confd[i, j]).long()
                    A = data[:, int(torch.round(Net_confd[i,j]))].reshape(1, -1)
                    B = out_s[f"s{i},s{j}"].detach().cpu().numpy()
                    B[np.isnan(B)] = 0
                    CCA_matrix[i, j] = CCA(A, B)
                if Net_ground[i, j] != 2:
                    B = out_s[f"s{i},s{j}"].detach().cpu().numpy()
                    B[np.isnan(B)] = 0
                    CCA_matrix[i, j] = CCA(data[:, Z0].reshape(1, -1), B)
    CCA_label2 = CCA_matrix[Net_ground == 2].tolist()
    CCA_label0 = CCA_matrix[Net_ground != 2].tolist()
    CCA_data = CCA_label2 + CCA_label0
    df = pd.DataFrame({'CCA': CCA_data, 'label': [2] * len(CCA_label2) + [0] * len(CCA_label0)})
    df['label'] = df['label'].replace({2: 'Confounder', 0: 'Non-confounder'})
    return CCA_matrix, df

def confound_CCA1(data, Net_ground, Net_confd, out_s, Z0):
    """
    Alternative CCA computation for confounders.
    """
    N = Net_ground.shape[1]
    CCA_matrix = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            if i != j:   
                if Net_ground[i, j] == 2:
                    index = torch.round(Net_confd[i, j]).long()
                    A = data[:, int(torch.round(Net_confd[i,j]))].reshape(1, -1)
                    B = out_s[f"s{i},s{j}"].detach().cpu().numpy()
                    B[np.isnan(B)] = 0
                    CCA_matrix[i, j] = CCA4(A, B)
                if Net_ground[i, j] != 2:
                    B = out_s[f"s{i},s{j}"].detach().cpu().numpy()
                    B[np.isnan(B)] = 0
                    CCA_matrix[i, j] = CCA4(data[:, Z0].reshape(1, -1), B)
    CCA_label2 = CCA_matrix[Net_ground == 2].tolist()
    CCA_label0 = CCA_matrix[Net_ground != 2].tolist()
    CCA_data = CCA_label2 + CCA_label0
    df = pd.DataFrame({'CCA': CCA_data, 'label': [2] * len(CCA_label2) + [0] * len(CCA_label0)})
    df['label'] = df['label'].replace({2: 'Confounder', 0: 'Non-confounder'})
    return CCA_matrix, df

def confounder_index(Net, Net_ground):
    """
    Computes various metrics for confounder analysis.
    """
    TP = np.count_nonzero((Net.reshape(-1) == 1) & (Net_ground.reshape(-1) == 1))
    FN = np.count_nonzero((Net.reshape(-1) == 0) & (Net_ground.reshape(-1) == 1))
    TN = np.count_nonzero((Net.reshape(-1) == 0) & (Net_ground.reshape(-1) == 0))
    FP = np.count_nonzero((Net.reshape(-1) == 1) & (Net_ground.reshape(-1) == 0))
    precision1, recall1, thresholds = precision_recall_curve(Net_ground[~torch.eye(Net_ground.shape[0], dtype=bool)].reshape(-1), Net[~torch.eye(Net.shape[0], dtype=bool)].reshape(-1))
    precision0 = (TN / (TN + FP + 1e-6))
    recall0 = (TN / (TN + FN + 1e-6))
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = (2 * precision0 + recall1) / (precision1 + recall1 + 1e-6)  

    return precision0, precision1, recall0, recall1, accuracy, f1

# --------------------------- CCA Functions --------------------------- #

def CCA(A, B):
    A = np.transpose(A)
    min_samples = min(A.shape[0], B.shape[0])    
    A = A[:min_samples]  
    B = B[:min_samples]

    cca = sklearn_CCA(n_components=1)
    cca.fit(A, B)
    A_train_r, B_train_r = cca.transform(A, B)
    canonical_correlation = np.corrcoef(A_train_r[:,0], B_train_r[:,0])[0,1]
    canonical_correlation = np.clip(canonical_correlation, -1, 1)
    return canonical_correlation

def CCA2(A, B):
    A = np.transpose(A)
    min_samples = min(A.shape[0], B.shape[0])    
    A = A[:min_samples]
    B = B[:min_samples]    

    ica_A = FastICA(n_components=1)
    A_ica = ica_A.fit_transform(A)

    ica_B = FastICA(n_components=1)
    B_ica = ica_B.fit_transform(B)

    ica_corr = np.corrcoef(A_ica.squeeze(), B_ica.squeeze())[0, 1]
    return ica_corr

def CCA4(A, B):
    A = np.transpose(A)
    min_samples = min(A.shape[0], B.shape[0])    
    A = A[:min_samples]
    B = B[:min_samples]    

    corr_matrix = np.corrcoef(A.T, B.T)

    eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)

    max_eigenvalue_index = np.argmax(eigenvalues)
    max_eigenvector = eigenvectors[:, max_eigenvalue_index]
    overall_correlation = np.max(np.abs(max_eigenvector))

    return overall_correlation

# --------------------------- Plotting Functions --------------------------- #

def plot_score_matrix(mat, labels=None, annot=True, fontsize=8, fmt='.3f', linewidths=1, tight_layout=True,
                      cmap='YlGnBu', tick_bin=1, ticklabel_rotation=0, ax=None, figsize=(6, 5),
                      diag_line=False, **kwargs):
    """
    Plots a heatmap score matrix.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mat, annot=annot, annot_kws={'size': fontsize}, fmt=fmt, linewidths=linewidths, cmap=cmap, ax=ax, **kwargs)

    if labels is None:
        labels = np.arange(0, len(mat), tick_bin)

    ticks = np.arange(0+.5, len(mat)+.5, tick_bin)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    if len(ticks) == len(labels):
        ax.set_xticklabels(labels, rotation=ticklabel_rotation, fontsize=fontsize)
        ax.set_yticklabels(labels, rotation=ticklabel_rotation, fontsize=fontsize)

    if diag_line:
        ax.plot([0, len(mat)], [0, len(mat)], c='gray')

    if tight_layout:
        plt.tight_layout()

    return ax

def plot_annot_square(idx, ax=None, **kwargs):
    """
    Plots annotated squares on the heatmap.
    """
    ys, xs = idx
    if ax is None:
        fig, ax = plt.subplots()

    for x, y in zip(xs, ys):
        ax.plot([x, x, x+1, x+1, x], [y, y+1, y+1, y, y], **kwargs)
    return ax

def plot_performance(confounder_levels, avg_rewards_baseline, avg_rewards_causal):
    """
    Plot average rewards against confounder levels for both agents.
    Includes moving averages with window size 50.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=confounder_levels, y=avg_rewards_baseline, marker='o', label='Baseline PPO')
    sns.lineplot(x=confounder_levels, y=avg_rewards_causal, marker='s', label='Causal PPO')
    plt.xlabel('Confounder Level')
    plt.ylabel('Average Reward')
    plt.title('PPO Agents Performance under Varying Confounder Levels')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    wandb.log({"Robustness Analysis/Performance Comparison": wandb.Image(plt)})
    plt.close()

def plot_moving_average(rewards, window=50, label=''):
    """
    Plot moving average of rewards.
    """
    if len(rewards) < window:
        window = len(rewards)
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg, label=label)
    return moving_avg

# --------------------------- Latent Representation Analysis Function --------------------------- #

def latent_representation_analysis(model, dataloader):
    """
    Analyze the correlation between shared latents and ground truth variables using multiple statistical measures.
    """
    model.eval()
    latent_reps = []
    ground_truth_posX = []
    ground_truth_posY = []
    with torch.no_grad():
        for batch in dataloader:
            data_x, data_y = batch  # Assuming x = y for simplicity
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            output = model(data_x, data_y)
            # Extract shared latents: s_x and s_y
            s_x = output.sample["s_x"].cpu().numpy()
            s_y = output.sample["s_y"].cpu().numpy()
            latent_reps.append(s_x)
            ground_truth_posX.append(data_x[:,4].cpu().numpy())
            ground_truth_posY.append(data_x[:,5].cpu().numpy())
    latent_reps = np.concatenate(latent_reps, axis=0)
    ground_truth_posX = np.concatenate(ground_truth_posX, axis=0)
    ground_truth_posY = np.concatenate(ground_truth_posY, axis=0)

    # Compute Pearson, Kendall Tau, Spearman Rho correlations
    print("\n=== Latent Representation Correlation ===")
    for i in range(latent_reps.shape[1]):
        corr_pearson_x, _ = pearsonr(latent_reps[:,i], ground_truth_posX)
        corr_kendall_x, _ = kendalltau(latent_reps[:,i], ground_truth_posX)
        corr_spearman_x, _ = spearmanr(latent_reps[:,i], ground_truth_posX)
        
        corr_pearson_y, _ = pearsonr(latent_reps[:,i], ground_truth_posY)
        corr_kendall_y, _ = kendalltau(latent_reps[:,i], ground_truth_posY)
        corr_spearman_y, _ = spearmanr(latent_reps[:,i], ground_truth_posY)
        
        print(f"s_x[{i}] vs posX - Pearson: {corr_pearson_x:.4f}, Kendall Tau: {corr_kendall_x:.4f}, Spearman Rho: {corr_spearman_x:.4f}")
        print(f"s_x[{i}] vs posY - Pearson: {corr_pearson_y:.4f}, Kendall Tau: {corr_kendall_y:.4f}, Spearman Rho: {corr_spearman_y:.4f}")
        
        wandb.log({
            f"Latent_Correlation/s_x_{i}_vs_posX_Pearson": corr_pearson_x,
            f"Latent_Correlation/s_x_{i}_vs_posX_KendallTau": corr_kendall_x,
            f"Latent_Correlation/s_x_{i}_vs_posX_SpearmanRho": corr_spearman_x,
            f"Latent_Correlation/s_x_{i}_vs_posY_Pearson": corr_pearson_y,
            f"Latent_Correlation/s_x_{i}_vs_posY_KendallTau": corr_kendall_y,
            f"Latent_Correlation/s_x_{i}_vs_posY_SpearmanRho": corr_spearman_y
        })

    # Compute Mutual Information
    mutual_info_posX = normalized_mutual_info_score(ground_truth_posX, latent_reps.argmax(axis=1))
    mutual_info_posY = normalized_mutual_info_score(ground_truth_posY, latent_reps.argmax(axis=1))
    print(f"Mutual Information posX: {mutual_info_posX:.4f}, posY: {mutual_info_posY:.4f}")
    wandb.log({
        "Latent_Correlation/Mutual Information posX": mutual_info_posX,
        "Latent_Correlation/Mutual Information posY": mutual_info_posY
    })

# --------------------------- Causal Index Computation --------------------------- #

def compute_causal_index(model, dataloader):
    """
    Computes the causal index based on the latent representations from DualVAE.
    Utilizes multiple CCA techniques to measure the correlation between original data and latent variables.
    """
    model.eval()
    latent_reps = []
    original_data = []
    with torch.no_grad():
        for batch in dataloader:
            data_x, data_y = batch  # Assuming x = y for simplicity
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            output = model(data_x, data_y)
            # Extract shared latents: s_x and s_y
            s_x = output.sample["s_x"].cpu().numpy()
            s_y = output.sample["s_y"].cpu().numpy()
            latent_reps.append(s_x)
            original_data.append(data_x.cpu().numpy())
    latent_reps = np.concatenate(latent_reps, axis=0)
    original_data = np.concatenate(original_data, axis=0)

    # Reshape the arrays to have the same size when flattened
    original_data_flat = original_data.reshape(-1)
    latent_reps_flat = latent_reps.reshape(-1)
    
    # If sizes are still different, truncate to the smaller size
    min_size = min(len(original_data_flat), len(latent_reps_flat))
    original_data_flat = original_data_flat[:min_size]
    latent_reps_flat = latent_reps_flat[:min_size]

    # Perform various CCA analyses
    cca_pearson = sklearn_CCA(n_components=1)
    cca_pearson.fit(original_data, latent_reps)
    X_c, Y_c = cca_pearson.transform(original_data, latent_reps)
    correlation_pearson = np.corrcoef(X_c[:,0], Y_c[:,0])[0,1]
    correlation_pearson = np.clip(correlation_pearson, -1, 1)

    correlation_kendall, _ = kendalltau(original_data_flat, latent_reps_flat)
    correlation_spearman, _ = spearmanr(original_data_flat, latent_reps_flat)
    mutual_info = normalized_mutual_info_score(original_data_flat, latent_reps_flat)

    # Aggregate correlations
    correlations = {
        "Pearson": correlation_pearson,
        "Kendall Tau": correlation_kendall,
        "Spearman Rho": correlation_spearman,
        "Mutual Information": mutual_info
    }

    # Log correlations to wandb
    for key, value in correlations.items():
        wandb.log({f"Causal Index/{key}": value})

    print(f"CCA Pearson Correlation (Causal Index): {correlation_pearson:.4f}")
    return correlation_pearson

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

    loss_history = []

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
        loss_history.append(avg_loss)

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

# --------------------------- Causal Index Computation --------------------------- #

# CCA functions already defined above

# --------------------------- PPO Agent Definitions --------------------------- #

class PPOActorCritic(nn.Module):
    """
    Actor-Critic network for PPO, incorporating separate state and latent encoders.
    """
    def __init__(self, state_dim, latent_dim, action_dim, causal_var_indices):
        super(PPOActorCritic, self).__init__()
        self.causal_var_indices = causal_var_indices  # Indices of causal variables in the latent vector

        # State processing stream
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        # Latent features processing stream (only shared latents)
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic network
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
    PPO Agent that incorporates frontdoor adjustments to deconfound policy learning.
    """
    def __init__(self, state_dim, latent_dim, action_dim, causal_var_indices, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2, causal_weight=0.1):
        self.policy = PPOActorCritic(state_dim, latent_dim, action_dim, causal_var_indices).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = PPOActorCritic(state_dim, latent_dim, action_dim, causal_var_indices).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.causal_weight = causal_weight
        self.mse_loss = nn.MSELoss()
        
        # Frontdoor Adjustment Modules
        # Module to estimate P(M | A)
        self.PM_given_A = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),  # Assuming mediator has 'latent_dim' dimensions
            nn.Sigmoid()  # Assuming mediators are normalized between 0 and 1
        ).to(device)
        
        # Module to estimate P(Y | M, A)
        self.PY_given_MA = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Predicting expected reward
        ).to(device)
        
        # Optimizer for Frontdoor Modules
        self.optimizer_frontdoor = optim.Adam(list(self.PM_given_A.parameters()) + list(self.PY_given_MA.parameters()), lr=lr)
        
    def select_action(self, state, latent):
        """
        Select action using the current policy.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        latent = torch.FloatTensor(latent).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob, _ = self.policy_old.act(state, latent)
        return action, log_prob.item()
    
    def update_frontdoor_modules(self, actions, latents, rewards):
        """
        Update the frontdoor adjustment modules P(M | A) and P(Y | M, A).
        """
        actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=self.policy.actor[-1].out_features).float().to(device)
        
        # Estimate P(M | A)
        PM_A = self.PM_given_A(actions_one_hot)
        
        # Estimate P(Y | M, A)
        MA = torch.cat([PM_A, actions_one_hot], dim=1)  # Concatenate mediator and action
        PY_MA = self.PY_given_MA(MA).squeeze(-1)  # Expected reward
        
        # Compute loss between estimated P(Y | M, A) and actual rewards
        loss_PY = self.mse_loss(PY_MA, rewards)
        
        # Backpropagation
        self.optimizer_frontdoor.zero_grad()
        loss_PY.backward()
        self.optimizer_frontdoor.step()
        
        return loss_PY.item()
    
    def compute_frontdoor_adjusted_rewards(self, actions, rewards):
        """
        Compute frontdoor-adjusted rewards using the frontdoor formula.
        """
        actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=self.policy.actor[-1].out_features).float().to(device)
        
        # Estimate P(M | A)
        PM_A = self.PM_given_A(actions_one_hot)
        
        # Estimate P(Y | M, A)
        MA = torch.cat([PM_A, actions_one_hot], dim=1)  # Concatenate mediator and action
        PY_MA = self.PY_given_MA(MA).squeeze(-1)  # Expected reward
        
        # Frontdoor Adjustment: P(Y | do(A)) = sum_M P(M | A) * P(Y | M, A)
        # Since M is continuous, we approximate the integral with expectation
        frontdoor_adjusted_rewards = torch.sum(PM_A * PY_MA.unsqueeze(-1), dim=1)
        
        return frontdoor_adjusted_rewards
    
    def update_policy(self, memory, causal_index):
        """
        Update the PPO policy using the stored memory and causal index.
        """
        old_states = torch.FloatTensor(memory['states']).to(device)
        old_latents = torch.FloatTensor(memory['latents']).to(device)
        old_actions = torch.LongTensor(memory['actions']).to(device)
        old_logprobs = torch.FloatTensor(memory['logprobs']).to(device)
        rewards = torch.FloatTensor(memory['rewards']).to(device)
        is_terminals = memory['is_terminals']
        
        # Frontdoor Adjustment: Compute adjusted rewards
        adjusted_rewards = self.compute_frontdoor_adjusted_rewards(old_actions, rewards)
        
        # Generalized Advantage Estimation (GAE)
        advantages = []
        returns = []
        G = 0
        gae = 0
        lam = 0.95  # GAE lambda
        for reward, done in zip(reversed(adjusted_rewards), reversed(is_terminals)):
            if done:
                G = 0
                gae = 0
            delta = reward + self.gamma * G - 0  # Assuming value function baseline is 0
            gae = delta + self.gamma * lam * gae
            advantages.insert(0, gae)
            G = reward + self.gamma * G
            returns.insert(0, G)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        # Initialize counters for logging
        policy_loss_total = 0.0
        entropy_total = 0.0
        regularization_total = 0.0
        loss_PY_total = 0.0
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            logprobs, state_values, entropy, dist = self.policy.evaluate(old_states, old_latents, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # PPO Loss
            loss = (-torch.min(surr1, surr2)).mean() + \
                   0.5 * self.mse_loss(state_values, returns) - \
                   0.01 * entropy.mean()
            
            # Causal Regularization: Penalize high reliance on non-causal latent variables
            reg_loss = torch.tensor(0.0, device=device)  # Initialize as tensor
            # Define non-causal indices
            non_causal_indices = [i for i in range(self.policy.latent_encoder[0].in_features) if i not in self.policy.causal_var_indices]
            for idx in non_causal_indices:
                reg_loss += torch.norm(self.policy.latent_encoder[0].weight[:, idx], 2)
            loss += self.causal_weight * reg_loss
            regularization_total += (self.causal_weight * reg_loss).item()
            
            # Frontdoor Adjustment Loss (optional: encourage accurate adjustment)
            # For simplicity, we're already updating the frontdoor modules separately
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            # Accumulate losses for logging
            policy_loss_total += loss.item()
            entropy_total += entropy.mean().item()
        
        # Calculate average losses
        policy_loss_avg = policy_loss_total / self.K_epochs
        entropy_avg = entropy_total / self.K_epochs
        regularization_avg = regularization_total / self.K_epochs
        
        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Adjust causal_weight based on causal_index
        # Define a dynamic adjustment strategy to stabilize learning
        threshold_high = 0.6
        threshold_low = 0.4
        adjustment_factor = 0.05  # Adjust by 5%
        
        if causal_index > threshold_high:
            # High confounding: reduce regularization to allow the policy to adapt
            self.causal_weight *= (1 - adjustment_factor)
            print(f"Causal Index {causal_index:.2f} > {threshold_high}, decreasing causal_weight to {self.causal_weight:.4f}")
        elif causal_index < threshold_low:
            # Low confounding: increase regularization to enforce deconfounding
            self.causal_weight *= (1 + adjustment_factor)
            self.causal_weight = min(self.causal_weight, 1.0)  # Cap at 1.0
            print(f"Causal Index {causal_index:.2f} < {threshold_low}, increasing causal_weight to {self.causal_weight:.4f}")
        
        # Log losses to wandb
        wandb.log({
            "Causal PPO/Policy Loss": policy_loss_avg,
            "Causal PPO/Entropy": entropy_avg,
            "Causal PPO/Regularization": regularization_avg,
            "Causal PPO/Causal Weight": self.causal_weight
        })
        
        print(f"Causal PPO Update - Policy Loss: {policy_loss_avg:.4f}, Entropy: {entropy_avg:.4f}, Regularization: {regularization_avg:.4f}, Causal Weight: {self.causal_weight:.4f}")

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

        # Generalized Advantage Estimation (GAE)
        advantages = []
        returns = []
        G = 0
        gae = 0
        lam = 0.95  # GAE lambda
        for reward, done in zip(reversed(rewards), reversed(is_terminals)):
            if done:
                G = 0
                gae = 0
            delta = reward + self.gamma * G - 0  # Assuming value function baseline is 0
            gae = delta + self.gamma * lam * gae
            advantages.insert(0, gae)
            G = reward + self.gamma * G
            returns.insert(0, G)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Initialize counters for logging
        policy_loss_total = 0.0
        entropy_total = 0.0

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            logprobs, state_values, entropy, dist = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = (-torch.min(surr1, surr2)).mean() + \
                   0.5 * self.mse_loss(state_values, returns) - \
                   0.01 * entropy.mean()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            # Accumulate losses for logging
            policy_loss_total += loss.item()
            entropy_total += entropy.mean().item()

        # Calculate average losses
        policy_loss_avg = policy_loss_total / self.K_epochs
        entropy_avg = entropy_total / self.K_epochs

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Log losses to wandb
        wandb.log({
            "Baseline PPO/Policy Loss": policy_loss_avg,
            "Baseline PPO/Entropy": entropy_avg
        })

        # Log to console
        print(f"Baseline PPO Update - Policy Loss: {policy_loss_avg:.4f}, Entropy: {entropy_avg:.4f}")

# --------------------------- Additional Analysis Functions --------------------------- #

def perform_statistical_tests(rewards_causal, rewards_baseline):
    """
    Perform t-tests to assess the statistical significance of reward differences.
    """
    print("\n=== Statistical Significance Testing ===")
    t_stat, p_value = ttest_ind(rewards_causal, rewards_baseline)
    print(f"T-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    wandb.log({"Statistical_Test/t_stat": t_stat, "Statistical_Test/p_value": p_value})
    if p_value < 0.05:
        print("Result: Reject the null hypothesis (statistically significant difference).")
    else:
        print("Result: Fail to reject the null hypothesis (no statistically significant difference).")

# --------------------------- Main Training Loop for Robustness Analysis --------------------------- #
    
    def main():
        # Initialize wandb project
        wandb.init(project="DualVAE_PPO_Frontdoor_Analysis", config={
            "num_data_steps": 100000,  # Increased from 50,000 to 100,000 for more comprehensive data
            "dualvae_epochs": 100,      # Increased from 50 to 100 epochs
            "dualvae_batch_size": 128,
            "dualvae_learning_rate": 1e-3,
            "ppo_max_episodes": 10000,   # Increased from 2000 to 10000 episodes
            "ppo_max_timesteps": 200,
            "ppo_update_timestep": 4000,
            "ppo_lr": 3e-4,
            "ppo_gamma": 0.99,
            "ppo_K_epochs": 10,
            "ppo_eps_clip": 0.2,
            "ppo_causal_weight": 0.1,
            "dualvae_loss_weights": [1.0, 1.0, 1.0, 0.1, 0.1, 0.001, 0.001],
            "causal_var_indices": [0, 1],  # Adjusted indices for posX and posY within s_x
            "moving_average_window": 50
        })
        
        # Define different confounder amplitude levels for robustness analysis
        confounder_levels = {
            "Low": [
                {"state_multiplier": np.ones(6, dtype=np.float32), "reward_bias": 0.0},
                {"state_multiplier": np.ones(6, dtype=np.float32) * 0.95, "reward_bias": -0.2},
                {"state_multiplier": np.ones(6, dtype=np.float32) * 1.05, "reward_bias": 0.2},
            ],
            "Medium": [
                {"state_multiplier": np.ones(6, dtype=np.float32), "reward_bias": 0.0},
                {"state_multiplier": np.ones(6, dtype=np.float32) * 0.9, "reward_bias": -0.5},
                {"state_multiplier": np.ones(6, dtype=np.float32) * 1.1, "reward_bias": 0.5},
            ],
            "High": [
                {"state_multiplier": np.ones(6, dtype=np.float32), "reward_bias": 0.0},
                {"state_multiplier": np.ones(6, dtype=np.float32) * 0.8, "reward_bias": -1.0},
                {"state_multiplier": np.ones(6, dtype=np.float32) * 1.2, "reward_bias": 1.0},
            ]
        }
        
        # Initialize lists to store average rewards for each confounder level
        avg_rewards_baseline = []
        avg_rewards_causal = []
        level_names = []
        
        # Iterate over each confounder level
        for level_name, confounder_amplitudes in confounder_levels.items():
            print(f"\n=== Starting Robustness Analysis for Confounder Level: {level_name} ===")
            
            # Initialize environment with specific confounder amplitudes
            base_env = NavigationEnv(num_games=1, max_steps=200)
            env = ConfounderEnvWrapper(base_env, confounder_states=confounder_amplitudes)
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
            
            # Perform Latent Representation Analysis
            latent_representation_analysis(dualvae, dataloader)
            
            # Compute causal index
            print("Computing causal index based on DualVAE latent representations...")
            causal_index = compute_causal_index(dualvae, dataloader)
            wandb.log({"Causal Index": causal_index})
            print(f"Causal Index: {causal_index:.4f}\n")
            
            # Freeze DualVAE parameters to prevent updates during PPO training
            for param in dualvae.parameters():
                param.requires_grad = False
            
            # Initialize PPO agents
            causal_var_indices = wandb.config.causal_var_indices  # Indices of posX and posY within s_x
            ppo_agent_baseline = BaselinePPOAgent(state_dim, action_dim,
                                                 lr=wandb.config.ppo_lr,
                                                 gamma=wandb.config.ppo_gamma,
                                                 K_epochs=wandb.config.ppo_K_epochs,
                                                 eps_clip=wandb.config.ppo_eps_clip)  # Baseline PPO
            ppo_agent_causal = CausalPPOAgent(state_dim, latent_dim=4, action_dim=action_dim,
                                             causal_var_indices=causal_var_indices,
                                             lr=wandb.config.ppo_lr,
                                             gamma=wandb.config.ppo_gamma,
                                             K_epochs=wandb.config.ppo_K_epochs,
                                             eps_clip=wandb.config.ppo_eps_clip,
                                             causal_weight=wandb.config.ppo_causal_weight)  # Causal PPO with Frontdoor
            
            # Training parameters
            max_episodes = wandb.config.ppo_max_episodes  # Number of episodes for training
            max_timesteps = wandb.config.ppo_max_timesteps    # Max timesteps per episode
            update_timestep = wandb.config.ppo_update_timestep  # Update PPO agent every 'update_timestep' timesteps
            timestep = 0
            
            # Initialize memory for PPO agents
            memory_baseline = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
            memory_causal = {'states': [], 'latents': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
            
            # Initialize rewards tracking
            rewards_baseline = []
            rewards_causal = []
            moving_avg_baseline = []
            moving_avg_causal = []
            
            # --------------------------- Training Baseline PPO Agent --------------------------- #
            
            print("Starting training of Baseline PPO agent...")
            timestep = 0
            for episode in tqdm(range(1, max_episodes + 1), desc=f"Baseline PPO Training [{level_name}]"):
                state = env.reset()
                ep_reward = 0
                for t in range(max_timesteps):
                    timestep += 1
                    action, log_prob = ppo_agent_baseline.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    
                    # Save raw environment data in memory
                    memory_baseline['states'].append(state)
                    memory_baseline['actions'].append(action)
                    memory_baseline['logprobs'].append(log_prob)
                    memory_baseline['rewards'].append(reward)
                    memory_baseline['is_terminals'].append(done)
                    
                    state = next_state
                    ep_reward += reward
                    
                    # Update PPO agent with raw data
                    if timestep % update_timestep == 0:
                        ppo_agent_baseline.update_policy(memory_baseline)
                        memory_baseline = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
                    
                    if done:
                        break
                rewards_baseline.append(ep_reward)
                # Compute moving average
                if len(rewards_baseline) >= wandb.config.moving_average_window:
                    avg = np.mean(rewards_baseline[-wandb.config.moving_average_window:])
                    moving_avg_baseline.append(avg)
                else:
                    moving_avg_baseline.append(np.mean(rewards_baseline))
                # Logging after every episode
                wandb.log({
                    "Baseline PPO/Episode Reward": ep_reward,
                    "Baseline PPO/Moving Average Reward": moving_avg_baseline[-1],
                    "Episode": episode,
                    "Confounder Level": level_name
                })
                # Logging progress
                if episode % 100 == 0 or episode == 1:
                    print(f"Baseline PPO Agent - Episode {episode}, Average Reward (last {wandb.config.moving_average_window}): {moving_avg_baseline[-1]:.2f}")
            print("\nTraining of Baseline PPO agent completed.\n")
            
            # --------------------------- Training Causal-enhanced PPO Agent with Frontdoor --------------------------- #
            
            print("Starting training of Causal-enhanced PPO agent with Frontdoor adjustments...")
            timestep = 0
            for episode in tqdm(range(1, max_episodes + 1), desc=f"Causal PPO Training [{level_name}]"):
                state = env.reset()
                ep_reward = 0
                for t in range(max_timesteps):
                    timestep += 1
                    # Use DualVAE for state representation
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    dualvae.eval()
                    with torch.no_grad():
                        output = dualvae(state_tensor, state_tensor)
                        s_x = output.sample["s_x"].cpu().numpy()[0]
                    
                    # Select action using Causal PPO with DualVAE representation
                    action, log_prob = ppo_agent_causal.select_action(state, s_x)
                    next_state, reward, done, _ = env.step(action)
                    
                    # Save data in memory including latent representations
                    memory_causal['states'].append(state)
                    memory_causal['latents'].append(s_x)
                    memory_causal['actions'].append(action)
                    memory_causal['logprobs'].append(log_prob)
                    memory_causal['rewards'].append(reward)
                    memory_causal['is_terminals'].append(done)
                    
                    state = next_state
                    ep_reward += reward
                    
                    # Update Causal PPO agent with Frontdoor adjustments
                    if timestep % update_timestep == 0:
                        # Update frontdoor modules P(M | A) and P(Y | M, A)
                        actions_tensor = torch.LongTensor(memory_causal['actions']).to(device)
                        latents_tensor = torch.FloatTensor(memory_causal['latents']).to(device)
                        rewards_tensor = torch.FloatTensor(memory_causal['rewards']).to(device)
                        
                        # Update frontdoor modules
                        loss_PY = ppo_agent_causal.update_frontdoor_modules(actions_tensor, latents_tensor, rewards_tensor)
                        wandb.log({"Causal PPO/Frontdoor PY Loss": loss_PY})
                        
                        # Compute causal index based on DualVAE
                        causal_index = compute_causal_index(dualvae, dataloader)
                        
                        # Update policy with frontdoor-adjusted rewards
                        ppo_agent_causal.update_policy(memory_causal, causal_index=causal_index)
                        memory_causal = {'states': [], 'latents': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
                    
                    if done:
                        break
                rewards_causal.append(ep_reward)
                # Compute moving average
                if len(rewards_causal) >= wandb.config.moving_average_window:
                    avg = np.mean(rewards_causal[-wandb.config.moving_average_window:])
                    moving_avg_causal.append(avg)
                else:
                    moving_avg_causal.append(np.mean(rewards_causal))
                # Logging after every episode
                wandb.log({
                    "Causal PPO/Episode Reward": ep_reward,
                    "Causal PPO/Moving Average Reward": moving_avg_causal[-1],
                    "Episode": episode,
                    "Confounder Level": level_name
                })
                # Logging progress
                if episode % 100 == 0 or episode == 1:
                    print(f"Causal-enhanced PPO Agent - Episode {episode}, Average Reward (last {wandb.config.moving_average_window}): {moving_avg_causal[-1]:.2f}")
            print("\nTraining of Causal-enhanced PPO agent with Frontdoor adjustments completed.\n")
            
            # Calculate average rewards for this confounder level
            avg_reward_baseline = np.mean(rewards_baseline[-300:])  # Average over last 300 episodes
            avg_reward_causal = np.mean(rewards_causal[-300:])      # Average over last 300 episodes
            avg_rewards_baseline.append(avg_reward_baseline)
            avg_rewards_causal.append(avg_reward_causal)
            level_names.append(level_name)
            
            # Log comparison for this confounder level with moving averages
            plt.figure(figsize=(10, 6))
            plt.plot(moving_avg_baseline, label='Baseline PPO')
            plt.plot(moving_avg_causal, label='Causal PPO with Frontdoor')
            plt.xlabel('Episode')
            plt.ylabel(f'Moving Average Reward (Window={wandb.config.moving_average_window})')
            plt.title(f'Performance Comparison at {level_name} Confounder Level')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            wandb.log({f"Performance_Comparison/{level_name}": wandb.Image(plt)})
            plt.close()
        
        # --------------------------- Performance Visualization --------------------------- #
        
        plot_performance(level_names, avg_rewards_baseline, avg_rewards_causal)
        
        # --------------------------- Statistical Significance Testing --------------------------- #
        
        perform_statistical_tests(np.array(avg_rewards_causal), np.array(avg_rewards_baseline))
        
        # --------------------------- Finish Training --------------------------- #
        
        print("Robustness Analysis with Frontdoor Adjustments completed successfully.")
        wandb.finish()

    # --------------------------- Execution Entry Point --------------------------- #
    
    if __name__ == '__main__':
        main()

# At the bottom of the file
if __name__ == "__main__":
    # Initialize environment
    env = NavigationEnv(num_games=1, max_steps=100)
    print("Environment initialized")
    
    # Test environment
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    # Initialize the confounded environment
    confounded_env = ConfounderEnvWrapper(env)
    print("\nStarting training loop...")
    
    # Initialize agents
    state_dim = 6
    action_dim = 4
    latent_dim = 4
    causal_var_indices = [4, 5]  # posX and posY
    
    causal_agent = CausalPPOAgent(state_dim, latent_dim, action_dim, causal_var_indices)
    baseline_agent = BaselinePPOAgent(state_dim, action_dim)
    
    # Run a few test episodes
    n_episodes = 5
    for episode in range(n_episodes):
        state = confounded_env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < env.max_steps:
            action, _ = causal_agent.select_action(state)
            next_state, reward, done, _ = confounded_env.step(action)
            episode_reward += reward
            state = next_state
            step += 1
            
        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}, Steps = {step}")
    
    print("\nTest run completed!")