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

# --------------------------- Reproducibility and Device Configuration --------------------------- #

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random_seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------- Custom Gym Environment Definition --------------------------- #

# (Removed NavigationEnv definition since we're replacing it)

# --------------------------- Confounded Environment Wrapper Definition --------------------------- #

class ConfounderEnvWrapper(gym.Wrapper):
    """
    Gym environment wrapper that adds hidden confounders influencing both state transitions and rewards.
    """
    def __init__(self, env, confounder_freqs=None, confounder_amplitudes=None, noise_std=0.1):
        super(ConfounderEnvWrapper, self).__init__(env)
        self.confounder_freqs = confounder_freqs if confounder_freqs is not None else [0.5, 0.3, 0.2, 0.4]
        self.confounder_amplitudes = confounder_amplitudes if confounder_amplitudes is not None else [0.2, 0.15, 0.1, 0.25]
        self.noise_std = noise_std
        self.time_step = 0

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
        modified_reward = reward + confounder * 0.2  # Increased confounder impact on reward
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

# --------------------------- BiddingSimulation Environment Definition --------------------------- #

from typing import Callable, Iterable, Optional, Union

def sum_array_bool(arr: np.ndarray) -> int:
    """Sum a boolean array as integers."""
    return int(np.sum(arr))

def sum_array(arr: np.ndarray) -> float:
    """Sum a numpy array."""
    return float(np.sum(arr))

def sum_list(lst: list) -> float:
    """Sum a list of numbers."""
    return float(np.sum(lst))

def list_to_zeros(lst: list) -> np.ndarray:
    """Convert a list to a numpy array of zeros with the same length."""
    return np.zeros(len(lst))

def binomial_impressions(num_auctions: int, impression_rate: float) -> int:
    """Simulate the number of impressions using a binomial distribution."""
    return np.random.binomial(num_auctions, impression_rate)

def threshold_sigmoid(x: float, params: dict) -> float:
    """
    Apply a thresholded sigmoid function to x.

    Parameters:
        x (float): The input value.
        params (dict): Parameters containing 'impression_slope', 'impression_bid_intercept', and 'impression_thresh'.

    Returns:
        float: The sigmoid-transformed and thresholded output.
    """
    s = params.get("impression_slope", 1.0)
    t = params.get("impression_bid_intercept", 1.0)
    thresh = params.get("impression_thresh", 0.05)
    rate = 1.0 / (1.0 + np.exp(-s * (x - t)))
    rate = np.clip(rate, thresh, 1.0 - thresh)
    return rate

def cost_create(bid: float, n: int) -> np.ndarray:
    """
    Create costs for buyside clicks based on the bid.

    Parameters:
        bid (float): The bid amount.
        n (int): Number of costs to generate.

    Returns:
        np.ndarray: Array of costs.
    """
    mean_cost = np.sqrt(bid) / 4 + bid / 2
    cost_noise = np.random.normal(0, 1e-10 + np.sqrt(bid) / 6, size=n)
    costs = np.around(np.clip(mean_cost + cost_noise, 0.0, bid), 2).astype(float)
    return costs

def repr_outcomes_py(outcomes: dict) -> str:
    """Represent bidding outcomes as a string."""
    return str(outcomes)

def nonneg_int_normal_sampler(the_mean: float, std: float) -> Callable[[], int]:
    """
    Create a sampler for non-negative integers from a normal distribution.

    Parameters:
        the_mean (float): The mean of the normal distribution.
        std (float): The standard deviation of the normal distribution.

    Returns:
        Callable[[], int]: A function that samples an integer.
    """
    def sampler() -> int:
        sample = np.random.normal(the_mean, std)
        return int(max(0, round(sample)))
    return sampler

class Keyword:
    """
    A base class to represent a keyword for bidding.

    Attributes:
        rng (np.random.Generator): Random number generator.
        volume_sampler (Callable[[], int]): Function to sample volume.
        buyside_ctr (float): Probability of buyside click given an impression.
        sellside_paid_ctr (float): Probability of sellside conversion given a buyside click.
        reward_distribution_sampler (Callable[[int], Iterable[float]]): Function to sample rewards.
    """
    def __init__(self, params: dict = {}, verbose=False) -> None:
        """Initialize the Keyword with given parameters."""
        self._validate_parameters(params, verbose)
        self.rng = self._rng_init(params)
        self.volume_sampler = self._volume_sampler_init(params)
        self.buyside_ctr = self._buyside_ctr_init(params)
        self.sellside_paid_ctr = self._sellside_paid_ctr_init(params)
        self.reward_distribution_sampler = self._reward_distribution_sampler_init(params)

    def sample_volume(self, n: int = 1) -> np.ndarray:
        """Sample volume n times."""
        return np.array([self.volume_sampler() for _ in range(n)])

    def auction(self, bid: float, num_auctions: int) -> tuple:
        """Simulate auctions (to be implemented by subclasses)."""
        raise NotImplementedError("Must use a subclass like ExplicitKeyword or ImplicitKeyword")

    def sample_buyside_click(self, n: int = 1) -> np.ndarray:
        """Sample buyside clicks."""
        return self.rng.random(n) <= self.buyside_ctr

    def sample_sellside_paid_click(self, n: int = 1) -> np.ndarray:
        """Sample sellside paid clicks."""
        return self.rng.random(n) <= self.sellside_paid_ctr

    def sample_reward(self, n: int = 1) -> np.ndarray:
        """Sample rewards."""
        return np.array(self.reward_distribution_sampler(n)).reshape((n,))

    @staticmethod
    def _validate_parameters(params: dict, verbose: bool = False) -> None:
        """Validate provided parameters."""
        # Implement validation as needed
        pass

    @staticmethod
    def _rng_init(params: dict) -> np.random.Generator:
        """Initialize the random number generator."""
        seed = params.get("seed", 1729)
        return np.random.default_rng(seed)

    @staticmethod
    def _volume_sampler_init(params: dict) -> Callable[[], int]:
        """Initialize the volume sampler."""
        param_volume_sampler = params.get("volume_sampler")
        if param_volume_sampler:
            return param_volume_sampler
        else:
            vol = params.get("volume", 1000)
            std = params.get("volume_std", 100)
            return nonneg_int_normal_sampler(vol, std)

    @staticmethod
    def _buyside_ctr_init(params: dict) -> float:
        """Initialize buyside CTR."""
        param_buyside_ctr = params.get("buyside_ctr")
        if param_buyside_ctr is not None:
            return max(0.0, min(1.0, float(param_buyside_ctr)))
        else:
            return 0.05  # Default value

    @staticmethod
    def _sellside_paid_ctr_init(params: dict) -> float:
        """Initialize sellside paid CTR."""
        param_sellside_paid_ctr = params.get("sellside_paid_ctr")
        if param_sellside_paid_ctr is not None:
            return max(0.0, min(1.0, float(param_sellside_paid_ctr)))
        else:
            return 0.36  # Default value

    @staticmethod
    def _reward_distribution_sampler_init(params: dict) -> Callable[[int], Iterable[float]]:
        """Initialize the reward distribution sampler."""
        param_reward_distribution_sampler = params.get("reward_distribution_sampler")
        if param_reward_distribution_sampler:
            return param_reward_distribution_sampler
        else:
            mean_revenue = params.get("mean_revenue", 5.0)
            std_revenue = params.get("std_revenue", 1.0)
            return lambda n: np.around(np.maximum(np.random.normal(mean_revenue, std_revenue, n), 2), 2).tolist()

class ExplicitKeyword(Keyword):
    """
    A Keyword with an explicit model of auction relationships.
    """
    def __init__(self, params: dict = {}, verbose: bool = False) -> None:
        """Initialize ExplicitKeyword with given parameters."""
        super().__init__(params, verbose)
        self.impression_rate = self._impression_rate_init(params)
        self.cost_per_buyside_click = self._cost_per_buyside_click_init(params)

    def auction(self, bid: float, num_auctions: int) -> tuple:
        """Simulate auctions using an explicit impression and cost model."""
        impressions = binomial_impressions(num_auctions, self.impression_rate(bid))
        costs = self.cost_per_buyside_click(bid, impressions)
        placements = [0] * impressions  # Placeholder for placements
        return impressions, placements, costs

    @staticmethod
    def _impression_rate_init(params: dict) -> Callable[[float], float]:
        """Initialize the impression rate function."""
        param_impression_rate = params.get("impression_rate")
        if param_impression_rate:
            return param_impression_rate
        else:
            def thresholded_sigmoid(x: float) -> float:
                return threshold_sigmoid(x, params)
            return thresholded_sigmoid

    @staticmethod
    def _cost_per_buyside_click_init(params: dict) -> Callable[[float, int], np.ndarray]:
        """Initialize the cost per buyside click function."""
        param_cost_per_buyside_click = params.get("cost_per_buyside_click")
        if param_cost_per_buyside_click:
            return param_cost_per_buyside_click
        else:
            def default_cost(bid: float, n: int) -> np.ndarray:
                return cost_create(bid, n)
            return default_cost

class ImplicitKeyword(Keyword):
    """
    A Keyword with an implicit model of the bid->impression_share and bid->cost relationships.
    """
    def __init__(self, params: dict = {}, verbose: bool = False) -> None:
        """Initialize ImplicitKeyword with given parameters."""
        super().__init__(params, verbose)
        self.bidder_distribution = self._bidder_distribution_init(params)
        self.bid_distribution = self._bid_distribution_init(params)

    def auction(self, bid: float, num_auctions: int, n_winners: int = 1) -> tuple:
        """Simulate auctions using an implicit nth-price auction model."""
        other_bids = self.bid_distribution(self.bidder_distribution(), num_auctions).T  # Shape: (num_auctions, num_bidders)
        impressions, placements, costs = nth_price_auction(bid, other_bids, n=2, num_winners=n_winners)
        return impressions, placements, costs

    @staticmethod
    def _bidder_distribution_init(params: dict) -> Callable[[], int]:
        """Initialize the bidder distribution sampler."""
        param_bidder_distribution = params.get("bidder_distribution")
        if param_bidder_distribution:
            return param_bidder_distribution
        else:
            max_bidders = params.get("max_bidders", 30)
            participation_rate = params.get("participation_rate", 0.6)
            def sample_binomial() -> int:
                return np.random.binomial(max_bidders, participation_rate)
            return sample_binomial

    @staticmethod
    def _bid_distribution_init(params: dict) -> Callable[[int, int], np.ndarray]:
        """Initialize the bid distribution sampler."""
        param_bid_distribution = params.get("bid_distribution")
        if param_bid_distribution:
            return param_bid_distribution
        else:
            bid_loc = params.get("bid_loc", 0.0)
            bid_scale = params.get("bid_scale", 0.1)
            rng = np.random.default_rng()
            def sample_laplacian(s: int, n: int) -> np.ndarray:
                return np.around(
                    np.maximum(np.abs(rng.laplace(bid_loc, bid_scale, (s, n))), 0.0).astype(float),
                    2
                )
            return sample_laplacian

def nth_price_auction(
    bid: float,
    other_bids: np.ndarray,  # Shape: (num_auctions, num_bidders)
    n: int = 2,  # >= 1
    num_winners: int = 1,
) -> tuple:
    """
    Simulate a nth price auction.

    Parameters:
        bid (float): The bid that the agent is submitting.
        other_bids (np.ndarray): Array of other bidders' bids with shape (num_auctions, num_bidders).
        n (int): The nth price (e.g., n=2 for second-price).
        num_winners (int): Number of winners (ads that win).

    Returns:
        tuple: (impressions, placements, costs)
            - impressions (int): Number of impressions (auctions won).
            - placements (list): Placement indices.
            - costs (list): Costs paid per impression.
    """
    num_auctions, num_bidders = other_bids.shape
    impressions = 0
    placements = []
    costs = []

    for auction_bids in other_bids:
        # Combine agent's bid with other bids
        all_bids = np.append(auction_bids, bid)
        # Sort bids in descending order
        sorted_indices = np.argsort(-all_bids)
        # Find agent's position
        agent_position = np.where(sorted_indices == num_bidders)[0][0]

        # Check if agent is among the top 'num_winners'
        if agent_position < num_winners:
            impressions += 1
            placements.append(agent_position)
            # Determine the nth price
            if (agent_position + n) < len(all_bids):
                cost = all_bids[agent_position + n]
            else:
                cost = all_bids[-1] if len(all_bids) >= n else 0.0
            costs.append(cost)

    return impressions, placements, costs

def probify(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Clamp the input to [0, 1]."""
    return np.clip(x, 0.0, 1.0).astype(type(x))

def nonnegify(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Ensure the input is non-negative."""
    return np.maximum(x, 0.0).astype(type(x))

def bid_abs_laplace(
    bid_loc: float,
    scale: float,
    rng: np.random.Generator,
    lowest_bid: float = 0.0
) -> Callable[[int, int], np.ndarray]:
    """
    Create a sampler for absolute Laplace-distributed bids.

    Parameters:
        bid_loc (float): Location parameter of the Laplace distribution.
        scale (float): Scale parameter of the Laplace distribution.
        rng (np.random.Generator): Random number generator.
        lowest_bid (float): Minimum bid value.

    Returns:
        Callable[[int, int], np.ndarray]: Function to sample bids.
    """
    def sampler(s: int, n: int) -> np.ndarray:
        return np.around(
            np.maximum(np.abs(rng.laplace(bid_loc, scale, (s, n))), lowest_bid).astype(float),
            2
        )
    return sampler

class BiddingOutcomes(dict):
    """
    A dictionary-like structure to hold bidding outcomes.

    Attributes:
        bid (float): The bid used.
        impressions (int): Number of impressions.
        impression_share (float): Proportion of total possible impressions observed.
        buyside_clicks (int): Number of buyside clicks.
        costs (list): Costs per buyside click.
        sellside_conversions (int): Number of sellside conversions.
        revenues (list): Revenues from sellside conversions.
        revenues_per_cost (list): Revenues corresponding to each cost.
        profit (float): Total profit.
    """
    def __init__(self, *args, **kwargs):
        super(BiddingOutcomes, self).__init__(*args, **kwargs)

def simulate_epoch_of_bidding(
    keyword: Keyword,
    bid: float,
    budget: float = float("inf"),
    n_auctions: Optional[int] = None,
) -> BiddingOutcomes:
    """
    Simulate bidding at a fixed price for a given day or number of auctions.

    Parameters:
        keyword (Keyword): The keyword being bid on.
        bid (float): The bid amount.
        budget (float): The budget constraint.
        n_auctions (Optional[int]): Number of auctions to simulate.

    Returns:
        BiddingOutcomes: The outcomes of the bidding simulation.
    """
    volume = n_auctions if n_auctions is not None else int(keyword.sample_volume()[0])

    outcomes = BiddingOutcomes({
        "bid": bid,
        "impressions": 0,
        "impression_share": 0.0,
        "buyside_clicks": 0,
        "costs": [],
        "sellside_conversions": 0,
        "revenues": [],
        "revenues_per_cost": [],
        "profit": 0.0,
    })

    impressions, placements, click_costs = keyword.auction(bid, num_auctions=volume)
    outcomes["impressions"] = impressions
    outcomes["impression_share"] = (impressions / volume) if volume > 0 else 0.0

    # Sample buyside clicks
    buyside_clicks = keyword.sample_buyside_click(impressions)
    for clicked, cost in zip(buyside_clicks, click_costs):
        if clicked:
            if budget >= cost:
                outcomes["buyside_clicks"] += 1
                outcomes["costs"].append(cost)
                budget -= cost
            else:
                break

    # Sample sellside conversions
    sellside_clicks = keyword.sample_sellside_paid_click(outcomes["buyside_clicks"])
    outcomes["sellside_conversions"] = sum_array_bool(sellside_clicks)

    # Sample revenues
    revenues = keyword.sample_reward(outcomes["sellside_conversions"])
    outcomes["revenues"] = revenues.tolist()

    # Calculate revenues_per_cost
    revenues_per_cost = list_to_zeros(outcomes["costs"])
    revenues_per_cost[:len(revenues)] = revenues  # Align revenues with costs
    outcomes["revenues_per_cost"] = revenues_per_cost.tolist()

    # Calculate profit
    total_revenue = sum_array(np.array(outcomes["revenues"]))
    total_cost = sum_list(outcomes["costs"])
    outcomes["profit"] = total_revenue - total_cost

    return outcomes

def combine_outcomes(*outcomes: BiddingOutcomes) -> BiddingOutcomes:
    """
    Combine multiple bidding outcomes into a single outcome.

    Parameters:
        *outcomes (BiddingOutcomes): Multiple bidding outcomes.

    Returns:
        BiddingOutcomes: The combined bidding outcomes.
    """
    result = BiddingOutcomes(outcomes[0])
    addable_fields = ["impressions", "buyside_clicks", "sellside_conversions", "profit"]
    catable_fields = ["costs", "revenues", "revenues_per_cost"]

    for outcome in outcomes[1:]:
        for field in addable_fields:
            result[field] += outcome[field]
        for field in catable_fields:
            result[field] = np.concatenate((result[field], outcome[field]))
        if result["impression_share"] > 0:
            total_volume = result["impressions"] / result["impression_share"]
        else:
            total_volume = 0.0
        if outcome["impression_share"] > 0:
            total_volume += outcome["impression_share"] * (len(outcome["costs"]) + len(outcome["revenues"]))
        else:
            total_volume += 0.0
        result["impression_share"] = (result["impressions"] / total_volume) if total_volume > 0 else 0.0

    return result

def uniform_get_auctions_per_timestep(
    timesteps: int, *kws: Keyword
) -> Iterable[Iterable[int]]:
    """
    Distribute auctions uniformly across timesteps.

    Parameters:
        timesteps (int): Number of timesteps.
        *kws (Keyword): Keywords involved in the campaign.

    Returns:
        Iterable[Iterable[int]]: Auctions per timestep per keyword.
    """
    volumes = [int(kw.sample_volume()[0]) for kw in kws]
    volume_step = [vol // timesteps for vol in volumes]
    auctions_per_timestep = []
    for t in range(timesteps):
        timestep_auctions = []
        for i, vol in enumerate(volumes):
            if t == timesteps - 1:
                auctions = vol - volume_step[i] * (timesteps - 1)
            else:
                auctions = volume_step[i]
            timestep_auctions.append(auctions)
        auctions_per_timestep.append(timestep_auctions)
    return auctions_per_timestep

def simulate_epoch_of_bidding_on_campaign(
    keywords: Iterable[Keyword],
    bids: Iterable[float],
    budget: float = float("inf"),
    auctions_per_timestep: Optional[Iterable[Iterable[int]]] = None,
) -> list:
    """
    Simulate bidding across multiple keywords over an epoch.

    Parameters:
        keywords (Iterable[Keyword]): Keywords in the campaign.
        bids (Iterable[float]): Bids corresponding to each keyword.
        budget (float): Total budget for bidding.
        auctions_per_timestep (Optional[Iterable[Iterable[int]]]): Auctions per timestep per keyword.

    Returns:
        list: Outcomes for each keyword.
    """
    keywords = list(keywords)
    outcomes_per_keyword = [
        BiddingOutcomes({
            "bid": bids[i],
            "impressions": 0,
            "impression_share": 0.0,
            "buyside_clicks": 0,
            "costs": [],
            "sellside_conversions": 0,
            "revenues": [],
            "revenues_per_cost": [],
            "profit": 0.0,
        })
        for i in range(len(keywords))
    ]

    if auctions_per_timestep is None:
        auctions_per_timestep = uniform_get_auctions_per_timestep(24, *keywords)

    remaining_budget = budget

    for timestep_auctions in auctions_per_timestep:
        for kw_index, (keyword, bid) in enumerate(zip(keywords, bids)):
            auctions = timestep_auctions[kw_index]
            new_outcomes = simulate_epoch_of_bidding(
                keyword=keyword,
                bid=bid,
                budget=remaining_budget,
                n_auctions=auctions,
            )
            total_cost = sum_list(new_outcomes["costs"])
            remaining_budget -= total_cost
            outcomes_per_keyword[kw_index] = combine_outcomes(outcomes_per_keyword[kw_index], new_outcomes)
            if remaining_budget <= 0:
                break
        if remaining_budget <= 0:
            break

    return outcomes_per_keyword

# --------------------------- Gymnasium Environment --------------------------- #

class BiddingSimulation(gym.Env):
    """
    Gymnasium environment for keyword auctions.

    The environment simulates bidding on keywords with a fixed or dynamic budget over multiple timesteps.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        keyword_config: Optional[dict] = None,
        num_keywords: int = 10,
        budget: float = 1000.0,
        render_mode: Optional[str] = None,
        loss_threshold: float = 10000.0,
        max_days: int = 60,
        updater_params: list = [["vol", 0.03], ["ctr", 0.03], ["cvr", 0.03]],
        updater_mask: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the BiddingSimulation environment.

        Parameters:
            keyword_config (Optional[dict]): Configuration for keyword sampling.
            num_keywords (int): Number of keywords in the simulation.
            budget (float): Budget for bidding.
            render_mode (Optional[str]): Rendering mode.
            loss_threshold (float): Threshold for cumulative loss to truncate the episode.
            max_days (int): Maximum number of days (timesteps) for bidding.
            updater_params (list): Parameters for updating keywords.
            updater_mask (Optional[list]): Mask to determine which keywords to update.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.keyword_config = keyword_config
        self.num_keywords = num_keywords
        self.budget = budget
        self.max_days = max_days
        self.loss_threshold = loss_threshold
        self.updater_params = updater_params
        self.updater_mask = updater_mask
        self.render_mode = render_mode if render_mode in self.metadata["render_modes"] else None

        # Define action and observation spaces
        self.action_space = self.get_action_space(self.num_keywords)
        self.observation_space = self.get_observation_space(self.num_keywords, self.budget)

        # Initialize environment state
        self.keywords: list = []
        self.keyword_params: list = []
        self._have_keywords = False
        self.current_day = 0
        self.cumulative_profit = 0.0
        self._current_text = "New start\n"

    @staticmethod
    def get_action_space(num_keywords: int) -> spaces.Dict:
        """Define the action space."""
        return spaces.Dict({
            "keyword_bids": spaces.Box(
                low=0.01, high=10.0, shape=(num_keywords,), dtype=np.float32
            ),
            "budget": spaces.Box(low=0.01, high=10000.0, shape=(1,), dtype=np.float32),
        })

    @staticmethod
    def get_observation_space(num_keywords: int, budget: float) -> spaces.Dict:
        """Define the observation space."""
        NonnegativeIntBox = spaces.Box(low=0, high=np.inf, shape=(num_keywords,), dtype=np.int32)
        CostSpace = spaces.Box(low=0, high=budget, shape=(num_keywords,), dtype=np.float32)
        NonnegativeFloatBox = spaces.Box(
            low=0.0, high=np.inf, shape=(num_keywords,), dtype=np.float32
        )
        return spaces.Dict({
            "impressions": NonnegativeIntBox,
            "buyside_clicks": NonnegativeIntBox,
            "cost": CostSpace,
            "sellside_conversions": NonnegativeIntBox,
            "revenue": NonnegativeFloatBox,
            "cumulative_profit": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            "days_passed": spaces.Box(
                low=0, high=np.inf, shape=(1,), dtype=np.int32
            ),
        })

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple:
        """
        Reset the environment to an initial state.
        """
        import gym
        try:
            reset_kwargs = {}
            if seed is not None:
                reset_kwargs['seed'] = seed
            if options is not None:
                reset_kwargs['options'] = options
            super().reset(**reset_kwargs)
        except TypeError:
            # Fallback if 'seed' is not supported
            if options is not None:
                super().reset(**{'options': options})
            else:
                super().reset()

        # Initialize environment state
        self.current_day = 0
        self.cumulative_profit = 0.0
        self._current_text = "Reset environment\n\nNew start\n"

        # Create initial observation
        observations = {
            "impressions": np.zeros(self.num_keywords, dtype=np.int32),
            "buyside_clicks": np.zeros(self.num_keywords, dtype=np.int32),
            "cost": np.zeros(self.num_keywords, dtype=np.float32),
            "sellside_conversions": np.zeros(self.num_keywords, dtype=np.int32),
            "revenue": np.zeros(self.num_keywords, dtype=np.float32),
            "cumulative_profit": np.array([self.cumulative_profit], dtype=np.float32),
            "days_passed": np.array([self.current_day], dtype=np.int32),
        }

        info = {"keyword_params": self.repr_all_params(self.keyword_params)}

        return observations, info

    def step(self, action: dict) -> tuple:
        """
        Take a step in the environment.

        Parameters:
            action (dict): Action dictionary containing 'keyword_bids' and 'budget'.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        assert self._have_keywords, "Environment must be reset before stepping."

        # Extract action components
        budget_array = action.get("budget", np.array([self.budget], dtype=np.float32))
        bid_array = action.get("keyword_bids", np.array([0.0]*self.num_keywords, dtype=np.float32))
        self.budget = float(np.round(budget_array[0], 2))

        # Ensure bid_array is an array, not a scalar
        if isinstance(bid_array, (int, float)):
            bid_array = np.full(self.num_keywords, bid_array, dtype=np.float32)

        keywords = []
        bids = []
        for bid, keyword in zip(bid_array, self.keywords):
            bid = max(0.01, round(float(bid), 2))
            keywords.append(keyword)
            bids.append(bid)

        # Simulate bidding on the campaign
        bidding_outcomes = simulate_epoch_of_bidding_on_campaign(
            keywords=keywords,
            bids=bids,
            budget=self.budget,
        )

        # Calculate profits
        profits = sum([kw["profit"] for kw in bidding_outcomes])
        self.cumulative_profit += profits

        # Check for truncation
        truncated = self.cumulative_profit < -self.loss_threshold

        # Update day count
        self.current_day += 1
        terminated = self.current_day >= self.max_days

        # Calculate reward with reward shaping
        # Here, the reward is directly the profit. You can modify this as needed for better shaping.
        reward = profits

        # Create observation
        observations = {
            "impressions": np.array([kw["impressions"] for kw in bidding_outcomes], dtype=np.int32),
            "buyside_clicks": np.array([kw["buyside_clicks"] for kw in bidding_outcomes], dtype=np.int32),
            "cost": np.array([sum_list(kw["costs"]) for kw in bidding_outcomes], dtype=np.float32),
            "sellside_conversions": np.array([kw["sellside_conversions"] for kw in bidding_outcomes], dtype=np.int32),
            "revenue": np.array([sum_array(np.array(kw["revenues"])) for kw in bidding_outcomes], dtype=np.float32),
            "cumulative_profit": np.array([self.cumulative_profit], dtype=np.float32),
            "days_passed": np.array([self.current_day], dtype=np.int32),
        }

        # Update keywords if necessary (e.g., dynamic keyword parameters)
        self.update_keywords()

        # Create info dictionary
        info = {
            "bids": bids,
            "bidding_outcomes": repr_outcomes_py(bidding_outcomes),
            "keyword_params": self.repr_all_params(self.keyword_params),
        }

        # Update render text
        if self.render_mode == "ansi":
            self._current_text = (
                f"Day: {self.current_day}/{self.max_days},   "
                f"Average profit per kw in day: {profits / self.num_keywords:.2f},   "
                f"Budget: {self.budget}   "
                f"Total profit in day: {profits:.2f},   "
                f"Cumulative profit: {self.cumulative_profit:.2f}\n"
            )
            if truncated:
                self._current_text += (
                    "Bidding simulation truncated early, we spent too much.\n"
                    f"Our allowed spend was ({self.loss_threshold:.2f}),\n"
                    f"but our cumulative loss was ({self.cumulative_profit:.2f})"
                )

        return observations, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """
        Render the environment.

        Returns:
            Optional[str]: The rendered text if 'ansi' mode is active.
        """
        if self.render_mode == "ansi":
            return self._current_text

    def close(self):
        """Close the environment."""
        pass

    def update_keywords(self) -> None:
        """
        Update keyword parameters based on updater_mask and updater_params.

        This is a placeholder function. Implement dynamic updates as needed.
        """
        # Example: Adjust CTRs based on updater_params
        # Implement dynamic updates as required
        pass

    @staticmethod
    def sample_random_keywords(
        num_keywords: int, rng: np.random.Generator
    ) -> tuple:
        """
        Sample random keywords for the simulation.

        Parameters:
            num_keywords (int): Number of keywords to sample.
            rng (np.random.Generator): Random number generator.

        Returns:
            tuple: (List of keywords, List of their parameters)
        """
        keywords = []
        params_list = []
        for _ in range(num_keywords):
            # Randomly choose between ExplicitKeyword and ImplicitKeyword
            keyword_type = rng.choice(['explicit', 'implicit'])
            if keyword_type == 'explicit':
                keyword = ExplicitKeyword()
                params = {
                    "impression_rate": keyword.impression_rate,
                    "buyside_ctr": keyword.buyside_ctr,
                    "sellside_paid_ctr": keyword.sellside_paid_ctr,
                    "reward_distribution_sampler": keyword.reward_distribution_sampler
                }
            else:
                keyword = ImplicitKeyword()
                params = {
                    "bidder_distribution": keyword.bidder_distribution,
                    "bid_distribution": keyword.bid_distribution,
                    "buyside_ctr": keyword.buyside_ctr,
                    "sellside_paid_ctr": keyword.sellside_paid_ctr,
                    "reward_distribution_sampler": keyword.reward_distribution_sampler
                }
            keywords.append(keyword)
            params_list.append(params)
        return keywords, params_list

    @staticmethod
    def repr_all_params(params_list: list) -> str:
        """Represent all keyword parameters as a string."""
        return "\n".join([f"kw{n} params: {params}" for n, params in enumerate(params_list)])

# --------------------------- Utility Functions --------------------------- #

def flatten_observation(observation):
    """
    Flatten the observation dictionary into a single numpy array.
    """
    flattened = []
    for key in sorted(observation.keys()):
        flattened.append(observation[key].flatten())
    return np.concatenate(flattened)

# --------------------------- DualVAE Model Definitions with SCM Integration --------------------------- #

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

class StructuralEquation(nn.Module):
    """
    Structural equation modeling how z is generated from s.
    """
    def __init__(self, s_dim, z_dim, hidden_dims=[256, 256], activation=nn.ReLU()):
        super(StructuralEquation, self).__init__()
        self.model = MLPNetwork(s_dim, z_dim, hidden_dims, activation=activation)

    def forward(self, s):
        return self.model(s)

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

class DualVAEModel(nn.Module):
    """
    DualVAE model with SCM integration and inference encoder for z_x from S_t.
    """
    def __init__(self, state_dim, action_dim, reward_dim=1, z_dim=4, s_dim=4, hidden_dims=[256, 256], activation=nn.ReLU(), batch_norm=False):
        super(DualVAEModel, self).__init__()
        self.z_dim = z_dim
        self.s_dim = s_dim
        self.state_dim = state_dim  # Store state_dim as an instance variable
        self.action_dim = action_dim
        self.reward_dim = reward_dim

        input_dim_x = state_dim + action_dim  # x = [S_t, A_t]
        input_dim_y = reward_dim + state_dim  # y = [R_t, S_{t+1}]

        # Shared confounder encoder
        self.encoder_s = Encoder(input_dim_x + input_dim_y, s_dim, hidden_dims, batch_norm=batch_norm, activation=activation)

        # Structural equations for z_x and z_y
        self.structural_eq_zx = StructuralEquation(s_dim, z_dim, hidden_dims, activation=activation)
        self.structural_eq_zy = StructuralEquation(s_dim, z_dim, hidden_dims, activation=activation)

        # Residual encoders for z_x and z_y
        self.encoder_residual_x = Encoder(input_dim_x, z_dim, hidden_dims, batch_norm=batch_norm, activation=activation)
        self.encoder_residual_y = Encoder(input_dim_y, z_dim, hidden_dims, batch_norm=batch_norm, activation=activation)

        # Inference encoder to map S_t to z_x
        self.inference_encoder_zx = Encoder(state_dim, z_dim, hidden_dims, batch_norm=batch_norm, activation=activation)

        # Decoders
        self.decoder_X = Decoder(z_dim + s_dim, input_dim_x, hidden_dims, batch_norm=batch_norm, activation=activation)
        self.decoder_Y = Decoder(z_dim + s_dim, input_dim_y, hidden_dims, batch_norm=batch_norm, activation=activation)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar) + 1e-7
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        """
        Forward pass through DualVAE with SCM integration.
        """
        # Concatenate x and y to encode s
        xy = torch.cat([x, y], dim=1)
        s_mu, s_logvar = self.encoder_s(xy)
        s = self.reparameterize(s_mu, s_logvar)

        # Structural equations for z_x and z_y
        z_x_mu_structural = self.structural_eq_zx(s)
        z_y_mu_structural = self.structural_eq_zy(s)

        # Residual encoders for z_x and z_y
        z_x_res_mu, z_x_res_logvar = self.encoder_residual_x(x)
        z_y_res_mu, z_y_res_logvar = self.encoder_residual_y(y)

        # Combine structural and residual parts
        z_x_mu = z_x_mu_structural + z_x_res_mu
        z_y_mu = z_y_mu_structural + z_y_res_mu

        z_x_logvar = z_x_res_logvar  # Assume structural part has no variance
        z_y_logvar = z_y_res_logvar

        z_x = self.reparameterize(z_x_mu, z_x_logvar)
        z_y = self.reparameterize(z_y_mu, z_y_logvar)

        # Decode x and y
        recon_X = self.decoder_X(torch.cat([z_x, s], dim=1))
        recon_Y = self.decoder_Y(torch.cat([z_y, s], dim=1))

        # Inference z_x from S_t
        z_x_infer_mu, z_x_infer_logvar = self.inference_encoder_zx(x[:, :self.state_dim])
        z_x_infer = self.reparameterize(z_x_infer_mu, z_x_infer_logvar)

        # Store outputs
        mean = {
            "s": s_mu,
            "z_x": z_x_mu,
            "z_y": z_y_mu,
            "z_x_infer": z_x_infer_mu
        }
        logvar = {
            "s": s_logvar,
            "z_x": z_x_logvar,
            "z_y": z_y_logvar,
            "z_x_infer": z_x_infer_logvar
        }
        sample = {
            "s": s,
            "z_x": z_x,
            "z_y": z_y,
            "z_x_infer": z_x_infer
        }
        recon = {
            "x": recon_X,
            "y": recon_Y
        }
        output = VAEOutput(mean, logvar, sample, recon)

        return output

    def infer_z_x(self, state):
        """
        Infer z_x from state using the inference encoder.
        """
        z_x_mu, _ = self.inference_encoder_zx(state)
        return z_x_mu

# --------------------------- DualVAE Training and Evaluation Functions --------------------------- #

def dualvae_loss(x, y, output, weights):
    """
    Computes the DualVAE loss components with SCM integration.
    """
    # Reconstruction losses
    recon_X_loss = nn.functional.mse_loss(output.recon["x"], x, reduction='sum')
    recon_Y_loss = nn.functional.mse_loss(output.recon["y"], y, reduction='sum')

    # KL Divergence losses
    kld_s = kl_divergence_loss(output.mean["s"], output.logvar["s"])
    kld_zx = kl_divergence_loss(output.mean["z_x"], output.logvar["z_x"])
    kld_zy = kl_divergence_loss(output.mean["z_y"], output.logvar["z_y"])
    kld_total = kld_s + kld_zx + kld_zy

    # Orthogonality and Conditional Independence losses
    ortho_loss_zx_s = orthogonality_loss(output.sample["z_x"], output.sample["s"])
    ortho_loss_zy_s = orthogonality_loss(output.sample["z_y"], output.sample["s"])
    ci_loss = conditional_independence_loss(output.sample["z_x"], output.sample["z_y"], output.sample["s"])

    # Inference z_x loss
    inference_zx_loss = nn.functional.mse_loss(output.sample["z_x_infer"], output.sample["z_x"], reduction='sum')

    total_loss = (weights[0] * recon_X_loss +
                  weights[1] * recon_Y_loss +
                  weights[2] * kld_total +
                  weights[3] * ortho_loss_zx_s +
                  weights[4] * ortho_loss_zy_s +
                  weights[5] * ci_loss +
                  weights[6] * inference_zx_loss)

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

def train_dualvae(model, weights, dataset, device, num_epochs=100, batch_size=128, learning_rate=1e-3, run_id=1):
    """
    Trains the DualVAE model and logs metrics to wandb.
    """
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
            # Flatten observations
            x_batch = torch.FloatTensor([flatten_observation(obs.numpy()) for obs in x_batch]).to(device)
            y_batch = torch.FloatTensor([flatten_observation(obs.numpy()) for obs in y_batch]).to(device)
            output = model(x_batch, y_batch)
            loss_dict = dualvae_loss(x_batch, y_batch, output, weights)
            loss = loss_dict["total_loss"]
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader.dataset)
        training_losses.append(avg_loss)

        # Log metrics to wandb
        wandb.log({
            "DualVAE/Epoch": epoch + 1,
            "DualVAE/Total Loss": avg_loss,
            "DualVAE/Reconstruction X Loss": loss_dict["recon_X_loss"].item() / len(dataloader.dataset),
            "DualVAE/Reconstruction Y Loss": loss_dict["recon_Y_loss"].item() / len(dataloader.dataset),
            "DualVAE/KL Divergence": loss_dict["kld_total"].item() / len(dataloader.dataset),
            "DualVAE/Orthogonality ZX-S": loss_dict["ortho_loss_zx_s"].item() / len(dataloader.dataset),
            "DualVAE/Orthogonality ZY-S": loss_dict["ortho_loss_zy_s"].item() / len(dataloader.dataset),
            "DualVAE/Conditional Independence Loss": loss_dict["ci_loss"].item() / len(dataloader.dataset),
            "DualVAE/Inference ZX Loss": loss_dict["inference_zx_loss"].item() / len(dataloader.dataset)
        })

        # Scheduler step
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Run {run_id}: Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

    # Plot training loss
    plt.figure(figsize=(10, 5))
    sns.set(style="whitegrid")
    plt.plot(range(1, num_epochs + 1), training_losses, label='DualVAE Training Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DualVAE Training Progress')
    plt.legend()
    plt.tight_layout()
    plot_path = f"DualVAE_Training_Run_{run_id}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    wandb.log({"DualVAE_Training_Progress": wandb.Image(plot_path)})

    return model

# --------------------------- PPO Agent Definitions --------------------------- #

# (No changes needed here unless the observation space affects PPO Agents)

# --------------------------- Memory Class Definitions --------------------------- #

# (No changes needed here)

# --------------------------- Fixed Policies Definitions --------------------------- #

# (No changes needed here)

# --------------------------- Function to Run Fixed Policies --------------------------- #

# (No changes needed here)

# --------------------------- ATE Estimation Plotting Function --------------------------- #

def plot_ate_error(ate_error_summary, confounder_levels, run_id):
    """
    Plots the ATE estimation error against confounder intensity for each agent.
    """
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid", font_scale=1.2)
    
    agents = list(ate_error_summary.keys())
    confounder_intensities = list(confounder_levels.keys())  # e.g., 'Low', 'Medium', 'High'
    intensity_values = range(len(confounder_intensities))  # Numerical representation for plotting

    for agent in agents:
        errors_mean = ate_error_summary[agent]["error_mean"]
        errors_std = ate_error_summary[agent]["error_std"]
        plt.plot(intensity_values, errors_mean, label=agent, marker='o')
        plt.fill_between(intensity_values, 
                         np.array(errors_mean) - np.array(errors_std), 
                         np.array(errors_mean) + np.array(errors_std), 
                         alpha=0.2)

    plt.xticks(intensity_values, confounder_intensities)
    plt.xlabel('Confounder Intensity Level')
    plt.ylabel('ATE Estimation Error')
    plt.title(f'Run {run_id}: ATE Estimation Error vs. Confounder Intensity')
    plt.legend()
    plt.tight_layout()
    plot_path = f"ATE_Estimation_Error_Run_{run_id}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    wandb.log({"ATE Estimation Error": wandb.Image(plot_path)})
    print(f"ATE Estimation Error plot saved: {plot_path}")

# --------------------------- Run Experiment Function --------------------------- #

def run_experiment(run_id, config, confounder_level, confounder_amplitudes):
    """
    Runs a single experiment including:
    1. Training Baseline PPO Agent
    2. Training Baseline Causal PPO Agent
    3. Collecting Data for DualVAE
    4. Training DualVAE
    5. Computing Causal Index
    6. Training Causal-enhanced PPO Agent
    7. Comparing Performances
    8. Measuring Compute Flows
    9. Estimating ATE and Plotting ATE Error
    Returns:
        baseline_rewards: List of rewards per episode for Baseline PPO
        baseline_causal_rewards: List of rewards per episode for Baseline Causal PPO
        causal_rewards: List of rewards per episode for Causal PPO
        compute_flows: Dictionary of number of parameters per agent
        ate_errors: Dictionary containing ATE estimation errors per agent
    """
    # Initialize wandb run
    wandb.init(project="SCM_DualVAE_PPO_Experiment",
               config=config,
               name=f"Run_{run_id}_Confounder_{confounder_level}",
               reinit=True)
    
    # Initialize environment with confounders
    # --------------------------- CHANGED: Replace NavigationEnv with BiddingSimulation --------------------------- #
    base_env = BiddingSimulation(
        keyword_config=config.get("keyword_config", {}),
        num_keywords=config.get("num_keywords", 10),
        budget=config.get("initial_budget", 1000.0),
        render_mode=None,
        loss_threshold=config.get("loss_threshold", 10000.0),
        max_days=config.get("max_days", 60),
    )
    env = ConfounderEnvWrapper(base_env,
                               confounder_amplitudes=confounder_amplitudes)
    # --------------------------- END CHANGE --------------------------- #
    
    # Compute state_dim based on flattened observation
    example_observation = env.reset()[0]
    flattened_observation = flatten_observation(example_observation)
    state_dim = len(flattened_observation)
    action_dim = env.action_space["keyword_bids"].shape[0]  # Number of keywords

    # --------------------------- Adjust DualVAE and PPO Agents --------------------------- #

    # The rest of the code remains mostly the same, but ensure that observations are flattened
    # and state_dim is correctly set as above.

    # --------------------------- Phase 1: Train Baseline PPO Agent --------------------------- #

    print(f"\n=== Run {run_id} | Confounder Level: {confounder_level} ===")
    print("=== Phase 1 - Training Baseline PPO Agent ===\n")

    # Initialize Baseline PPO agent
    ppo_agent_baseline = BaselinePPOAgent(state_dim=state_dim, action_dim=action_dim,
                                         lr=config["ppo_lr_baseline"],
                                         gamma=config["ppo_gamma_baseline"],
                                         K_epochs=config["ppo_K_epochs_baseline"],
                                         eps_clip=config["ppo_eps_clip_baseline"])

    # Training parameters for Baseline PPO
    max_episodes_baseline = config["ppo_max_episodes_baseline"]  # Number of episodes for training
    max_timesteps_baseline = config["ppo_max_timesteps_baseline"]    # Max timesteps per episode
    update_timestep_baseline = config["ppo_update_timestep_baseline"]  # Update PPO agent every 'update_timestep' timesteps
    timestep_baseline = 0

    # Initialize memory for Baseline PPO agent
    memory_baseline = Memory()

    # Initialize rewards tracking
    rewards_baseline = []

    # Initialize moving averages
    moving_avg_baseline = []

    # Initialize episode counters for moving averages
    moving_avg_window = config.get("moving_avg_window", 100)

    # Training loop for Baseline PPO agent
    print("Starting training of Baseline PPO agent...")
    for episode in tqdm(range(1, max_episodes_baseline + 1), desc="Baseline PPO Training", leave=False):
        observation, _ = env.reset()
        state = flatten_observation(observation)
        ep_reward = 0

        for t in range(max_timesteps_baseline):
            timestep_baseline += 1

            # Select action using Baseline PPO agent
            action, log_prob, state_value = ppo_agent_baseline.select_action(state)
            # Prepare action dictionary
            action_dict = {
                "keyword_bids": np.array(action).astype(np.float32),
                "budget": np.array([env.budget], dtype=np.float32)
            }
            next_observation, reward, done, info = env.step(action_dict)

            # Flatten next_observation
            next_state = flatten_observation(next_observation)

            # Save data in memory
            memory_baseline.states.append(state)
            memory_baseline.actions.append(action)
            memory_baseline.logprobs.append(log_prob)
            memory_baseline.state_values.append(state_value)
            memory_baseline.rewards.append(reward)
            memory_baseline.is_terminals.append(done)

            ep_reward += reward
            state = next_state

            # Update Baseline PPO agent
            if timestep_baseline % update_timestep_baseline == 0:
                ppo_agent_baseline.update_policy(memory_baseline)
                memory_baseline = Memory()

            if done:
                break

        rewards_baseline.append(ep_reward)

        # Compute moving average
        if len(rewards_baseline) >= moving_avg_window:
            moving_avg_val = np.mean(rewards_baseline[-moving_avg_window:])
            moving_avg_baseline.append(moving_avg_val)
        else:
            moving_avg_val = np.mean(rewards_baseline)
            moving_avg_baseline.append(moving_avg_val)

        # Log to wandb
        wandb.log({
            "PPO/Baseline Agent Reward": ep_reward,
            f"PPO/Baseline Agent Moving Avg ({moving_avg_window})": moving_avg_val,
            "Episode": episode
        })

        # Print progress every 50 episodes
        if episode % 50 == 0 or episode == 1:
            print(f"Run {run_id} | Confounder {confounder_level}: Episode {episode} - Baseline PPO Avg Reward (last {moving_avg_window}): {moving_avg_val:.2f}")

    print("\nBaseline PPO Agent training completed.\n")

    # --------------------------- Phase 1.1: Train Baseline Causal PPO Agent --------------------------- #

    print("=== Phase 1.1 - Training Baseline Causal PPO Agent ===\n")

    # Initialize Baseline Causal PPO agent
    ppo_agent_baseline_causal = BaselineCausalPPOAgent(causal_dim=2, action_dim=action_dim,
                                                      lr=config["ppo_lr_baseline_causal"],
                                                      gamma=config["ppo_gamma_baseline_causal"],
                                                      K_epochs=config["ppo_K_epochs_baseline_causal"],
                                                      eps_clip=config["ppo_eps_clip_baseline_causal"])

    # Training parameters for Baseline Causal PPO
    max_episodes_baseline_causal = config["ppo_max_episodes_baseline_causal"]  # Number of episodes for training
    max_timesteps_baseline_causal = config["ppo_max_timesteps_baseline_causal"]    # Max timesteps per episode
    update_timestep_baseline_causal = config["ppo_update_timestep_baseline_causal"]  # Update PPO agent every 'update_timestep' timesteps
    timestep_baseline_causal = 0

    # Initialize memory for Baseline Causal PPO agent
    memory_baseline_causal = MemoryCausal()

    # Initialize rewards tracking
    rewards_baseline_causal = []

    # Initialize moving averages
    moving_avg_baseline_causal = []

    # Training loop for Baseline Causal PPO agent
    print("Starting training of Baseline Causal PPO agent...")
    for episode in tqdm(range(1, max_episodes_baseline_causal + 1), desc="Baseline Causal PPO Training", leave=False):
        observation, _ = env.reset()
        state = flatten_observation(observation)
        # Extract causal variables: posX and posY (assuming they are part of the observation)
        # Adjust indices based on the new observation structure
        # Here, let's assume the first two keywords' 'revenue' correspond to posX and posY
        # This may need adjustment based on actual keyword configurations
        causal_vars = state[:2]  # Example: first two elements
        ep_reward = 0

        for t in range(max_timesteps_baseline_causal):
            timestep_baseline_causal += 1

            # Select action using Baseline Causal PPO agent
            action, log_prob, state_value = ppo_agent_baseline_causal.select_action(causal_vars)
            # Prepare action dictionary
            action_dict = {
                "keyword_bids": np.array(action).astype(np.float32),
                "budget": np.array([env.budget], dtype=np.float32)
            }
            next_observation, reward, done, info = env.step(action_dict)

            # Flatten next_observation
            next_state = flatten_observation(next_observation)
            # Extract causal variables
            new_causal_vars = next_state[:2]  # Example: first two elements

            # Save data in memory
            memory_baseline_causal.causal_vars.append(causal_vars)
            memory_baseline_causal.actions.append(action)
            memory_baseline_causal.logprobs.append(log_prob)
            memory_baseline_causal.state_values.append(state_value)
            memory_baseline_causal.rewards.append(reward)
            memory_baseline_causal.is_terminals.append(done)

            ep_reward += reward
            causal_vars = new_causal_vars

            # Update Baseline Causal PPO agent
            if timestep_baseline_causal % update_timestep_baseline_causal == 0:
                ppo_agent_baseline_causal.update_policy(memory_baseline_causal)
                memory_baseline_causal = MemoryCausal()

            if done:
                break

        rewards_baseline_causal.append(ep_reward)

        # Compute moving average
        if len(rewards_baseline_causal) >= moving_avg_window:
            moving_avg_val = np.mean(rewards_baseline_causal[-moving_avg_window:])
            moving_avg_baseline_causal.append(moving_avg_val)
        else:
            moving_avg_val = np.mean(rewards_baseline_causal)
            moving_avg_baseline_causal.append(moving_avg_val)

        # Log to wandb
        wandb.log({
            "PPO/Baseline Causal Agent Reward": ep_reward,
            f"PPO/Baseline Causal Agent Moving Avg ({moving_avg_window})": moving_avg_val,
            "Episode": episode
        })

        # Print progress every 50 episodes
        if episode % 50 == 0 or episode == 1:
            print(f"Run {run_id} | Confounder {confounder_level}: Episode {episode} - Baseline Causal PPO Avg Reward (last {moving_avg_window}): {moving_avg_val:.2f}")

    print("\nBaseline Causal PPO Agent training completed.\n")

    # --------------------------- Phase 2: Collect Data for DualVAE --------------------------- #

    print("=== Phase 2 - Collecting Data for DualVAE ===\n")

    # Collect data using the trained Baseline PPO agent
    num_data_steps = config["num_data_steps"]  # Number of steps for data collection
    x_data = []
    y_data = []
    observation, _ = env.reset()
    state = flatten_observation(observation)

    print("Collecting data for DualVAE training...")
    for _ in tqdm(range(num_data_steps), desc="Data Collection for DualVAE", leave=False):
        # Select action using Baseline PPO agent
        action, _, _ = ppo_agent_baseline.select_action(state)
        # Prepare action dictionary
        action_dict = {
            "keyword_bids": np.array(action).astype(np.float32),
            "budget": np.array([env.budget], dtype=np.float32)
        }
        next_observation, reward, done, info = env.step(action_dict)

        # Flatten observations
        next_state = flatten_observation(next_observation)

        # Prepare x = [S_t, A_t]
        x = np.concatenate([state, np.array(action)])
        # Prepare y = [R_t, S_{t+1}]
        y = np.concatenate([np.array([reward]), next_state])
    
        x_data.append(x)
        y_data.append(y)

        state = next_state
        if done:
            observation, _ = env.reset()
            state = flatten_observation(observation)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    print(f"Data collection completed. Collected {len(x_data)} samples.\n")

    # --------------------------- Phase 3: Train DualVAE --------------------------- #

    print("=== Phase 3 - Training DualVAE ===\n")

    # Initialize DualVAE model
    dualvae = DualVAEModel(state_dim=state_dim, action_dim=action_dim, reward_dim=1, z_dim=4, s_dim=4, hidden_dims=[256, 256], batch_norm=False).to(device)
    # Initialize weights for loss components: [recon_X, recon_Y, kld_total, ortho_zx_s, ortho_zy_s, ci_loss, inference_zx_loss]
    dualvae_loss_weights = config["dualvae_loss_weights"]

    # Prepare dataset
    dataset = TensorDataset(torch.FloatTensor(x_data), torch.FloatTensor(y_data))
    dataloader = DataLoader(dataset, batch_size=config["dualvae_batch_size"], shuffle=True)

    # Train DualVAE
    print("Starting DualVAE training...")
    dualvae = train_dualvae(dualvae, dualvae_loss_weights, dataset, device,
                            num_epochs=config["dualvae_epochs"],
                            batch_size=config["dualvae_batch_size"],
                            learning_rate=config["dualvae_learning_rate"],
                            run_id=run_id)
    print("DualVAE training completed.\n")

    # Freeze DualVAE parameters to prevent updates during PPO training
    for param in dualvae.parameters():
        param.requires_grad = False

    # --------------------------- Phase 4: Compute Causal Index --------------------------- #

    print("=== Phase 4 - Computing Causal Index ===\n")

    # Compute causal index
    causal_index = compute_causal_index(dualvae, dataloader, config["causal_var_indices"])
    wandb.log({"Causal Index": causal_index})
    print(f"Causal Index: {causal_index:.4f}\n")

    # --------------------------- Phase 5: Train Causal-enhanced PPO Agent --------------------------- #

    print("=== Phase 5 - Training Causal-enhanced PPO Agent ===\n")

    # Initialize Causal PPO agent
    ppo_agent_causal = CausalPPOAgent(z_dim=4, action_dim=action_dim,
                                      lr=config["ppo_lr_causal"],
                                      gamma=config["ppo_gamma_causal"],
                                      K_epochs=config["ppo_K_epochs_causal"],
                                      eps_clip=config["ppo_eps_clip_causal"])

    # Training parameters for Causal PPO agent
    max_episodes_causal = config["ppo_max_episodes_causal"]  # Number of episodes for training
    max_timesteps_causal = config["ppo_max_timesteps_causal"]    # Max timesteps per episode
    update_timestep_causal = config["ppo_update_timestep_causal"]  # Update PPO agent every 'update_timestep' timesteps
    timestep_causal = 0

    # Initialize memory for Causal PPO agent
    memory_causal = {'z_x': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': [], 'state_values': []}

    # Initialize rewards tracking
    rewards_causal = []

    # Initialize moving averages
    moving_avg_causal = []

    # Initialize episode counters for moving averages
    moving_avg_window = config.get("moving_avg_window", 100)

    # Training loop for Causal PPO agent
    print("Starting training of Causal-enhanced PPO agent...")
    for episode in tqdm(range(1, max_episodes_causal + 1), desc="Causal PPO Training", leave=False):
        observation, _ = env.reset()
        state = flatten_observation(observation)
        # Obtain latent representations from DualVAE inference encoder
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        dualvae.eval()
        with torch.no_grad():
            z_x_mu = dualvae.infer_z_x(state_tensor)
            z_x = z_x_mu.cpu().numpy()[0]
        ep_reward = 0

        for t in range(max_timesteps_causal):
            timestep_causal += 1

            # Select action using Causal PPO agent
            action, log_prob, state_value = ppo_agent_causal.select_action(z_x)
            # Prepare action dictionary
            action_dict = {
                "keyword_bids": np.array(action).astype(np.float32),
                "budget": np.array([env.budget], dtype=np.float32)
            }
            next_observation, reward, done, info = env.step(action_dict)

            # Flatten next_observation
            next_state = flatten_observation(next_observation)
            # Obtain latent representations for next state
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            with torch.no_grad():
                z_x_mu_next = dualvae.infer_z_x(next_state_tensor)
                z_x_next = z_x_mu_next.cpu().numpy()[0]

            # Save data in memory
            memory_causal['z_x'].append(z_x)
            memory_causal['actions'].append(action)
            memory_causal['logprobs'].append(log_prob)
            memory_causal['state_values'].append(state_value)
            memory_causal['rewards'].append(reward)
            memory_causal['is_terminals'].append(done)

            ep_reward += reward
            z_x = z_x_next

            # Update Causal PPO agent
            if timestep_causal % update_timestep_causal == 0:
                ppo_agent_causal.update_policy(memory_causal)
                memory_causal = {'z_x': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': [], 'state_values': []}

            if done:
                break

        rewards_causal.append(ep_reward)

        # Compute moving average
        if len(rewards_causal) >= moving_avg_window:
            moving_avg_val = np.mean(rewards_causal[-moving_avg_window:])
            moving_avg_causal.append(moving_avg_val)
        else:
            moving_avg_val = np.mean(rewards_causal)
            moving_avg_causal.append(moving_avg_val)

        # Log to wandb
        wandb.log({
            "PPO/Causal Agent Reward": ep_reward,
            f"PPO/Causal Agent Moving Avg ({moving_avg_window})": moving_avg_val,
            "Episode": episode
        })

        # Print progress every 50 episodes
        if episode % 50 == 0 or episode == 1:
            print(f"Run {run_id} | Confounder {confounder_level}: Episode {episode} - Causal PPO Avg Reward (last {moving_avg_window}): {moving_avg_val:.2f}")

    print("\nCausal-enhanced PPO Agent training completed.\n")

    # --------------------------- Phase 6: Compare Performances --------------------------- #

    print("=== Phase 6 - Comparing Performances ===\n")

    # Collect the rewards for this run
    run_data = {
        "PPO/Baseline": rewards_baseline,
        "PPO/Baseline Causal": rewards_baseline_causal,
        "PPO/Causal": rewards_causal
    }

    wandb.log(run_data)

    print("Performance comparison completed. Check the WandB dashboard for detailed metrics.")

    # --------------------------- Phase 7: Measure Compute Flows --------------------------- #

    print("=== Phase 7 - Measuring Compute Flows ===\n")

    # Measure number of parameters for each agent
    num_params_baseline = sum(p.numel() for p in ppo_agent_baseline.policy.parameters() if p.requires_grad)
    num_params_baseline_causal = sum(p.numel() for p in ppo_agent_baseline_causal.policy.parameters() if p.requires_grad)
    num_params_causal = sum(p.numel() for p in ppo_agent_causal.policy.parameters() if p.requires_grad)

    compute_flows = {
        "Baseline PPO": num_params_baseline,
        "Baseline Causal PPO": num_params_baseline_causal,
        "Causal PPO with DualVAE": num_params_causal
    }

    # Log compute flows to wandb
    wandb.log({
        "Compute Flows/Number of Parameters": compute_flows
    })

    print(f"Compute Flows (Number of Parameters): {compute_flows}\n")

    # --------------------------- Phase 8: Estimating ATE and Plotting ATE Error --------------------------- #

    print("=== Phase 8 - Estimating ATE and Plotting ATE Error ===\n")

    # Define number of episodes to run fixed policies for ground truth ATE
    num_episodes_ate = config.get("num_episodes_ate", 100)

    # Run Policy A (Always Up)
    avg_reward_A = run_fixed_policy(env, policy_A, num_episodes=num_episodes_ate)
    print(f"Policy A (Always Up) Average Reward: {avg_reward_A:.2f}")

    # Run Policy B (Always Down)
    avg_reward_B = run_fixed_policy(env, policy_B, num_episodes=num_episodes_ate)
    print(f"Policy B (Always Down) Average Reward: {avg_reward_B:.2f}")

    # Compute Ground Truth ATE
    ground_truth_ATE = avg_reward_A - avg_reward_B
    print(f"Ground Truth ATE (Policy A - Policy B): {ground_truth_ATE:.2f}")

    # Initialize ATE errors dictionary
    ate_errors = {
        "Baseline PPO": {"error_mean": 0.0, "error_std": 0.0},
        "Baseline Causal PPO": {"error_mean": 0.0, "error_std": 0.0},
        "Causal PPO with DualVAE": {"error_mean": 0.0, "error_std": 0.0}
    }

    # Function to estimate ATE using agent's policy (treated as Policy A)
    def estimate_ate(agent, env, dualvae, num_episodes=100, agent_type='baseline'):
        """
        Estimates ATE by treating the agent's policy as Policy A and using Policy B as Policy B.
        agent_type: 'baseline', 'baseline_causal', 'causal_dualvae'
        Returns the estimated ATE.
        """
        # Estimate Policy A using agent's policy
        rewards_policy_A = []
        for _ in range(num_episodes):
            observation, _ = env.reset()
            state = flatten_observation(observation)
            ep_reward = 0
            done = False
            while not done:
                if agent_type == 'baseline':
                    # Baseline PPO agent uses full state
                    action, _, _ = agent.select_action(state)
                elif agent_type == 'baseline_causal':
                    # Baseline Causal PPO agent uses causal variables (posX and posY)
                    causal_vars = state[:2]
                    action, _, _ = agent.select_action(causal_vars)
                elif agent_type == 'causal_dualvae':
                    # Causal PPO with DualVAE uses z_x
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    dualvae.eval()
                    with torch.no_grad():
                        z_x_mu = dualvae.infer_z_x(state_tensor)
                        z_x = z_x_mu.cpu().numpy()[0]
                    action, _, _ = agent.select_action(z_x)
                else:
                    raise ValueError(f"Unknown agent type: {agent_type}")

                # Prepare action dictionary
                action_dict = {
                    "keyword_bids": np.array(action).astype(np.float32),
                    "budget": np.array([env.budget], dtype=np.float32)
                }
                next_observation, reward, done, info = env.step(action_dict)

                # Flatten next_observation
                next_state = flatten_observation(next_observation)

                ep_reward += reward
                state = next_state
            rewards_policy_A.append(ep_reward)
        avg_reward_A_est = np.mean(rewards_policy_A)

        # Estimate Policy B using fixed Policy B
        avg_reward_B_est = run_fixed_policy(env, policy_B, num_episodes=num_episodes)

        # Estimated ATE
        estimated_ATE = avg_reward_A_est - avg_reward_B_est
        return estimated_ATE

    # Estimate ATE for Baseline PPO Agent
    estimated_ATE_baseline = estimate_ate(ppo_agent_baseline, env, dualvae, num_episodes=config.get("num_episodes_ate", 100), agent_type='baseline')
    error_baseline = abs(estimated_ATE_baseline - ground_truth_ATE)
    ate_errors["Baseline PPO"]["error_mean"] = error_baseline
    ate_errors["Baseline PPO"]["error_std"] = 0.0  # Single estimate, no std

    print(f"Baseline PPO Agent Estimated ATE: {estimated_ATE_baseline:.2f}, Error: {error_baseline:.2f}")

    # Estimate ATE for Baseline Causal PPO Agent
    estimated_ATE_baseline_causal = estimate_ate(ppo_agent_baseline_causal, env, dualvae, num_episodes=config.get("num_episodes_ate", 100), agent_type='baseline_causal')
    error_baseline_causal = abs(estimated_ATE_baseline_causal - ground_truth_ATE)
    ate_errors["Baseline Causal PPO"]["error_mean"] = error_baseline_causal
    ate_errors["Baseline Causal PPO"]["error_std"] = 0.0  # Single estimate, no std

    print(f"Baseline Causal PPO Agent Estimated ATE: {estimated_ATE_baseline_causal:.2f}, Error: {error_baseline_causal:.2f}")

    # Estimate ATE for Causal PPO with DualVAE Agent
    estimated_ATE_causal = estimate_ate(ppo_agent_causal, env, dualvae, num_episodes=config.get("num_episodes_ate", 100), agent_type='causal_dualvae')
    error_causal = abs(estimated_ATE_causal - ground_truth_ATE)
    ate_errors["Causal PPO with DualVAE"]["error_mean"] = error_causal
    ate_errors["Causal PPO with DualVAE"]["error_std"] = 0.0  # Single estimate, no std

    print(f"Causal PPO with DualVAE Agent Estimated ATE: {estimated_ATE_causal:.2f}, Error: {error_causal:.2f}")

    # Log ATE errors to wandb
    wandb.log({
        "ATE/Ground Truth ATE": ground_truth_ATE,
        "ATE/Baseline PPO Estimated ATE": estimated_ATE_baseline,
        "ATE/Baseline PPO Error": error_baseline,
        "ATE/Baseline Causal PPO Estimated ATE": estimated_ATE_baseline_causal,
        "ATE/Baseline Causal PPO Error": error_baseline_causal,
        "ATE/Causal PPO with DualVAE Estimated ATE": estimated_ATE_causal,
        "ATE/Causal PPO with DualVAE Error": error_causal
    })

    # --------------------------- Phase 9: Plot ATE Estimation Errors --------------------------- #

    # Initialize ATE errors plot summary dictionary
    ate_error_summary = {
        "Baseline PPO": {"error_mean": [error_baseline], "error_std": [0.0]},
        "Baseline Causal PPO": {"error_mean": [error_baseline_causal], "error_std": [0.0]},
        "Causal PPO with DualVAE": {"error_mean": [error_causal], "error_std": [0.0]}
    }

    # Plot ATE estimation error for this run
    plot_ate_error(ate_error_summary, {confounder_level: confounder_level}, run_id)

    # Finish wandb run
    wandb.finish()

    return rewards_baseline, rewards_baseline_causal, rewards_causal, compute_flows, ate_errors

# --------------------------- Main Function --------------------------- #

def main():
    # Configuration parameters
    config = {
        "num_data_steps": 100000,
        "dualvae_epochs": 100,
        "dualvae_batch_size": 128,
        "dualvae_learning_rate": 1e-3,
        "ppo_max_episodes_baseline": 10000,
        "ppo_max_timesteps_baseline": 200,
        "ppo_update_timestep_baseline": 4000,
        "ppo_lr_baseline": 3e-4,
        "ppo_gamma_baseline": 0.99,
        "ppo_K_epochs_baseline": 10,
        "ppo_eps_clip_baseline": 0.2,
        "ppo_max_episodes_baseline_causal": 10000,
        "ppo_max_timesteps_baseline_causal": 200,
        "ppo_update_timestep_baseline_causal": 4000,
        "ppo_lr_baseline_causal": 3e-4,
        "ppo_gamma_baseline_causal": 0.99,
        "ppo_K_epochs_baseline_causal": 10,
        "ppo_eps_clip_baseline_causal": 0.2,
        "dualvae_loss_weights": [1.0, 1.0, 0.001, 0.1, 0.1, 0.1, 1.0],
        "causal_var_indices": [0, 1],  # Adjusted based on the flattened observation
        "ppo_max_episodes_causal": 10000,
        "ppo_max_timesteps_causal": 200,
        "ppo_update_timestep_causal": 4000,
        "ppo_lr_causal": 3e-4,
        "ppo_gamma_causal": 0.99,
        "ppo_K_epochs_causal": 10,
        "ppo_eps_clip_causal": 0.2,
        "dualvae_learning_rate": 1e-3,
        "dualvae_batch_size": 128,
        "dualvae_epochs": 100,
        "moving_avg_window": 100,
        "num_episodes_ate": 100,  # Number of episodes to estimate ATE
        "num_keywords": 10,
        "initial_budget": 1000.0,
        "max_days": 60,
        "loss_threshold": 10000.0
    }

    # Define confounder levels with corresponding amplitudes
    confounder_levels = {
        "High": [0.15, 0.1, 0.1, 0.15],
        "Medium": [0.1, 0.1, 0.1, 0.1],
        "Low": [0.05, 0.05, 0.05, 0.05]
    }

    num_runs_per_level = 5  # Number of runs per confounder level
    baseline_rewards_runs = {level: [] for level in confounder_levels}
    baseline_causal_rewards_runs = {level: [] for level in confounder_levels}
    causal_rewards_runs = {level: [] for level in confounder_levels}
    compute_flows_runs = {level: [] for level in confounder_levels}
    ate_errors_runs = {level: {"Baseline PPO": [], "Baseline Causal PPO": [], "Causal PPO with DualVAE": []} for level in confounder_levels}

    # Run the experiment multiple times for each confounder level
    for level, amplitudes in confounder_levels.items():
        for run_id in range(1, num_runs_per_level + 1):
            print(f"\n========== Starting Run {run_id} for Confounder Level: {level} ==========\n")
            baseline_rewards, baseline_causal_rewards, causal_rewards, compute_flows, ate_errors = run_experiment(run_id, config, level, amplitudes)
            baseline_rewards_runs[level].append(baseline_rewards)
            baseline_causal_rewards_runs[level].append(baseline_causal_rewards)
            causal_rewards_runs[level].append(causal_rewards)
            compute_flows_runs[level].append(compute_flows)
            # Collect ATE errors
            for agent in ate_errors:
                ate_errors_runs[level][agent].append(ate_errors[agent]["error_mean"])
            print(f"========== Run {run_id} for Confounder Level: {level} Completed ==========\n")

    # Convert lists to numpy arrays for easier manipulation
    for level in confounder_levels:
        baseline_rewards_runs[level] = np.array(baseline_rewards_runs[level])  # Shape: (num_runs, episodes)
        baseline_causal_rewards_runs[level] = np.array(baseline_causal_rewards_runs[level])  # Shape: (num_runs, episodes)
        causal_rewards_runs[level] = np.array(causal_rewards_runs[level])      # Shape: (num_runs, episodes)
        ate_errors_runs[level] = np.array(ate_errors_runs[level])  # Shape: (num_runs,)

    # Compute mean and standard deviation across runs for rewards
    baseline_mean = {level: baseline_rewards_runs[level].mean(axis=0) for level in confounder_levels}
    baseline_std = {level: baseline_rewards_runs[level].std(axis=0) for level in confounder_levels}

    baseline_causal_mean = {level: baseline_causal_rewards_runs[level].mean(axis=0) for level in confounder_levels}
    baseline_causal_std = {level: baseline_causal_rewards_runs[level].std(axis=0) for level in confounder_levels}

    causal_mean = {level: causal_rewards_runs[level].mean(axis=0) for level in confounder_levels}
    causal_std = {level: causal_rewards_runs[level].std(axis=0) for level in confounder_levels}

    # Compute moving averages
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    baseline_moving_avg = {level: np.array([moving_average(run, config["moving_avg_window"]) for run in baseline_rewards_runs[level]]) for level in confounder_levels}
    baseline_causal_moving_avg = {level: np.array([moving_average(run, config["moving_avg_window"]) for run in baseline_causal_rewards_runs[level]]) for level in confounder_levels}
    causal_moving_avg = {level: np.array([moving_average(run, config["moving_avg_window"]) for run in causal_rewards_runs[level]]) for level in confounder_levels}

    # Compute mean and std of moving averages
    baseline_moving_avg_mean = {level: baseline_moving_avg[level].mean(axis=0) for level in confounder_levels}
    baseline_moving_avg_std = {level: baseline_moving_avg[level].std(axis=0) for level in confounder_levels}

    baseline_causal_moving_avg_mean = {level: baseline_causal_moving_avg[level].mean(axis=0) for level in confounder_levels}
    baseline_causal_moving_avg_std = {level: baseline_causal_moving_avg[level].std(axis=0) for level in confounder_levels}

    causal_moving_avg_mean = {level: causal_moving_avg[level].mean(axis=0) for level in confounder_levels}
    causal_moving_avg_std = {level: causal_moving_avg[level].std(axis=0) for level in confounder_levels}

    # --------------------------- Plotting Comparison with Uncertainty Intervals --------------------------- #

    print("\n=== Plotting Performance Comparison with Uncertainty Intervals ===\n")

    sns.set(style="whitegrid", font_scale=1.2)

    for level in confounder_levels:
        plt.figure(figsize=(12, 8))
        plt.title(f'Performance Comparison of PPO Agents - Confounder Level: {level}')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

        # Baseline PPO
        episodes_baseline = np.arange(1, config["ppo_max_episodes_baseline"] + 1)
        plt.plot(episodes_baseline, baseline_mean[level], label='Baseline PPO', color='orange')
        plt.fill_between(episodes_baseline, baseline_mean[level] - baseline_std[level], baseline_mean[level] + baseline_std[level],
                         color='orange', alpha=0.3)

        # Baseline Causal PPO
        episodes_baseline_causal = np.arange(1, config["ppo_max_episodes_baseline_causal"] + 1)
        plt.plot(episodes_baseline_causal, baseline_causal_mean[level], label='Baseline Causal PPO', color='purple')
        plt.fill_between(episodes_baseline_causal, baseline_causal_mean[level] - baseline_causal_std[level], baseline_causal_mean[level] + baseline_causal_std[level],
                         color='purple', alpha=0.3)

        # Causal PPO with DualVAE
        episodes_causal = np.arange(1, config["ppo_max_episodes_causal"] + 1)
        plt.plot(episodes_causal, causal_mean[level], label='Causal PPO with DualVAE', color='blue')
        plt.fill_between(episodes_causal, causal_mean[level] - causal_std[level], causal_mean[level] + causal_std[level],
                         color='blue', alpha=0.3)

        plt.legend()
        plt.tight_layout()
        plot_path = f"performance_comparison_{level}_confounder.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {plot_path}")

    # --------------------------- Aggregate Moving Averages with Uncertainty --------------------------- #

    print("\n=== Plotting Aggregate Moving Averages with Uncertainty Intervals ===\n")

    for level in confounder_levels:
        plt.figure(figsize=(12, 8))
        plt.title(f'Running Averages with Uncertainty Intervals - Confounder Level: {level}')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

        # Baseline PPO Moving Average
        moving_avg_epochs_baseline = np.arange(config["moving_avg_window"], config["ppo_max_episodes_baseline"] + 1)
        plt.plot(moving_avg_epochs_baseline, baseline_moving_avg_mean[level], label='Baseline PPO MA', color='red')
        plt.fill_between(moving_avg_epochs_baseline,
                         baseline_moving_avg_mean[level] - baseline_moving_avg_std[level],
                         baseline_moving_avg_mean[level] + baseline_moving_avg_std[level],
                         color='red', alpha=0.2)

        # Baseline Causal PPO Moving Average
        moving_avg_epochs_baseline_causal = np.arange(config["moving_avg_window"], config["ppo_max_episodes_baseline_causal"] + 1)
        plt.plot(moving_avg_epochs_baseline_causal, baseline_causal_moving_avg_mean[level], label='Baseline Causal PPO MA', color='blue')
        plt.fill_between(moving_avg_epochs_baseline_causal,
                         baseline_causal_moving_avg_mean[level] - baseline_causal_moving_avg_std[level],
                         baseline_causal_moving_avg_mean[level] + baseline_causal_moving_avg_std[level],
                         color='blue', alpha=0.2)

        # Causal PPO with DualVAE Moving Average
        moving_avg_epochs_causal = np.arange(config["moving_avg_window"], config["ppo_max_episodes_causal"] + 1)
        plt.plot(moving_avg_epochs_causal, causal_moving_avg_mean[level], label='Causal PPO with DualVAE MA', color='green')
        plt.fill_between(moving_avg_epochs_causal,
                         causal_moving_avg_mean[level] - causal_moving_avg_std[level],
                         causal_moving_avg_mean[level] + causal_moving_avg_std[level],
                         color='green', alpha=0.2)

        plt.legend()
        plt.tight_layout()
        plot_path = f"moving_average_comparison_{level}_confounder.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Moving average plot saved: {plot_path}")

    # --------------------------- Plotting ATE Estimation Errors --------------------------- #

    print("\n=== Plotting ATE Estimation Errors ===\n")

    # Prepare data for ATE Error plot
    ate_error_summary = {
        "Baseline PPO": {"error_mean": [], "error_std": []},
        "Baseline Causal PPO": {"error_mean": [], "error_std": []},
        "Causal PPO with DualVAE": {"error_mean": [], "error_std": []}
    }

    confounder_intensity_levels = list(confounder_levels.keys())  # e.g., ['High', 'Medium', 'Low']

    for level in confounder_intensity_levels:
        # Calculate mean and std of ATE errors across runs for this confounder level
        for agent in ate_error_summary:
            errors = ate_errors_runs[level][agent]
            ate_error_summary[agent]["error_mean"].append(np.mean(errors))
            ate_error_summary[agent]["error_std"].append(np.std(errors))

    # Plot ATE Estimation Error
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid", font_scale=1.2)

    agents = list(ate_error_summary.keys())
    confounder_labels = confounder_intensity_levels
    confounder_intensities_numeric = range(len(confounder_labels))  # Numerical representation for plotting

    for agent in agents:
        errors_mean = ate_error_summary[agent]["error_mean"]
        errors_std = ate_error_summary[agent]["error_std"]
        plt.plot(confounder_intensities_numeric, errors_mean, label=agent, marker='o')
        plt.fill_between(confounder_intensities_numeric, 
                         np.array(errors_mean) - np.array(errors_std), 
                         np.array(errors_mean) + np.array(errors_std), 
                         alpha=0.2)

    plt.xticks(confounder_intensities_numeric, confounder_labels)
    plt.xlabel('Confounder Intensity Level')
    plt.ylabel('ATE Estimation Error')
    plt.title('ATE Estimation Error vs. Confounder Intensity')
    plt.legend()
    plt.tight_layout()
    plot_path = f"ATE_Estimation_Error_All_Runs.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    wandb.log({"ATE Estimation Error vs. Confounder Intensity": wandb.Image(plot_path)})
    print(f"ATE Estimation Error plot saved: {plot_path}")

    print("\nAll runs completed and results plotted.\n")

# --------------------------- Execution Entry Point --------------------------- #

if __name__ == "__main__":
    main()
