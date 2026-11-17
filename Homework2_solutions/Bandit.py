"""
Homework 2:  A/B Testing with Multi-Armed Bandits
================================================
Author: Sereen Sipan
Course: Marketing Analytics
Instructor: Karen Hovhannisyan
Teaching Assistant: Artur Avagyan



The four arms have Gaussian reward distributions with known means
[1, 2, 3, 4] and known precision tau=1 (variance = 1/tau).

Algorithms:
    * EpsilonGreedy: epsilon_t = 1/t, with sample-average value estimates.
    * ThompsonSampling: Gaussian Thompson Sampling with known reward
      precision and a Normal prior over each unknown arm mean.

Run:
    python Bandit.py

The script runs 20,000 trials for each algorithm, saves a combined CSV,
prints cumulative reward/regret, and creates two figures.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger


# ------------------------------ CONFIGURATION ------------------------------

BANDIT_REWARD = [1, 2, 3, 4]
NUMBER_OF_TRIALS = 20_000
KNOWN_PRECISION = 1.0
PRIOR_MEAN = 0.0
PRIOR_PRECISION = 0.01
RANDOM_SEED = 42
OUTPUT_DIR = Path("results")


# -------------------------------- BANDIT -----------------------------------

class Bandit(ABC):
    """Abstract base class for a multi-armed bandit algorithm.

    Parameters
    ----------
    p : list[float]
        True expected reward (mean) of each arm. Rewards are sampled from
        Normal(p[i], 1 / known_precision).
    """

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # Store data in CSV, print cumulative reward and cumulative regret.
        pass


# ------------------------------ VISUALIZATION ------------------------------

class Visualization:
    """Create learning-curve and algorithm-comparison plots."""

    def plot1(self, algorithms):
        """Plot each arm's estimated value on linear and logarithmic axes."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for algorithm in algorithms:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            x = np.arange(1, algorithm.n_trials + 1)

            for arm in range(algorithm.n_arms):
                axes[0].plot(
                    x, algorithm.value_history[:, arm],
                    label=f"Arm {arm + 1} (true mean={algorithm.p[arm]:g})"
                )
                axes[1].plot(
                    x, algorithm.value_history[:, arm],
                    label=f"Arm {arm + 1} (true mean={algorithm.p[arm]:g})"
                )

            axes[0].set_title(f"{algorithm.name}: learning process (linear)")
            axes[0].set_xlabel("Trial")
            axes[0].set_ylabel("Estimated expected reward")
            axes[0].grid(alpha=0.3)

            axes[1].set_title(f"{algorithm.name}: learning process (log scale)")
            axes[1].set_xlabel("Trial (log scale)")
            axes[1].set_ylabel("Estimated expected reward")
            axes[1].set_xscale("log")
            axes[1].grid(alpha=0.3, which="both")

            axes[0].legend()
            axes[1].legend()
            fig.tight_layout()
            filename = OUTPUT_DIR / f"{algorithm.name.lower()}_learning.png"
            fig.savefig(filename, dpi=150)
            plt.close(fig)
            logger.info("Saved learning plot to {}", filename)

    def plot2(self, algorithms):
        """Compare cumulative observed rewards and cumulative pseudo-regret."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for algorithm in algorithms:
            x = np.arange(1, algorithm.n_trials + 1)
            axes[0].plot(x, algorithm.cumulative_rewards,
                         label=algorithm.name)
            axes[1].plot(x, algorithm.cumulative_regrets,
                         label=algorithm.name)

        axes[0].set_title("Cumulative reward comparison")
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("Cumulative observed reward")
        axes[1].set_title("Cumulative regret comparison")
        axes[1].set_xlabel("Trial")
        axes[1].set_ylabel("Cumulative pseudo-regret")

        for axis in axes:
            axis.grid(alpha=0.3)
            axis.legend()

        fig.tight_layout()
        filename = OUTPUT_DIR / "algorithm_comparison.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        logger.info("Saved comparison plot to {}", filename)


# ------------------------------ EPSILON-GREEDY ------------------------------

class EpsilonGreedy(Bandit):
    """Epsilon-Greedy with epsilon_t = 1/t and sample-average estimates."""

    def __init__(self, p, n_trials=NUMBER_OF_TRIALS,
                 known_precision=KNOWN_PRECISION, seed=RANDOM_SEED):
        self.p = np.asarray(p, dtype=float)
        self.n_arms = len(self.p)
        self.n_trials = int(n_trials)
        self.known_precision = float(known_precision)
        self.rng = np.random.default_rng(seed)
        self.name = "Epsilon-Greedy"

        self.counts = np.zeros(self.n_arms, dtype=int)
        self.estimates = np.zeros(self.n_arms, dtype=float)
        self.selected_arms = []
        self.rewards = []
        self.regrets = []
        self.value_history = np.zeros((self.n_trials, self.n_arms))
        self.cumulative_rewards = np.zeros(self.n_trials)
        self.cumulative_regrets = np.zeros(self.n_trials)
        self.t = 0
        self.last_arm = None
        self.last_reward = None

    def __repr__(self):
        return (f"{self.name}(arms={self.n_arms}, "
                f"trials={self.n_trials}, epsilon=1/t)")

    def pull(self):
        """Choose an arm, then draw its stochastic reward."""
        self.t += 1
        epsilon = 1.0 / self.t

        if self.rng.random() < epsilon:
            arm = int(self.rng.integers(self.n_arms))
        else:
            best_arms = np.flatnonzero(self.estimates == self.estimates.max())
            arm = int(self.rng.choice(best_arms))

        reward = float(self.rng.normal(
            loc=self.p[arm], scale=1.0 / np.sqrt(self.known_precision)
        ))
        self.last_arm = arm
        self.last_reward = reward
        return reward

    def update(self):
        """Update the chosen arm's sample-average reward estimate."""
        arm = self.last_arm
        reward = self.last_reward
        self.counts[arm] += 1
        self.estimates[arm] += (
            reward - self.estimates[arm]
        ) / self.counts[arm]
        self.selected_arms.append(arm)
        self.rewards.append(reward)
        self.regrets.append(float(np.max(self.p) - self.p[arm]))

    def experiment(self):
        """Run the configured number of trials and store learning histories."""
        for trial in range(self.n_trials):
            self.pull()
            self.update()
            self.value_history[trial] = self.estimates
            previous_reward = self.cumulative_rewards[trial - 1] if trial else 0.0
            previous_regret = self.cumulative_regrets[trial - 1] if trial else 0.0
            self.cumulative_rewards[trial] = previous_reward + self.rewards[-1]
            self.cumulative_regrets[trial] = previous_regret + self.regrets[-1]
        return self

    def report(self, csv_path=None):
        """Append this algorithm's trial data to the requested CSV file."""
        if csv_path is None:
            csv_path = OUTPUT_DIR / "bandit_rewards.csv"
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0

        with csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(["Bandit", "Reward", "Algorithm"])
            for arm, reward in zip(self.selected_arms, self.rewards):
                writer.writerow([arm + 1, reward, self.name])

        logger.info(
            "{} | Cumulative reward: {:.3f} | Cumulative regret: {:.3f}",
            self.name, self.cumulative_rewards[-1],
            self.cumulative_regrets[-1]
        )
        return {
            "algorithm": self.name,
            "cumulative_reward": float(self.cumulative_rewards[-1]),
            "cumulative_regret": float(self.cumulative_regrets[-1]),
        }


# ------------------------------ THOMPSON SAMPLING ----------------------------

class ThompsonSampling(Bandit):
    """Gaussian Thompson Sampling with known reward precision.

    Each arm has a Normal prior N(PRIOR_MEAN, 1/PRIOR_PRECISION).
    Given observations with known precision tau, the posterior precision is
    prior_precision + n*tau and the posterior mean is the precision-weighted
    average of the prior mean and the observed rewards.
    """

    def __init__(self, p, n_trials=NUMBER_OF_TRIALS,
                 known_precision=KNOWN_PRECISION, seed=RANDOM_SEED + 1,
                 prior_mean=PRIOR_MEAN, prior_precision=PRIOR_PRECISION):
        self.p = np.asarray(p, dtype=float)
        self.n_arms = len(self.p)
        self.n_trials = int(n_trials)
        self.known_precision = float(known_precision)
        self.prior_mean = float(prior_mean)
        self.prior_precision = float(prior_precision)
        self.rng = np.random.default_rng(seed)
        self.name = "Thompson Sampling"

        self.counts = np.zeros(self.n_arms, dtype=int)
        self.reward_sums = np.zeros(self.n_arms, dtype=float)
        self.posterior_means = np.full(self.n_arms, self.prior_mean)
        self.posterior_precisions = np.full(
            self.n_arms, self.prior_precision
        )
        self.selected_arms = []
        self.rewards = []
        self.regrets = []
        self.value_history = np.zeros((self.n_trials, self.n_arms))
        self.cumulative_rewards = np.zeros(self.n_trials)
        self.cumulative_regrets = np.zeros(self.n_trials)
        self.t = 0
        self.last_arm = None
        self.last_reward = None

    def __repr__(self):
        return (f"{self.name}(arms={self.n_arms}, "
                f"trials={self.n_trials}, known_precision="
                f"{self.known_precision:g})")

    def pull(self):
        """Sample each arm's plausible mean and select the largest sample."""
        self.t += 1
        sampled_means = self.rng.normal(
            loc=self.posterior_means,
            scale=1.0 / np.sqrt(self.posterior_precisions)
        )
        arm = int(np.argmax(sampled_means))
        reward = float(self.rng.normal(
            loc=self.p[arm], scale=1.0 / np.sqrt(self.known_precision)
        ))
        self.last_arm = arm
        self.last_reward = reward
        return reward

    def update(self):
        """Update the selected arm's conjugate Normal posterior."""
        arm = self.last_arm
        reward = self.last_reward
        self.counts[arm] += 1
        self.reward_sums[arm] += reward

        self.posterior_precisions[arm] = (
            self.prior_precision + self.counts[arm] * self.known_precision
        )
        self.posterior_means[arm] = (
            self.prior_precision * self.prior_mean
            + self.known_precision * self.reward_sums[arm]
        ) / self.posterior_precisions[arm]

        self.selected_arms.append(arm)
        self.rewards.append(reward)
        self.regrets.append(float(np.max(self.p) - self.p[arm]))

    def experiment(self):
        """Run the configured number of trials and store learning histories."""
        for trial in range(self.n_trials):
            self.pull()
            self.update()
            self.value_history[trial] = self.posterior_means
            previous_reward = self.cumulative_rewards[trial - 1] if trial else 0.0
            previous_regret = self.cumulative_regrets[trial - 1] if trial else 0.0
            self.cumulative_rewards[trial] = previous_reward + self.rewards[-1]
            self.cumulative_regrets[trial] = previous_regret + self.regrets[-1]
        return self

    def report(self, csv_path=None):
        """Append this algorithm's trial data to the requested CSV file."""
        if csv_path is None:
            csv_path = OUTPUT_DIR / "bandit_rewards.csv"
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0

        with csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(["Bandit", "Reward", "Algorithm"])
            for arm, reward in zip(self.selected_arms, self.rewards):
                writer.writerow([arm + 1, reward, self.name])

        logger.info(
            "{} | Cumulative reward: {:.3f} | Cumulative regret: {:.3f}",
            self.name, self.cumulative_rewards[-1],
            self.cumulative_regrets[-1]
        )
        return {
            "algorithm": self.name,
            "cumulative_reward": float(self.cumulative_rewards[-1]),
            "cumulative_regret": float(self.cumulative_regrets[-1]),
        }


# -------------------------------- COMPARISON ---------------------------------

def comparison():
    """Run both algorithms, save their outputs, and compare performance."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "bandit_rewards.csv"
    if csv_path.exists():
        csv_path.unlink()  # Ensure a fresh CSV on every complete run.

    epsilon_greedy = EpsilonGreedy(BANDIT_REWARD).experiment()
    thompson_sampling = ThompsonSampling(BANDIT_REWARD).experiment()
    algorithms = [epsilon_greedy, thompson_sampling]

    reports = [algorithm.report(csv_path) for algorithm in algorithms]
    visualization = Visualization()
    visualization.plot1(algorithms)
    visualization.plot2(algorithms)

    logger.info("Results saved in '{}'", OUTPUT_DIR.resolve())
    return reports


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    comparison()
