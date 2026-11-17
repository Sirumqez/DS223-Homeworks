"""
Homework 2: A/B Testing with Multi-Armed Bandits
================================================
Author: Sereen Sipan
Course: Marketing Analytics
Instructor: Karen Hovhannisyan
Teaching Assistant: Artur Avagyan

Four advertisement options (arms) have Gaussian rewards with true means
[1, 2, 3, 4] and known precision tau = 1 (variance = 1 / tau).

Algorithms
----------
* EpsilonGreedy: epsilon_t = 1 / t, sample-average value estimates.
* ThompsonSampling: Gaussian Thompson Sampling with known reward precision
  and a Normal prior over every unknown arm mean.

Run
---
    python Bandit.py

Outputs (folder ``results/``): combined CSV, learning-curve plots,
algorithm comparison plot and a multi-seed robustness plot. Log messages
(loguru) report cumulative/average reward and regret.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import csv
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend: works without a display
import matplotlib.pyplot as plt
from loguru import logger


# ------------------------------ CONFIGURATION ------------------------------

BANDIT_REWARD = [1, 2, 3, 4]
NUMBER_OF_TRIALS = 20_000
KNOWN_PRECISION = 1.0
PRIOR_MEAN = 0.0
PRIOR_PRECISION = 0.01
RANDOM_SEED = 42
N_ROBUSTNESS_RUNS = 30
OUTPUT_DIR = Path("results")


# -------------------------------- BANDIT -----------------------------------

class Bandit(ABC):
    """Abstract base class for a multi-armed bandit algorithm."""

    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        """Initialise the bandit.

        Parameters
        ----------
        p : list of float
            True expected reward (mean) of each arm.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """Return a readable description of the algorithm."""
        pass

    @abstractmethod
    def pull(self):
        """Choose an arm and draw its stochastic reward."""
        pass

    @abstractmethod
    def update(self):
        """Update the algorithm's knowledge with the last observed reward."""
        pass

    @abstractmethod
    def experiment(self):
        """Run all trials."""
        pass

    @abstractmethod
    def report(self):
        """Store data in CSV, log average/cumulative reward and regret."""
        pass


class _GaussianBanditBase(Bandit):
    """Shared machinery for the two algorithms (added; Bandit is untouched).

    Parameters
    ----------
    p : list of float
        True mean reward of each arm.
    n_trials : int
        Number of trials to run.
    known_precision : float
        Precision (1 / variance) of the Gaussian reward noise.
    seed : int
        Seed of the random number generator.
    """

    name = "Base"

    def __init__(self, p, n_trials=NUMBER_OF_TRIALS,
                 known_precision=KNOWN_PRECISION, seed=RANDOM_SEED):
        self.p = np.asarray(p, dtype=float)
        self.n_arms = len(self.p)
        self.n_trials = int(n_trials)
        self.known_precision = float(known_precision)
        self.rng = np.random.default_rng(seed)

        self.counts = np.zeros(self.n_arms, dtype=int)
        self.selected_arms = []
        self.rewards = []
        self.regrets = []
        self.value_history = np.zeros((self.n_trials, self.n_arms))
        self.cumulative_rewards = np.zeros(self.n_trials)
        self.cumulative_regrets = np.zeros(self.n_trials)
        self.t = 0
        self.last_arm = None
        self.last_reward = None

    def _draw_reward(self, arm):
        """Draw a Gaussian reward for ``arm`` and remember the choice.

        Parameters
        ----------
        arm : int
            Index of the chosen arm.

        Returns
        -------
        float
            Observed reward.
        """
        reward = float(self.rng.normal(
            loc=self.p[arm], scale=1.0 / np.sqrt(self.known_precision)))
        self.last_arm, self.last_reward = arm, reward
        return reward

    def _log_outcome(self, arm, reward):
        """Store the trial outcome and its pseudo-regret.

        Parameters
        ----------
        arm : int
            Index of the chosen arm.
        reward : float
            Observed reward.
        """
        self.selected_arms.append(arm)
        self.rewards.append(reward)
        self.regrets.append(float(np.max(self.p) - self.p[arm]))

    def _current_values(self):
        """Return the per-arm value estimates (implemented by subclasses)."""
        raise NotImplementedError

    def experiment(self):
        """Run ``n_trials`` pull/update cycles and record learning curves.

        Returns
        -------
        _GaussianBanditBase
            The instance itself (allows chaining).
        """
        logger.debug("Starting experiment for {}", self.name)
        total_reward = total_regret = 0.0
        for trial in range(self.n_trials):
            self.pull()
            self.update()
            self.value_history[trial] = self._current_values()
            total_reward += self.rewards[-1]
            total_regret += self.regrets[-1]
            self.cumulative_rewards[trial] = total_reward
            self.cumulative_regrets[trial] = total_regret
        return self

    def report(self, csv_path=None):
        """Append trial data to a CSV and log reward / regret summaries.

        CSV columns: ``Bandit`` (arm number 1-based), ``Reward``, ``Algorithm``.

        Parameters
        ----------
        csv_path : str or Path, optional
            Destination file. Defaults to ``results/bandit_rewards.csv``.

        Returns
        -------
        dict
            Cumulative and average reward / regret of the algorithm.
        """
        csv_path = Path(csv_path or OUTPUT_DIR / "bandit_rewards.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(["Bandit", "Reward", "Algorithm"])
            for arm, reward in zip(self.selected_arms, self.rewards):
                writer.writerow([arm + 1, reward, self.name])

        summary = {
            "algorithm": self.name,
            "cumulative_reward": float(self.cumulative_rewards[-1]),
            "cumulative_regret": float(self.cumulative_regrets[-1]),
            "average_reward": float(np.mean(self.rewards)),
            "average_regret": float(np.mean(self.regrets)),
        }
        logger.info(
            f"{self.name} | cumulative reward: "
            f"{summary['cumulative_reward']:.2f} | cumulative regret: "
            f"{summary['cumulative_regret']:.2f}")
        logger.info(
            f"{self.name} | average reward: {summary['average_reward']:.4f} "
            f"| average regret: {summary['average_regret']:.4f}")
        return summary


# ------------------------------ EPSILON-GREEDY ------------------------------

class EpsilonGreedy(_GaussianBanditBase):
    """Epsilon-Greedy with decaying exploration ``epsilon_t = 1 / t``.

    Value estimates are incremental sample averages (initialised at 0).
    """

    name = "Epsilon-Greedy"

    def __init__(self, p, n_trials=NUMBER_OF_TRIALS,
                 known_precision=KNOWN_PRECISION, seed=RANDOM_SEED):
        super().__init__(p, n_trials, known_precision, seed)
        self.estimates = np.zeros(self.n_arms)

    def __repr__(self):
        return (f"{self.name}(arms={self.n_arms}, "
                f"trials={self.n_trials}, epsilon=1/t)")

    def _current_values(self):
        """Return the sample-average estimate of every arm."""
        return self.estimates

    def pull(self):
        """Explore with probability 1/t, otherwise exploit the best estimate.

        Returns
        -------
        float
            Observed reward of the chosen arm.
        """
        self.t += 1
        if self.rng.random() < 1.0 / self.t:
            arm = int(self.rng.integers(self.n_arms))
        else:
            best = np.flatnonzero(self.estimates == self.estimates.max())
            arm = int(self.rng.choice(best))  # random tie-breaking
        return self._draw_reward(arm)

    def update(self):
        """Update the chosen arm's sample average."""
        arm, reward = self.last_arm, self.last_reward
        self.counts[arm] += 1
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]
        self._log_outcome(arm, reward)


# ------------------------------ THOMPSON SAMPLING ----------------------------

class ThompsonSampling(_GaussianBanditBase):
    """Gaussian Thompson Sampling with known reward precision.

    Each arm's unknown mean has prior ``N(prior_mean, 1 / prior_precision)``.
    After ``n`` rewards with sum ``S`` the posterior is Normal with
    precision ``tau_0 + n * tau`` and mean
    ``(tau_0 * mu_0 + tau * S) / (tau_0 + n * tau)``.
    """

    name = "Thompson Sampling"

    def __init__(self, p, n_trials=NUMBER_OF_TRIALS,
                 known_precision=KNOWN_PRECISION, seed=RANDOM_SEED + 1,
                 prior_mean=PRIOR_MEAN, prior_precision=PRIOR_PRECISION):
        super().__init__(p, n_trials, known_precision, seed)
        self.prior_mean = float(prior_mean)
        self.prior_precision = float(prior_precision)
        self.reward_sums = np.zeros(self.n_arms)
        self.posterior_means = np.full(self.n_arms, self.prior_mean)
        self.posterior_precisions = np.full(self.n_arms, self.prior_precision)

    def __repr__(self):
        return (f"{self.name}(arms={self.n_arms}, trials={self.n_trials}, "
                f"known_precision={self.known_precision:g})")

    def _current_values(self):
        """Return the posterior mean of every arm."""
        return self.posterior_means

    def pull(self):
        """Sample a plausible mean per arm from its posterior; pick the max.

        Returns
        -------
        float
            Observed reward of the chosen arm.
        """
        self.t += 1
        sampled = self.rng.normal(
            loc=self.posterior_means,
            scale=1.0 / np.sqrt(self.posterior_precisions))
        return self._draw_reward(int(np.argmax(sampled)))

    def update(self):
        """Conjugate Normal update of the chosen arm's posterior."""
        arm, reward = self.last_arm, self.last_reward
        self.counts[arm] += 1
        self.reward_sums[arm] += reward
        self.posterior_precisions[arm] = (
            self.prior_precision + self.counts[arm] * self.known_precision)
        self.posterior_means[arm] = (
            self.prior_precision * self.prior_mean
            + self.known_precision * self.reward_sums[arm]
        ) / self.posterior_precisions[arm]
        self._log_outcome(arm, reward)


# ------------------------------ VISUALIZATION ------------------------------

class Visualization:
    """Learning-curve, comparison and robustness plots."""

    def plot1(self, algorithms):
        """Plot each arm's estimated value (linear and log trial axis).

        Parameters
        ----------
        algorithms : list
            Fitted bandit objects (after ``experiment()``).
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for algorithm in algorithms:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            x = np.arange(1, algorithm.n_trials + 1)
            for arm in range(algorithm.n_arms):
                label = f"Arm {arm + 1} (true mean={algorithm.p[arm]:g})"
                for axis in axes:
                    axis.plot(x, algorithm.value_history[:, arm], label=label)
            for axis, scale in zip(axes, ["linear", "log"]):
                axis.set_title(f"{algorithm.name}: learning process ({scale})")
                axis.set_xlabel("Trial" + (" (log scale)" if scale == "log" else ""))
                axis.set_ylabel("Estimated expected reward")
                axis.grid(alpha=0.3, which="both")
                axis.legend()
            axes[1].set_xscale("log")
            fig.tight_layout()
            safe = algorithm.name.lower().replace(" ", "_")
            filename = OUTPUT_DIR / f"{safe}_learning.png"
            fig.savefig(filename, dpi=150)
            plt.close(fig)
            logger.info("Saved learning plot to {}", filename)

    def plot2(self, algorithms):
        """Compare cumulative rewards and cumulative regrets.

        Parameters
        ----------
        algorithms : list
            Fitted bandit objects.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for algorithm in algorithms:
            x = np.arange(1, algorithm.n_trials + 1)
            axes[0].plot(x, algorithm.cumulative_rewards, label=algorithm.name)
            axes[1].plot(x, algorithm.cumulative_regrets, label=algorithm.name)
        axes[0].set_title("Cumulative reward comparison")
        axes[0].set_ylabel("Cumulative observed reward")
        axes[1].set_title("Cumulative regret comparison")
        axes[1].set_ylabel("Cumulative pseudo-regret")
        for axis in axes:
            axis.set_xlabel("Trial")
            axis.grid(alpha=0.3)
            axis.legend()
        fig.tight_layout()
        filename = OUTPUT_DIR / "algorithm_comparison.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        logger.info("Saved comparison plot to {}", filename)

    def plot3(self, regret_curves, final_regrets):
        """Plot multi-seed robustness (mean regret +- 95% CI, final spread).

        Parameters
        ----------
        regret_curves : dict
            ``{algorithm name: array (runs, trials)}`` of cumulative regret.
        final_regrets : dict
            ``{algorithm name: array (runs,)}`` of final cumulative regret.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for name, curves in regret_curves.items():
            x = np.arange(1, curves.shape[1] + 1)
            mean = curves.mean(axis=0)
            half = 1.96 * curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])
            axes[0].plot(x, mean, label=name)
            axes[0].fill_between(x, mean - half, mean + half, alpha=0.25)
        axes[0].set_title(f"Mean cumulative regret, {curves.shape[0]} seeds (95% CI)")
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("Cumulative pseudo-regret")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].boxplot(list(final_regrets.values()))
        axes[1].set_xticklabels(list(final_regrets.keys()))
        axes[1].set_title("Final cumulative regret across seeds")
        axes[1].set_ylabel("Cumulative pseudo-regret")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        filename = OUTPUT_DIR / "robustness_multi_seed.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        logger.info("Saved robustness plot to {}", filename)


# -------------------------------- COMPARISON ---------------------------------

def robustness_check(n_runs=N_ROBUSTNESS_RUNS):
    """Repeat the experiment with many seeds to avoid judging one lucky run.

    Parameters
    ----------
    n_runs : int
        Number of independent repetitions per algorithm.

    Returns
    -------
    tuple of dict
        ``(regret_curves, final_regrets)`` keyed by algorithm name.
    """
    curves = {EpsilonGreedy.name: [], ThompsonSampling.name: []}
    for run in range(n_runs):
        seed = 1000 + run
        eg = EpsilonGreedy(BANDIT_REWARD, seed=seed).experiment()
        ts = ThompsonSampling(BANDIT_REWARD, seed=seed + 10_000).experiment()
        curves[eg.name].append(eg.cumulative_regrets)
        curves[ts.name].append(ts.cumulative_regrets)
    curves = {k: np.array(v) for k, v in curves.items()}
    finals = {k: v[:, -1] for k, v in curves.items()}
    for name, values in finals.items():
        logger.info(
            f"{name} over {n_runs} seeds | final regret mean: "
            f"{values.mean():.1f} | median: {np.median(values):.1f} | "
            f"min: {values.min():.1f} | max: {values.max():.1f}")
    return curves, finals


def comparison():
    """Run both algorithms, save outputs and compare them visually.

    Returns
    -------
    list of dict
        Summary (cumulative / average reward and regret) per algorithm.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "bandit_rewards.csv"
    if csv_path.exists():
        csv_path.unlink()  # fresh CSV on every complete run

    epsilon_greedy = EpsilonGreedy(BANDIT_REWARD).experiment()
    thompson_sampling = ThompsonSampling(BANDIT_REWARD).experiment()
    algorithms = [epsilon_greedy, thompson_sampling]
    logger.debug("Algorithms: {}", algorithms)

    reports = [algorithm.report(csv_path) for algorithm in algorithms]
    viz = Visualization()
    viz.plot1(algorithms)
    viz.plot2(algorithms)
    viz.plot3(*robustness_check())
    logger.info("Results saved in '{}'", OUTPUT_DIR.resolve())
    return reports


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    comparison()
