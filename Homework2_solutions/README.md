# Homework 2 - Sereen Sipan - A/B Testing with Multi-Armed Bandits

## Files
- `Bandit.py` — complete implementation of the abstract `Bandit` class, Epsilon-Greedy, Thompson Sampling, visualization, and experiment runner.
- `requirements.txt` — Python dependencies.
- `results/bandit_rewards.csv` — 40,000 trial-level rows (20,000 per algorithm).
- `results/epsilon-greedy_learning.png` — per-arm estimated values for Epsilon-Greedy, linear and log trial axes.
- `results/thompson sampling_learning.png` — per-arm posterior means for Thompson Sampling, linear and log trial axes.
- `results/algorithm_comparison.png` — cumulative reward and cumulative regret comparison.

## Assumptions and experiment design
The assignment specifies `Bandit_Reward = [1, 2, 3, 4]` but does not define a reward distribution. These values are interpreted as the true expected rewards (means) of four Gaussian arms:
`Reward | arm i ~ Normal(mean=p[i], variance=1/known_precision)`.

This interpretation avoids treating 2, 3, and 4 as probabilities, which would be invalid for Bernoulli arms. The best arm is therefore arm 4, with expected reward 4.

- Number of trials: 20,000 per algorithm.
- Known reward precision: 1.0 (reward variance = 1.0).
- Epsilon-Greedy: `epsilon_t = 1/t`; random arm with probability epsilon, otherwise choose the arm with the highest sample-average estimate.
- Thompson Sampling: Gaussian conjugate model with known reward precision, Normal prior mean 0, and prior precision 0.01.
- Reproducibility: fixed random seeds.
- Regret: per-trial pseudo-regret = `max(true_means) - true_mean_of_selected_arm`; cumulative regret is the sum over trials. This measures the expected-reward opportunity cost and is not calculated from noisy reward samples.

## Run locally
Use Python 3.9 or newer.

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
python Bandit.py
```

Running the script recreates the `results/` directory outputs. It prints each algorithm's cumulative observed reward and cumulative pseudo-regret.

## How Thompson Sampling works here
For each arm, the unknown mean has prior `Normal(mu_0, 1/tau_0)`. After observing `n` rewards with sum `S`, with known reward precision `tau`, the posterior precision and mean are:

`tau_n = tau_0 + n*tau`

`mu_n = (tau_0*mu_0 + tau*S) / tau_n`

At each trial, one candidate mean is sampled from each arm's posterior; the arm with the largest sampled mean is selected. The selected arm's observed reward is then used to update its posterior.

## Bonus: a stronger implementation plan
A more production-ready A/B testing plan would:
1. **Validate the reward model first.** Confirm whether rewards are binary conversions, revenue, ratings, or continuous outcomes. Use Bernoulli/Beta Thompson Sampling for conversions, or a suitable continuous-outcome model for revenue.
2. **Use a warm-up phase.** Ensure every arm receives a minimum number of observations before aggressive exploitation, reducing the risk of prematurely ignoring an arm.
3. **Compare multiple random seeds.** Repeat the 20,000-trial experiment (for example, 30–100 runs) and report mean and confidence intervals, not just one lucky/unlucky run.
4. **Track business guardrails.** Monitor conversion rate, revenue per visitor, sample ratio mismatch, and relevant user-experience or safety metrics.
5. **Use a fixed-horizon or sequential-testing plan.** Decide in advance how the experiment will stop and how statistical uncertainty will be handled to avoid stopping simply because a result temporarily looks favorable.
6. **Consider contextual and non-stationary bandits.** If users or ad performance differ by device, location, or time, add context; if performance changes over time, use a sliding window or discount older observations.
7. **Validate with an online controlled experiment.** Bandit allocation is useful for adaptive traffic allocation, but a randomized holdout/control group can help measure causal lift and detect regressions.
