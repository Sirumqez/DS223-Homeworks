# Homework 2 - Sereen Sipan - A/B Testing with Multi-Armed Bandits

## Files
- `Bandit.py` - abstract `Bandit` class (untouched), Epsilon-Greedy, Thompson Sampling, visualization, experiment runner and a multi-seed robustness check.
- `requirements.txt` - dependencies (loguru, numpy, matplotlib).
- `results/bandit_rewards.csv` - 40,000 rows `{Bandit, Reward, Algorithm}` (20,000 per algorithm).
- `results/epsilon-greedy_learning.png`, `results/thompson_sampling_learning.png` - per-arm estimates, linear and log trial axes (plot1).
- `results/algorithm_comparison.png` - cumulative reward and regret, both algorithms (plot2).
- `results/robustness_multi_seed.png` - 30 independent runs: mean regret with 95% CI and final-regret spread.

## Run locally (Python 3.9+)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python Bandit.py
```
Runtime is about 30 seconds. Results are written to `results/`; cumulative and average reward/regret are logged with loguru.

## Assumptions
`Bandit_Reward = [1, 2, 3, 4]` is read as the true means of four Gaussian arms, `Reward | arm i ~ Normal(p[i], 1/tau)`, with known precision `tau = 1`. Values 2-4 cannot be Bernoulli probabilities. Best arm = arm 4.
- Epsilon-Greedy: `epsilon_t = 1/t`, sample-average estimates initialised at 0, random tie-breaking.
- Thompson Sampling: Normal prior (mean 0, precision 0.01) per arm, conjugate update with known precision: `tau_n = tau_0 + n*tau`, `mu_n = (tau_0*mu_0 + tau*S)/tau_n`.
- Regret: pseudo-regret `max(p) - p[chosen arm]`, summed over trials.
- Fixed seeds (Epsilon-Greedy 42, Thompson 43).

## Results (seed 42)
| Algorithm | Cumulative reward | Cumulative regret |
|---|---|---|
| Epsilon-Greedy | 73,907 | 6,181 |
| Thompson Sampling | 79,759 | 26 |

## Why Epsilon-Greedy looks so bad in the single run
With `epsilon_t = 1/t` the expected number of exploratory pulls over the whole run is `sum(1/t) ~ ln(20000) ~ 10`, i.e. only about 2-3 per arm. Estimates start at 0 and rewards are positive, so the greedy rule keeps picking an arm it has already tried and never revisits untried ones. In seed 42 the algorithm sat on arm 3 (regret 1 per trial) until a rare random exploration hit arm 4 at about trial 6,200. This is a property of the schedule, not a bug. Across 30 seeds Epsilon-Greedy has median final regret ~22 but a heavy tail (max ~20,000 when arm 4 is never found), while Thompson Sampling is consistently low (mean ~31, range 14-54). Judging from one seed would be misleading, hence the robustness plot.

## Bonus: a better implementation plan
1. **Validate the reward model first.** Binary conversions: Beta-Bernoulli Thompson Sampling. Revenue: a suitable continuous model (e.g. log-normal or Gaussian with unknown variance).
2. **Warm-up phase.** Give every arm a minimum number of impressions before exploiting; this directly removes the lock-in failure seen above.
3. **Better Epsilon-Greedy.** Optimistic initial values, or a floor such as `epsilon_t = max(eps_min, 1/t)`.
4. **Many seeds.** Report mean and confidence intervals (implemented in `robustness_check`).
5. **Business guardrails.** Track conversion, revenue per visitor, sample-ratio mismatch and user-experience metrics.
6. **Pre-registered stopping rule.** Fixed horizon or sequential testing, so you don't stop on a temporarily favourable result.
7. **Context and non-stationarity.** Contextual bandits (device, location, time); sliding window or discounting if ad performance drifts.
8. **Keep a randomized holdout.** Adaptive allocation is efficient but a control group measures causal lift and catches regressions.
