# Technique Glossary

A detailed reference for the analytical techniques used in this F1 strategy analysis project. Designed for technical discussions.

---

## Overview: The Three-Technique Framework

F1 strategy engineers rely on three core analytical techniques. This project demonstrates all three:

| Technique | Purpose | This Project |
|-----------|---------|--------------|
| **Regression Analysis** | Extract trends from noisy data | Tyre degradation rates from lap times |
| **Optimization** | Find best decision under constraints | Optimal pit lap via grid search |
| **Monte Carlo Simulation** | Quantify uncertainty | Win probabilities across 1,000 races |

---

## 1. Linear Regression (Ordinary Least Squares)

### What It Is

A statistical method that finds the best-fit straight line through data by minimizing the sum of squared residuals (differences between predicted and actual values).

### Why I Used It

Tyre degradation over a stint is approximately linear for most of the tyre's life. Linear regression extracts the degradation rate (slope) from noisy lap time data, giving a single interpretable number: **seconds lost per lap due to tyre wear**.

### How I Applied It

```python
from sklearn.linear_model import LinearRegression

# For each stint, fit: LapTime = β₀ + β₁ × TyreAge + ε
model = LinearRegression()
model.fit(X=tyre_age.values.reshape(-1, 1), y=lap_times)

degradation_rate = model.coef_[0]  # seconds per lap
initial_pace = model.intercept_     # pace on fresh tyres
```

### Results From This Project

| Stint | Compound | Degradation Rate | R² | Interpretation |
|-------|----------|------------------|-----|----------------|
| 1 | MEDIUM | -0.013 s/lap | 0.08 | Negative = improving (fuel burn dominates) |
| 2 | HARD | +0.047 s/lap | 0.63 | Moderate fit, clear degradation trend |
| 3 | MEDIUM | +0.033 s/lap | 0.17 | Low R² but trend valid |

### Model Validation

The physics-based model using these degradation rates achieved:
- **Mean Absolute Error (MAE): 0.856s** per lap
- **Root Mean Square Error (RMSE): 1.000s**
- **Mean Bias: +0.622s** (model predicts slightly slower than reality)
- **Total race time error: +0.74%** (40.4s over 65 laps)
- **Target met:** MAE < 1.0s ✓

### Stint-by-Stint Validation

| Stint | Compound | MAE | Bias | Interpretation |
|-------|----------|-----|------|----------------|
| 1 | MEDIUM | 1.244s | +1.244s | Fuel effect underestimated |
| 2 | HARD | 0.971s | +0.964s | Traffic/damage confounded |
| 3 | MEDIUM | **0.534s** | **-0.026s** | **Excellent fit** |

### Key Assumptions

1. **Linearity** — Degradation is constant per lap (valid mid-stint, breaks down at tyre cliff)
2. **Independence** — Each lap's residual is independent
3. **Homoscedasticity** — Variance of residuals is constant
4. **Normality** — Residuals are normally distributed (for confidence intervals)

### Q&A

**Q: Why linear regression instead of polynomial?**

> Linear regression is interpretable and robust for mid-stint data. A polynomial would fit the tyre cliff better, but risks overfitting and extrapolates poorly. For strategy decisions made mid-race, a simple linear approximation with known limitations is more useful than a complex model that fails unpredictably.

**Q: Your R² values are low (0.08 to 0.63). Doesn't that mean the model is bad?**

> Low R² means tyre age alone doesn't explain all lap time variance — which is expected. Fuel load, traffic, driver inputs, car setup and track evolution all affect lap times. The regression still extracts a valid degradation signal; R² tells us other factors matter too. For strategy, the trend is what we need, not perfect lap time prediction. The overall model validated at MAE = 0.856s, which confirms the approach works.

**Q: Stint 1 shows negative degradation. Is that wrong?**

> No — it's real. Early in the race, fuel burn (~0.03s/lap improvement) and tyre warm-up outweigh degradation. The negative slope captures the net effect. By stint 3, fuel is mostly burned, so degradation dominates and the slope turns positive.

**Q: What does the slope actually represent?**

> The slope is the marginal cost of one additional lap on the tyre, in seconds. A slope of +0.047 (Hard stint 2) means each extra lap costs 0.047 seconds. Over a 24-lap stint, that compounds to ~1.1 seconds of total degradation penalty.

---

## 2. Grid Search Optimization

### What It Is

A brute-force optimization method that evaluates every combination of decision variables within a defined search space to find the global minimum (or maximum).

### Why I Used It

Pit stop timing is a discrete decision (you pit on lap 18 or lap 19, not lap 18.5). The search space is small enough that exhaustive search is computationally tractable and **guarantees finding the global optimum** within the search space.

### How I Applied It

```python
# 1-Stop optimization
best_time = float('inf')
best_lap = None

for pit_lap in range(15, 50):  # feasible range
    total_time = simulate_race(pit_laps=[pit_lap])
    if total_time < best_time:
        best_time = total_time
        best_lap = pit_lap

# 2-Stop optimization
for pit1 in range(15, 35):
    for pit2 in range(pit1 + 10, 55):  # min stint length constraint
        total_time = simulate_race(pit_laps=[pit1, pit2])
        # ... track best
```

### Results From This Project

**Strategy Comparison:**

| Strategy | Pit Laps | Total Time | Gap to Best |
|----------|----------|------------|-------------|
| **1-Stop (M-H)** | 34 | 5986.25s | — |
| 2-Stop Optimal (M-H-M) | 25, 47 | 5992.62s | +6.37s |
| Piastri Actual (M-H-M) | 18, 42 | 5994.19s | +7.94s |

**Key finding:** Piastri's strategy cost ~1.6s vs optimal 2-stop timing, but was ~8s slower than a 1-stop approach.

### Computational Complexity

| Strategy | Complexity | Search Space (70-lap race) |
|----------|------------|---------------------------|
| 1-Stop | O(n) | ~35 combinations |
| 2-Stop | O(n²) | ~400 combinations |
| 3-Stop | O(n³) | ~4,000 combinations |

All tractable for modern computers in milliseconds.

### When Grid Search Fails

- **Continuous decision spaces** — Use gradient descent
- **High-dimensional problems** — Curse of dimensionality
- **Expensive evaluations** — Use Bayesian optimization

### Q&A

**Q: Why grid search instead of gradient descent?**

> Pit lap decisions are discrete integers, not continuous variables. Gradient descent requires a differentiable objective function, which we don't have. Grid search guarantees the global optimum within the search space, and the space is small enough that computational cost is negligible.

**Q: What are the limitations?**

> It doesn't scale to high-dimensional problems. If I had 10 decision variables with 20 options each, that's 20¹⁰ = 10 trillion combinations. For pit stop optimization with 2-3 stops, it's the right tool.

**Q: How did you handle constraints?**

> I filtered invalid combinations before evaluation: minimum stint length (10 laps), maximum tyre life, mandatory compound usage. Invalid strategies are skipped, not penalized — keeps the objective function clean.

**Q: The optimal is only ~1.6s faster than Piastri's actual. Is that significant?**

> In absolute terms, 1.6s over a 100-minute race is small. But F1 races are often decided by smaller margins. More importantly, this is the *time* difference — the *position* impact could be much larger due to track position effects that this model doesn't capture. Losing 3 positions at Hungary could cost 45-60s in traffic.

---

## 3. Monte Carlo Simulation

### What It Is

A computational technique that uses repeated random sampling to estimate the probability distribution of outcomes that depend on uncertain inputs.

### Why I Used It

Race outcomes depend on variables we can't predict exactly:
- Actual degradation rates (±15% variance)
- Pit stop execution time (22.0s ± 0.3s)
- Lap time variability (±0.15s per lap)
- Safety Car probability (2% per lap)

Monte Carlo lets us ask "what's the probability distribution of outcomes?" rather than "what's the single best answer?"

### How I Applied It

```python
import numpy as np

n_simulations = 1000
results = []

for _ in range(n_simulations):
    # Sample uncertain parameters
    deg_variance = np.random.normal(1.0, 0.15)  # ±15%
    pit_time = np.random.normal(22.0, 0.3)      # with 5% slow stop chance
    
    # Simulate race with sampled parameters
    total_time = simulate_race(
        deg_multiplier=deg_variance,
        pit_stop_time=pit_time
    )
    results.append(total_time)

# Analyze distribution
mean = np.mean(results)
std = np.std(results)
p5 = np.percentile(results, 5)   # best 5%
p95 = np.percentile(results, 95) # worst 5%
```

### Results From This Project

**Strategy Comparison (1,000 simulations each):**

| Strategy | Mean Time | Std Dev | P5 (Best) | P95 (Worst) | Stops |
|----------|-----------|---------|-----------|-------------|-------|
| **1-Stop (L34)** | 5986.25s | 4.35s | 5978.70s | 5993.29s | 1 |
| Soft Start (L12, L35) | 5991.74s | 4.44s | 5984.52s | 5999.26s | 2 |
| Optimal (L25, L47) | 5992.62s | 2.86s | 5987.81s | 5997.38s | 2 |
| Actual (L18, L42) | 5994.19s | 2.99s | 5989.12s | 5998.85s | 2 |
| Aggressive (L15, L38) | 5996.28s | 3.65s | 5990.61s | 6002.41s | 2 |

**Head-to-Head Win Probabilities:**

| Comparison | Probability |
|------------|-------------|
| Optimal 2-Stop beats Actual | **65.6%** |
| Actual beats Optimal | **34.4%** |
| 1-Stop beats 2-Stop | **90.0%** |

**Safety Car Impact:**
- P(at least one SC in 70 laps): **75.7%**
- Expected SC per race: **1.40**
- SC events favour 2-stop strategies (flexibility to pit under caution)

### Key Concepts

**Convergence:** Standard error decreases as 1/√n. With 1,000 simulations, results are stable to ~3% precision.

**Confidence Intervals:** The 5th-95th percentile range shows where 90% of outcomes fall — more useful than a point estimate.

**Head-to-Head Probability:** By pairing simulations (same random seed for both strategies), we compute P(A beats B) directly.

### Q&A

**Q: Why 1,000 simulations? Why not 500 or 10,000?**

> Standard error scales as 1/√n. Going from 1,000 to 10,000 only improves precision by √10 ≈ 3×, while taking 10× longer. For strategy decisions where we care about rough probabilities, 1,000 is sufficient. I validated by checking stability across multiple runs.

**Q: How did you choose the distributions?**

> Degradation variance (±15%) was estimated from regression residuals. Pit stop distribution came from published F1 data (~22s mean, 0.3s std for top teams). Lap time noise was calibrated to match observed variance in clean-air laps. These are data-driven, not arbitrary.

**Q: The optimal only wins 65.6% of the time. What does that mean?**

> The expected-value-optimal strategy isn't always robust. The 1.6s mean advantage is smaller than the combined variance, so outcomes overlap significantly. In a single race, variance dominates. This is why teams sometimes choose "conservative" strategies — a 65% edge isn't enough when championship points are at stake.

**Q: What's the difference between Monte Carlo and sensitivity analysis?**

> Sensitivity analysis varies one parameter at a time. Monte Carlo varies all parameters simultaneously according to their distributions, capturing interactions. Sensitivity asks "what if degradation is 10% higher?" Monte Carlo asks "given realistic uncertainty in everything, what's the distribution of outcomes?"

---

## 4. Physics-Based Modelling (Supporting Technique)

### What It Is

Building predictive models from first principles rather than purely from data patterns.

### Why I Used It

Pure ML models (XGBoost, neural nets) can fit training data well but extrapolate poorly. A physics-based model — where lap time = base pace + fuel effect + degradation + traffic — remains interpretable and behaves sensibly outside the training range.

### Model Structure

```python
def predict_lap_time(lap_num, tyre_age, compound, in_traffic):
    # Base pace (from clean-air laps)
    base = 82.5  # seconds
    
    # Fuel effect (car gets lighter)
    fuel_effect = -0.035 * fuel_load  # s per kg
    
    # Tyre degradation (compound-specific, from regression)
    if compound == 'MEDIUM':
        deg = 0.033 * tyre_age  # from Stint 3 data
    elif compound == 'HARD':
        deg = 0.047 * tyre_age  # from Stint 2 data
    
    # Dirty air penalty
    if gap_to_car_ahead < 2.0:
        traffic = 1.2 / (1 + (gap / 0.8)**2)
    else:
        traffic = 0.0
    
    return base + fuel_effect + deg + traffic
```

### Validation Results

| Metric | Value | Target |
|--------|-------|--------|
| Mean Absolute Error | **0.856s** | < 1.0s ✓ |
| RMSE | **1.000s** | — |
| Mean Bias | **+0.622s** | — |
| Total Race Time Error | **+0.74%** | — |

### Q&A

**Q: Why not just use XGBoost or neural networks?**

> I tested machine learning approaches. They fit training data well but predicted nonsense for untested scenarios — like negative degradation at lap 50. Physics-based models extrapolate sensibly because the structure encodes real relationships. The physics model validated at MAE = 0.856s, proving it captures the dominant effects.

**Q: How do you know your physics model is right?**

> Validation against real data. The model predicts Piastri's race time within 0.74% of actual (40.4s error over 5,483s). That's good enough for comparative strategy analysis, and the stint-by-stint breakdown shows where the model works well (Stint 3: MAE 0.534s) versus where it needs improvement (Stint 1: MAE 1.244s).

---

## Summary Table

| Technique | What It Answers | Key Result |
|-----------|-----------------|------------|
| **Regression** | How fast do tyres degrade? | HARD: +0.047 s/lap, MEDIUM: +0.033 s/lap |
| **Grid Search** | When should we pit? | 1-Stop: L34, 2-Stop: L25/L47 |
| **Monte Carlo** | How confident are we? | Optimal beats Actual only **65.6%** of the time |
| **Validation** | Does the model work? | MAE = **0.856s** ✓ |

These three techniques form the core analytical toolkit for strategy engineering — not just in F1, but across aerospace, energy, finance, and any field where decisions must be made under uncertainty.

---

*This glossary accompanies the F1 Strategy Analysis project and serves as an interview preparation reference.*

