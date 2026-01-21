# F1 Race Strategy Analysis: Oscar Piastri — 2023 Hungarian Grand Prix

A quantitative investigation into McLaren's pit stop strategy using real telemetry data, physics-based modelling, and Monte Carlo simulation.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastF1](https://img.shields.io/badge/FastF1-3.7+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## Motivation

During the 2023 Hungarian Grand Prix, Oscar Piastri jumped from P4 to P2 at the start, then lost positions late in the race to finish P5. McLaren pitted him on lap 18 — one lap after teammate Lando Norris. Watching the race, I questioned whether this early stop compromised his podium chances.

Rather than speculate, I built this analysis to investigate the decision using real FastF1 telemetry and quantitative methods.

**The question:** Did McLaren pit Piastri too early, and did this cost him track position?

---

## Key Finding

Lap 18 was close to **time-optimal** for a 2-stop strategy. The model-optimal timing (laps 25/47) saves only ~1.6s versus Piastri's actual (laps 18/42). However, the analysis revealed something more important:

**McLaren optimized for the wrong objective** — minimizing race time rather than maximizing track position.

At Hungary, where overtaking is notoriously difficult, track position value vastly exceeds pit timing precision. Monte Carlo analysis shows the "optimal" strategy only beats "suboptimal" alternatives **65.6%** of the time due to variance overlap.

**Critical insight:** A 1-stop strategy would be ~8s faster on average, but with higher variance. The remaining uncertainty explains why teams often choose "slower but safer" multi-stop strategies.

---

## Techniques Used

This project demonstrates the three core analytical techniques used by F1 strategy engineers:

| Technique | Application | Notebook |
|-----------|-------------|----------|
| **Regression Analysis** | Extracted tyre degradation rates from FastF1 telemetry | 04 |
| **Grid Search Optimization** | Found optimal pit windows under operational constraints | 03 |
| **Monte Carlo Simulation** | Quantified strategy uncertainty across 1,000 race simulations | 05 |

See **[TECHNIQUE_GLOSSARY.md](TECHNIQUE_GLOSSARY.md)** for detailed explanations.

---

## Results Summary

### 1. Validation Results (Notebook 04)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Mean Absolute Error (MAE)** | 0.856s | Good — target < 1.0s met ✓ |
| **Root Mean Square Error (RMSE)** | 1.000s | No catastrophic outliers |
| **Mean Bias** | +0.622s | Model predicts slightly slower than reality |
| **Maximum Error** | 1.805s | Worst single-lap prediction |
| **Total Race Time Error** | +0.74% | 40.4s over 65 laps |

**Stint-by-Stint Accuracy:**

| Stint | Compound | MAE | Bias | Interpretation |
|-------|----------|-----|------|----------------|
| 1 | MEDIUM | 1.244s | +1.244s | Fuel effect underestimated |
| 2 | HARD | 0.971s | +0.964s | Traffic/damage confounded |
| 3 | MEDIUM | 0.534s | -0.026s | **Excellent fit** |

### 2. Tyre Degradation Rates (Notebook 04)

| Stint | Compound | Degradation Rate | R² | Interpretation |
|-------|----------|------------------|-----|----------------|
| 1 | MEDIUM | -0.013 s/lap | 0.08 | Improving (fuel burn dominates) |
| 2 | HARD | +0.047 s/lap | 0.63 | Clear degradation trend |
| 3 | MEDIUM | +0.033 s/lap | 0.17 | Degrading (fuel mostly burned) |

### 3. Strategy Optimization (Notebook 03)

| Strategy | Pit Laps | Total Time | Gap to Best |
|----------|----------|------------|-------------|
| **1-Stop (M-H)** | 34 | 5986.25s | — |
| 2-Stop Optimal (M-H-M) | 25, 47 | 5992.62s | +6.37s |
| Piastri Actual (M-H-M) | 18, 42 | 5994.19s | +7.94s |

### 4. Monte Carlo Analysis (Notebook 05)

**1,000 simulations per strategy with randomized:**
- Tyre degradation (±15% variance)
- Pit stop execution (22.0s ± 0.3s, with 5% slow stop probability)
- Lap time variance (±0.15s per lap)

| Strategy | Mean Time | Std Dev | P5 (Best) | P95 (Worst) | Stops |
|----------|-----------|---------|-----------|-------------|-------|
| **1-Stop (L34)** | 5986.25s | 4.35s | 5978.70s | 5993.29s | 1 |
| Soft Start (L12, L35) | 5991.74s | 4.44s | 5984.52s | 5999.26s | 2 |
| Optimal (L25, L47) | 5992.62s | 2.86s | 5987.81s | 5997.38s | 2 |
| Actual (L18, L42) | 5994.19s | 2.99s | 5989.12s | 5998.85s | 2 |
| Aggressive (L15, L38) | 5996.28s | 3.65s | 5990.61s | 6002.41s | 2 |

**Head-to-Head Win Probabilities:**
- Optimal 2-Stop beats Actual: **65.6%**
- Actual beats Optimal: **34.4%**
- 1-Stop beats 2-Stop: **90.0%**

**Safety Car Analysis:**
- P(at least one SC in 70 laps): **75.7%**
- SC events favour 2-stop strategies (flexibility to pit under caution)

---

## Key Strategic Insight

> If traffic penalty is ~1.5s/lap (realistic for P5 vs P2 at Hungary), losing 3 positions costs approximately **45-60 seconds** over the remaining race distance. This vastly exceeds any lap-time gain from optimal pit timing.
>
> **Track position value >> tyre freshness benefit at Hungary.**

The Monte Carlo analysis shows that even the "optimal" strategy only beats alternatives **65.6%** of the time due to variance overlap. This quantifies why real teams make "conservative" decisions — they're managing variance, not just expected value.

---

## Project Structure

```
F1-Strategy-Hungary-2023/
├── 01_Tyre_Degradation_Physics.ipynb       # Physics foundations
├── 02_Multi_Car_Race_Simulation.ipynb      # Multi-car dynamics & dirty air
├── 03_Compound_Strategy_Optimisation.ipynb # Pit window optimization
├── 04_Real_Data_Validation.ipynb           # FastF1 validation
├── 05_Monte_Carlo_Uncertainty_Analysis.ipynb # Probabilistic analysis
├── README.md
├── TECHNIQUE_GLOSSARY.md
└── requirements.txt
```

### Notebook Progression

| Notebook | Purpose | Key Output |
|----------|---------|------------|
| **01** | Build physics-based tire/fuel model | `Car` class with lap time simulation |
| **02** | Add multi-car racing dynamics | `RaceSimulator` with dirty air & pit stops |
| **03** | Optimize compound selection & pit timing | Optimal pit windows via grid search |
| **04** | Validate against real FastF1 telemetry | MAE = 0.856s ✓ |
| **05** | Quantify uncertainty via Monte Carlo | Win probabilities & confidence intervals |

---

## Methodology

### Physics Model (Milliken & Milliken Reference)

**Lap Time = Base Pace + Tire Penalty + Fuel Effect + Dirty Air**

```
Tire Degradation:  ΔT = age × degradation_rate (s/lap)
Fuel Effect:       ΔT = fuel_load × 0.035 (s/kg)  
Dirty Air:         ΔT = max_penalty / (1 + (gap/ref_gap)²)
```

### Compound Characteristics (Hungary 2023)

| Compound | Pace Offset | Degradation Rate | Max Stint |
|----------|-------------|------------------|-----------|
| **SOFT** | -0.80s | 0.070 s/lap | 22 laps |
| **MEDIUM** | 0.00s (baseline) | 0.040 s/lap | 32 laps |
| **HARD** | +0.40s | 0.025 s/lap | 45 laps |

### Stochastic Components (Monte Carlo)

| Parameter | Distribution | Values |
|-----------|--------------|--------|
| Lap time variance | Normal | σ = 0.15s |
| Degradation variance | Normal | ±15% of base rate |
| Pit stop variance | Normal + outliers | μ = 22.0s, σ = 0.3s, 5% slow stop |
| Safety Car | Bernoulli | 2% probability per lap |

---

## What This Analysis Does

✓ Extracts real degradation rates from FastF1 telemetry  
✓ Validates physics-based model against actual race times (MAE = 0.856s)  
✓ Identifies optimal pit windows via grid search  
✓ Quantifies strategy uncertainty with Monte Carlo simulation  
✓ Computes head-to-head win probabilities between strategies  
✓ Models Safety Car impact on strategy outcomes  

## What This Analysis Does NOT Do

✗ Prove what Piastri "should have done" (requires position modelling)  
✗ Model competitor reactions or game-theoretic interactions  
✗ Account for tyre cliff behaviour (non-linear end-of-life degradation)  
✗ Model real-time decision-making under incomplete information  

---

## Limitations Acknowledged

1. **Linear degradation assumption** — Real tyres degrade non-linearly near end-of-life ("cliff")
2. **No position dynamics** — Model optimizes time, not track position
3. **Simplified traffic model** — Dirty air effects approximated, not measured per-car
4. **Single-agent optimization** — Does not model how competitors would react
5. **No tyre temperature modelling** — Grip changes with thermal state not captured
6. **Fuel load confounded** — True degradation likely worse than measured

---

## Future Work

To properly answer "should Piastri have extended his stint?", future development could include:

- [ ] Multi-car race simulation with position dynamics
- [ ] Non-linear tyre degradation models (cliff detection)
- [ ] Position-dependent pace models (dirty air quantified per gap)
- [ ] Probabilistic overtaking success rates by track position
- [ ] Game-theoretic strategy optimization (Nash equilibrium)
- [ ] Temperature-dependent grip model

---

##  Talking Points

**Q: Walk me through your modelling approach.**

> "I built a physics-based tire degradation model grounded in Milliken & Milliken principles, then progressively added complexity: multi-car dynamics with dirty air, compound-specific characteristics, and finally Monte Carlo uncertainty quantification. Each notebook builds on the previous, allowing me to isolate and validate each component."

**Q: What did validation reveal?**

> "The model achieved 0.856 seconds MAE against real FastF1 telemetry—below my 1.0s target. Interestingly, accuracy varied by stint: Stint 3 showed near-zero bias (MAE 0.534s), while early stints overpredicted by ~1.2s. This suggests fuel load effects are confounded with tire degradation early in the race."

**Q: What was the most surprising finding?**

> "That 'optimal' strategies only win about 65% of head-to-head comparisons against 'suboptimal' ones. Running 1,000 paired simulations, the mathematically optimal pit timing only beat the actual timing 65.6% of the time. The distributions overlap significantly due to variance. This quantifies why real teams make conservative decisions—they're managing variance, not just expected value."

**Q: How would you improve this model?**

> "Three main areas: First, model traffic as position-dependent rather than averaged—losing P2 to P5 costs ~45-60s in traffic. Second, add non-linear tire degradation to capture the 'cliff' effect. Third, include game-theoretic competitor reactions—real strategy is multi-agent, not single-agent optimization."

---

## Technical Stack

- **Python 3.8+**
- **FastF1** — Official F1 telemetry data
- **pandas / NumPy / SciPy** — Data manipulation
- **matplotlib / seaborn** — Visualization
- **tqdm** — Progress bars for Monte Carlo

---

## Getting Started

```bash
# Clone repository
git clone https://github.com/egyan175-creator/F1-Strategy-Hungary-2023.git
cd F1-Strategy-Hungary-2023

# Install dependencies
pip install -r requirements.txt

# Run notebooks in sequence (01 → 05)
jupyter notebook
```

---

## About

**Author:** Emmanuel Gyan  
**Education:** Final Year Aerospace Engineering, KNUST (Ghana)  
**Contact:** egyan175@gmail.com  
**GitHub:** [egyan175](https://github.com/egyan175)


---

## References

- FastF1 Documentation: https://docs.fastf1.dev/
- Milliken, W. & Milliken, D. (1995). *Race Car Vehicle Dynamics*
- Hungary 2023 race data via FastF1 API

---

## License

MIT License — free to use for educational purposes.

---

*This is an independent analysis project for educational and portfolio purposes. Not affiliated with any Formula 1 team. All data sourced from publicly available FastF1 telemetry.*

