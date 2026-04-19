# Network Topology as a Buffer Against Secularization

Agent-based model of secularization dynamics under varying network topology.

**Author:** Nikita Obraztsov | European University at St. Petersburg  

## Overview

Why do some religious communities persist under modernization while others secularize rapidly? This model tests whether cross-group network connectivity functions as a structural buffer against secularization, independently of theological or cultural factors.

500 agents (400 secular, 100 religious) interact on a Stochastic Block Model network. Cross-group connectivity (P_cross) varies across 6 conditions × 10 replications. Three mechanisms operate per step: ESH-driven modernization pressure, Deffuant bounded-confidence peer influence, and a social support buffer.

Key result: religiosity at P_cross=0.001 is approximately 9× higher than at P_cross=0.15. A tipping point is identified at P_cross ≈ 0.06–0.10.

## Files

- `Model_Obraztsov.py` — simulation: ablation study + sensitivity analysis
- `Visualization_Obraztsov.py` — figures (4 publication-ready plots)

## How to run

```bash
pip install numpy networkx pandas matplotlib tqdm
python3 Obraztsov_ABM.py      # runs simulation, saves CSVs to results/
python3 visualize_results.py  # generates figures
```

## Model mechanisms

1. **ESH pressure** — security grows each step, secularization rate proportional to perceived security (Inglehart 2020)
2. **Social support buffer** — co-religionist neighbors reduce effective decay rate (Smith 1998; Stroope 2011)
3. **Deffuant peer influence** — agents shift beliefs toward neighbors within confidence threshold ε (Deffuant et al. 2000)

## Reference

Obraztsov, N. A. Network Topology as a Buffer Against Secularization: Structural Mechanisms in Agent-Based Models. 
