#!/usr/bin/env python3
"""
Agent-based model of secularization dynamics under varying network topology.

Implements three mechanisms:
- Existential Security Hypothesis (Inglehart 2020)
- Deffuant bounded-confidence peer influence (Deffuant et al. 2000)
- Social support buffer (Smith 1998, Stroope 2011)

Runs ablation study (4 model variants) and sensitivity analysis (3 buffer values).
Results saved to results/ablation/ and results/sensitivity/ as CSV + JSON config.
Then run visualize_results.py to generate figures.

Nikita Obraztsov | European University at St. Petersburg
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "ablation").mkdir(exist_ok=True)
(RESULTS_DIR / "sensitivity").mkdir(exist_ok=True)


# --- Config ---


@dataclass
class Config:
    # Population: 400:100 reflects minority status of observant communities
    n_secular: int = 400
    n_religious: int = 100

    # Time: 100-step burn-in before modernization pressure begins
    steps: int = 300
    intervention_start: int = 100
    record_every: int = 10

    # Deffuant bounded-confidence model (Deffuant et al. 2000)
    epsilon: float = 0.50  # confidence threshold
    mu: float = 0.15  # convergence rate per interaction

    # Existential Security Hypothesis (Inglehart 2020)
    security_growth: float = 0.002  # security increase per step
    secularization_rate: float = 0.006  # base religiosity decay
    religious_buffer: float = 0.5  # compensatory control on perceived security

    # Network: equal within-group density eliminates density confound
    p_in: float = 0.15

    # Experiment
    cross_probs: tuple = (0.001, 0.01, 0.03, 0.06, 0.10, 0.15)
    n_replications: int = 10
    master_seed: int = 42


def validate(cfg: Config) -> None:
    assert 0 < cfg.intervention_start < cfg.steps
    assert 0 < cfg.epsilon <= 1
    assert 0 < cfg.mu <= 0.5
    assert 0 < cfg.p_in < 1
    assert all(0 < p < 1 for p in cfg.cross_probs)


# Ablation: four variants isolate mechanism contributions
MODEL_VARIANTS = {
    "reference": {"use_deffuant": False, "use_buffer": False},
    "deffuant_only": {"use_deffuant": True, "use_buffer": False},
    "buffer_only": {"use_deffuant": False, "use_buffer": True},
    "full": {"use_deffuant": True, "use_buffer": True},
}

# Sensitivity: buffer strength tested at three levels
BUFFER_VALUES = [0.5, 0.7, 0.9]


# --- Simulation ---


def make_seeds(master_seed: int, n: int) -> list[int]:
    """Derive n independent seeds from a single master seed via SeedSequence."""
    ss = np.random.SeedSequence(master_seed)
    return [int(child.generate_state(1)[0]) for child in ss.spawn(n)]


def run_simulation(
    p_cross: float,
    cfg: Config,
    use_deffuant: bool = True,
    use_buffer: bool = True,
    social_support_buffer: float = 0.7,
    seed: int = 42,
) -> pd.DataFrame:
    """
    One simulation run for a given cross-group connectivity level.
    Returns a time series of mean religiosity in the religious group.
    """
    assert 0 < p_cross < 1
    assert 0 <= social_support_buffer <= 1

    rng = np.random.default_rng(seed)

    # SBM network with equal within-group density
    G = nx.stochastic_block_model(
        [cfg.n_secular, cfg.n_religious],
        [[cfg.p_in, p_cross], [p_cross, cfg.p_in]],
        seed=int(seed),
    )

    # Initialize agent attributes (group, religiosity, security)
    node_type = {}
    R = {}  # religiosity in [0, 1]
    S = {}  # objective security in [0, 1]

    for i in sorted(G.nodes()):
        secular = i < cfg.n_secular
        node_type[i] = "Secular" if secular else "Religious"
        R[i] = rng.uniform(0.0, 0.3) if secular else rng.uniform(0.7, 1.0)
        S[i] = rng.uniform(0.6, 1.0) if secular else rng.uniform(0.2, 0.5)

    rel_nodes = [i for i in sorted(G.nodes()) if node_type[i] == "Religious"]

    # Precompute neighbors and cross-edge share (network is static)
    neighbors = {i: sorted(G.neighbors(i)) for i in sorted(G.nodes())}
    n_edges = G.number_of_edges()
    cross_edges = sum(1 for u, v in G.edges() if node_type[u] != node_type[v])
    pct_cross = cross_edges / n_edges if n_edges > 0 else 0.0

    records = []

    for t in range(cfg.steps):
        post = t >= cfg.intervention_start
        R_prev = R.copy()
        S_prev = S.copy()

        for i in sorted(G.nodes()):

            # Security update; perceived security dampened by religiosity
            # (compensatory control, Kay et al. 2010)
            new_S = min(1.0, S_prev[i] + cfg.security_growth) if post else S_prev[i]
            S[i] = new_S
            perceived_S = new_S * (1.0 - cfg.religious_buffer * R_prev[i])

            # Social support buffer: co-religionist neighbors reduce decay rate
            nbrs = neighbors[i]
            if use_buffer and nbrs and node_type[i] == "Religious":
                rel_count = sum(1 for n in nbrs if node_type[n] == "Religious")
                support = rel_count / len(nbrs)
                eff_rate = cfg.secularization_rate * (
                    1.0 - social_support_buffer * support
                )
            else:
                eff_rate = cfg.secularization_rate

            decay = eff_rate * perceived_S if post else 0.0

            # Deffuant peer influence: shift toward neighbor if within epsilon
            pull = 0.0
            if use_deffuant and nbrs:
                j = nbrs[rng.integers(len(nbrs))]
                if abs(R_prev[j] - R_prev[i]) <= cfg.epsilon:
                    pull = cfg.mu * (R_prev[j] - R_prev[i])

            R[i] = float(np.clip(R_prev[i] - decay + pull, 0.0, 1.0))

        if t % cfg.record_every == 0 or t == cfg.steps - 1:
            mean_r = float(np.mean([R[n] for n in rel_nodes]))
            supports = [
                sum(1 for x in neighbors[n] if node_type[x] == "Religious")
                / len(neighbors[n])
                for n in rel_nodes
                if neighbors[n]
            ]
            records.append(
                {
                    "Step": t,
                    "Mean_R": mean_r,
                    "Pct_Cross_Edges": pct_cross,
                    "Avg_Social_Support": float(np.mean(supports)) if supports else 0.0,
                }
            )

    return pd.DataFrame(records)


def run_batch(
    cfg: Config,
    use_deffuant: bool = True,
    use_buffer: bool = True,
    social_support_buffer: float = 0.7,
    label: str = "full",
) -> pd.DataFrame:
    """
     Parameter sweep across all p_cross values with n_replications replications.

    Seeds are drawn from a pre-generated matrix of shape
    (len(cross_probs) x n_replications), derived via SeedSequence from
    master_seed. Each (p_cross, replication) pair receives a unique seed,
    ensuring that replications are statistically independent both within
    and across connectivity conditions. The previous approach of offsetting
    a shared seed pool by int(p_c * 1_000_000) produced unpredictable seed
    collisions across conditions and is replaced here.
    """
    all_seeds = make_seeds(cfg.master_seed, cfg.n_replications * len(cfg.cross_probs))
    results = []

    for p_idx, p_c in enumerate(tqdm(cfg.cross_probs, desc=label, leave=False)):
        reps = []
        for rep_idx in range(cfg.n_replications):
            seed = all_seeds[p_idx * cfg.n_replications + rep_idx]
            df = run_simulation(
                p_cross=p_c,
                cfg=cfg,
                use_deffuant=use_deffuant,
                use_buffer=use_buffer,
                social_support_buffer=social_support_buffer,
                seed=seed,
            )
            df["rep"] = rep_idx
            reps.append(df)

        combined = pd.concat(reps, ignore_index=True)
        agg = (
            combined.groupby("Step")[
                ["Mean_R", "Pct_Cross_Edges", "Avg_Social_Support"]
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        agg.columns = [
            "Step",
            "Mean_R",
            "Mean_R_std",
            "Pct_Cross_Edges",
            "Pct_Cross_Edges_std",
            "Avg_Social_Support",
            "Avg_Social_Support_std",
        ]
        agg["p_cross"] = p_c
        agg["model"] = label
        agg["buffer_strength"] = social_support_buffer
        results.append(agg)
    return pd.concat(results, ignore_index=True)


def save_results(df: pd.DataFrame, cfg: Config, subfolder: str, filename: str) -> None:
    """Persist results and config for reproducibility."""
    out_dir = RESULTS_DIR / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(out_dir / f"{filename}.csv", index=False)
        with open(out_dir / f"{filename}_config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2, default=list)
        log.info(f"Saved: {out_dir / filename}.csv")
    except OSError as e:
        log.error(f"Could not save {filename}: {e}")


# --- Main ---

if __name__ == "__main__":
    CFG = Config()
    validate(CFG)

    log.info("Starting ablation study")
    for key, settings in tqdm(MODEL_VARIANTS.items(), desc="Ablation"):
        df = run_batch(cfg=CFG, label=key, social_support_buffer=0.7, **settings)
        save_results(df, CFG, subfolder="ablation", filename=f"ablation_{key}")

    log.info("Starting sensitivity analysis")
    for buf in tqdm(BUFFER_VALUES, desc="Sensitivity"):
        df = run_batch(
            cfg=CFG,
            use_deffuant=True,
            use_buffer=True,
            social_support_buffer=buf,
            label=f"full_buf{buf}",
        )
        save_results(df, CFG, subfolder="sensitivity", filename=f"sensitivity_buf{buf}")
