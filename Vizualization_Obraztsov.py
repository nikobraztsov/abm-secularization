#!/usr/bin/env python3
"""
Figures for "Network Topology as a Buffer Against Secularization."
Run AFTER Obraztsov_ABM.py has completed.

Nikita Obraztsov | European University at St. Petersburg 
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")

# Must match Config in Obraztsov_ABM.py
CROSS_PROBS = [0.001, 0.01, 0.03, 0.06, 0.10, 0.15]
CROSS_LABELS = [str(p) for p in CROSS_PROBS]
INTERVENTION_START = 100
BUFFER_VALUES = [0.5, 0.7, 0.9]

NAVY = "#1A2F5E"
COLORS_6 = ["#1A2F5E", "#1F4788", "#4472C4", "#7BA7D4", "#AACBE8", "#CADCFC"]


# --- Load ---

def load_ablation():
    ablation = {}
    for variant in ["reference", "deffuant_only", "buffer_only", "full"]:
        path = RESULTS_DIR / "ablation" / f"ablation_{variant}.csv"
        if path.exists():
            ablation[variant] = pd.read_csv(path)
        else:
            print(f"Warning: {path} not found")
    return ablation


def load_sensitivity():
    sensitivity = {}
    for buf in BUFFER_VALUES:
        path = RESULTS_DIR / "sensitivity" / f"sensitivity_buf{buf}.csv"
        if path.exists():
            sensitivity[buf] = pd.read_csv(path)
        else:
            print(f"Warning: {path} not found")
    return sensitivity


# --- Figures ---

def make_figures(ablation, sensitivity):

    # Fig 1: religiosity trajectories by p_cross (full model)
    df_full = ablation["full"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for p_c, col, lbl in zip(CROSS_PROBS, COLORS_6, CROSS_LABELS):
        sub = df_full[df_full["p_cross"] == p_c].sort_values("Step")
        ax.plot(sub["Step"], sub["Mean_R"], color=col, lw=2, label=f"P_cross = {lbl}")
        ax.fill_between(
            sub["Step"],
            sub["Mean_R"] - sub["Mean_R_std"],
            sub["Mean_R"] + sub["Mean_R_std"],
            color=col, alpha=0.12,
        )
    ax.axvline(INTERVENTION_START, color="gray", lw=1.2, ls="--", alpha=0.7)
    ax.text(INTERVENTION_START + 3, 0.88, "Modernization\nbegins", fontsize=8, color="gray")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Mean Religiosity (Religious Group)")
    ax.set_title("Figure 1. Religiosity trajectories by cross-group connectivity", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig1_trajectories.png", dpi=180)
    plt.show()

    # Fig 2: final religiosity vs p_cross — tipping point
    final = (
        df_full
        .loc[df_full.groupby("p_cross")["Step"].idxmax()]
        .copy()
        .sort_values("p_cross")
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar_colors = COLORS_6[:3] + ["#E8894A", "#D05020", "#B03010"]
    bars = ax.bar(
        [str(p) for p in final["p_cross"]],
        final["Mean_R"],
        color=bar_colors, width=0.6,
    )
    ax.errorbar(
        range(len(final)), final["Mean_R"], yerr=final["Mean_R_std"],
        fmt="none", color="#333", capsize=4, lw=1.2,
    )
    for bar, val in zip(bars, final["Mean_R"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=NAVY,
        )
    ax.axvspan(2.5, 4.5, color="#FFF3E0", alpha=0.55, zorder=0)
    ax.text(3.5, 0.88, "Tipping point zone\n(0.06 – 0.10)",
            ha="center", fontsize=8.5, color="#C05010", style="italic")
    ax.set_xlabel("Cross-Group Connectivity (P_cross)")
    ax.set_ylabel("Final Mean Religiosity (Step 300)")
    ax.set_title("Figure 2. Final religiosity as a function of cross-group connectivity", fontsize=11)
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig2_tipping_point.png", dpi=180)
    plt.show()

    # Fig 3: ablation at p_cross=0.001
    rows = []
    for key, df in ablation.items():
        last = df.loc[df.groupby("p_cross")["Step"].idxmax()].copy()
        last["variant"] = key
        rows.append(last)
    all_variants = pd.concat(rows, ignore_index=True)
    most_segregated = all_variants[all_variants["p_cross"] == 0.001].copy()

    variant_order = ["reference", "deffuant_only", "buffer_only", "full"]
    variant_labels = ["Reference\n(ESH only)", "Deffuant\nonly", "Buffer\nonly", "Full\nmodel"]
    abl_colors = [COLORS_6[5], COLORS_6[3], COLORS_6[1], NAVY]

    vals = [most_segregated.loc[most_segregated["variant"] == v, "Mean_R"].values[0]
            for v in variant_order]
    errs = [most_segregated.loc[most_segregated["variant"] == v, "Mean_R_std"].values[0]
            for v in variant_order]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(variant_labels, vals, color=abl_colors, height=0.5)
    ax.errorbar(vals, range(len(vals)), xerr=errs,
                fmt="none", color="#333", capsize=4, lw=1.2)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9, fontweight="bold", color=NAVY)
    ax.set_xlabel("Final Mean Religiosity (P_cross = 0.001)")
    ax.set_title("Figure 3. Ablation study: mechanism contributions", fontsize=11)
    ax.set_xlim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig3_ablation.png", dpi=180)
    plt.show()

    # Fig 4: sensitivity — buffer strength at p_cross=0.001
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sens_colors = [COLORS_6[4], NAVY, COLORS_6[2]]
    sens_styles = ["--", "-", ":"]
    for buf, col, ls in zip(BUFFER_VALUES, sens_colors, sens_styles):
        sub = sensitivity[buf][sensitivity[buf]["p_cross"] == 0.001].sort_values("Step")
        ax.plot(sub["Step"], sub["Mean_R"], color=col, lw=2, ls=ls, label=f"Buffer = {buf}")
        ax.fill_between(
            sub["Step"],
            sub["Mean_R"] - sub["Mean_R_std"],
            sub["Mean_R"] + sub["Mean_R_std"],
            color=col, alpha=0.1,
        )
    ax.axvline(INTERVENTION_START, color="gray", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Mean Religiosity (Religious Group)")
    ax.set_title("Figure 4. Sensitivity analysis: buffer strength (P_cross = 0.001)", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig4_sensitivity.png", dpi=180)
    plt.show()


# --- Main ---

if __name__ == "__main__":
    ablation = load_ablation()
    sensitivity = load_sensitivity()

    if not ablation or not sensitivity:
        print("No results found. Run Obraztsov_ABM.py first.")
        raise SystemExit(1)

    make_figures(ablation, sensitivity)

    df_full = ablation["full"]
    final = (
        df_full
        .loc[df_full.groupby("p_cross")["Step"].idxmax()]
        .copy()
        .sort_values("p_cross")
    )
    r_max, r_min = final["Mean_R"].max(), final["Mean_R"].min()

    print(f"\nMax religiosity (P_cross=0.001): R = {r_max:.3f}")
    print(f"Min religiosity (P_cross=0.15):  R = {r_min:.3f}")
    print(f"Ratio: {r_max / r_min:.1f}x\n")
    print(final[["p_cross", "Mean_R", "Mean_R_std"]].to_string(index=False))

    print("\nAblation (P_cross = 0.001):")
    for key in ["reference", "deffuant_only", "buffer_only", "full"]:
        row = ablation[key][ablation[key]["p_cross"] == 0.001].sort_values("Step").iloc[-1]
        print(f"  {key:15s}: R = {row['Mean_R']:.3f} ± {row['Mean_R_std']:.3f}")

    print("\nFigures saved to results/")
