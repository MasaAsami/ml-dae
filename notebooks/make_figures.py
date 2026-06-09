"""
Regenerate all manuscript figures as *vector* PDF (Editor comment J-8),
mirroring notebooks/tree_based_DEA.ipynb, plus:
  * improved Fig.4 (multi-leaf / single-leaf) labelling   -> Referee 2 #9 (H)
  * a new stability figure for the sensitivity analysis    -> D-2

Outputs are written into the LaTeX source directory so \\includegraphics picks
them up after switching the extensions from .png to .pdf.

Run:  python notebooks/make_figures.py
"""
import warnings
warnings.simplefilter("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

from revision_analysis import (generate_sim_data, fit_partition, dea_efficiency,
                               run_stability_realdata)

plt.style.use("ggplot")
TEXDIR = "ReviewerComments/元論文提出tex"

D_COL = "input_employee"
Y_COLS = ["output_salse", "output_profit"]
X_FEATS = ["input_capital", "input_assets", "prefecture_others", "prefecture_兵庫県",
           "prefecture_大阪府", "prefecture_愛知県", "prefecture_東京都", "prefecture_神奈川県"]


def add_efficiency(df_tree, leaf_ids, d_col, y_cols, label="dea_efficiency"):
    parts = []
    for _id in leaf_ids:
        g = df_tree.query("leaf_id == @_id").copy()
        g[label] = dea_efficiency(g[[d_col]].to_numpy(), g[list(y_cols)].to_numpy())
        parts.append(g)
    return pd.concat(parts)


# --------------------------------------------------------------------------- #
# Fig 2 (sim_regplot): global vs leaf-wise regressions on simulated data
# --------------------------------------------------------------------------- #
def fig_sim_regplot():
    df, feats = generate_sim_data(size=400, seed=2025)
    df_tree = fit_partition(df, feats, max_depth=3, min_samples_leaf=30)
    m1 = df_tree.query("model_id == 1")
    leaves = m1["leaf_id"].unique().tolist()
    pal = sns.color_palette(n_colors=len(leaves))

    fig, ax = plt.subplots(2, 2, figsize=(12, 11), dpi=150, sharex=True)
    sns.regplot(m1, x="target_input", y="output_1", color="black", ax=ax[0][0])
    sns.regplot(m1, x="target_input", y="output_2", color="black", ax=ax[0][1])
    for i, _id in enumerate(leaves):
        sub = m1.query("leaf_id == @_id")
        sns.regplot(sub, x="target_input", y="output_1", color=pal[i], ax=ax[1][0], ci=None)
        sns.regplot(sub, x="target_input", y="output_2", color=pal[i], ax=ax[1][1], ci=None)
    for a in ax.ravel():
        a.set_xlabel(r"Focal input $D$")
    ax[0][0].set_ylabel(r"Output $Y_1$"); ax[0][1].set_ylabel(r"Output $Y_2$")
    ax[1][0].set_ylabel(r"Output $Y_1$"); ax[1][1].set_ylabel(r"Output $Y_2$")
    ax[0][0].set_title("Global regression (ignoring $X$)")
    ax[0][1].set_title("Global regression (ignoring $X$)")
    ax[1][0].set_title("Leaf-wise regressions (after stratification)")
    ax[1][1].set_title("Leaf-wise regressions (after stratification)")
    fig.tight_layout()
    fig.savefig(f"{TEXDIR}/sim_regplot.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote sim_regplot.pdf")


# --------------------------------------------------------------------------- #
# Fig 3 (correlation-hist): within-leaf rank-correlation histogram, real data
# --------------------------------------------------------------------------- #
def load_gbiz():
    return (pd.read_csv("data/gbizinfo_df_japan2023.csv")
            .dropna(subset=[D_COL])
            .query("input_employee > 0 and input_assets > 0 and output_profit > 0"))


def fig_correlation_hist():
    gbiz = load_gbiz()
    df_tree = fit_partition(gbiz, X_FEATS, d_col=D_COL, max_depth=9,
                            min_samples_leaf=30, split_seed=2025)
    rows = []
    for _id, g in df_tree.groupby("leaf_id"):
        if len(g) < 5:
            continue
        rows.append({
            "cap": g[[D_COL, "input_capital"]].corr("spearman").values[0][1],
            "ast": g[[D_COL, "input_assets"]].corr("spearman").values[0][1],
        })
    cr = pd.DataFrame(rows)
    g_cap = gbiz[[D_COL, "input_capital"]].corr("spearman").values[0][1]
    g_ast = gbiz[[D_COL, "input_assets"]].corr("spearman").values[0][1]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    cr["cap"].hist(bins=50, ax=ax[0], alpha=0.5)
    cr["ast"].hist(bins=50, ax=ax[1], alpha=0.5)
    ax[0].axvline(g_cap, color="black", ls="dashed", label="Original (full-sample) correlation")
    ax[1].axvline(g_ast, color="black", ls="dashed", label="Original (full-sample) correlation")
    ax[0].axvline(cr["cap"].mean(), color="red", label="Avg. of within-leaf correlations")
    ax[1].axvline(cr["ast"].mean(), color="red", label="Avg. of within-leaf correlations")
    for a in ax:
        a.set_xlabel("rank correlation"); a.set_ylabel("frequency"); a.set_xlim(-0.5, 1)
    ax[0].set_title("(a) Capital stock"); ax[1].set_title("(b) Net assets")
    ax[1].legend(framealpha=0, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.tight_layout()
    fig.savefig(f"{TEXDIR}/correlation-hist.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote correlation-hist.pdf")
    return df_tree


def prep_realdata_dea(df_tree):
    """Replicate notebook cells 42-44: z-score outputs/D, ratios, pick 3 leaves."""
    def zscore(x):
        return (x - x.mean()) / x.std()
    dea_df = df_tree.copy()
    for c in Y_COLS + [D_COL]:
        dea_df[c] = zscore(dea_df[c])
        dea_df[c] += -dea_df[c].min() + 1
    dea_df["retio_1"] = dea_df[Y_COLS[0]] / dea_df[D_COL]
    dea_df["retio_2"] = dea_df[Y_COLS[1]] / dea_df[D_COL]
    q1, q2 = dea_df["retio_1"].quantile(0.99), dea_df["retio_2"].quantile(0.99)
    dea_df = dea_df.query("retio_1 <= @q1 and retio_2 <= @q2")
    sort_leaf = dea_df.groupby("leaf_id")[["retio_1", "retio_2"]].mean().sort_values("retio_1").index.tolist()
    corr_leaf = (dea_df.groupby("leaf_id")
                 .apply(lambda x: x[["retio_1", "retio_2"]].corr().values[0][1])
                 .sort_values().index.tolist())
    leaf_ids = [sort_leaf[0], sort_leaf[-1], corr_leaf[1]]
    return dea_df, leaf_ids


# --------------------------------------------------------------------------- #
# Fig 4 left (multi-leaf) and right (single-leaf) -- improved labelling (H)
# --------------------------------------------------------------------------- #
def fig_leafwise(df_tree):
    dea_df0, leaf_ids = prep_realdata_dea(df_tree)
    eff = add_efficiency(dea_df0, leaf_ids, D_COL, Y_COLS)
    pal = sns.color_palette(n_colors=len(leaf_ids))

    # ---- multi-leaf ----
    fig = plt.figure(figsize=(6.2, 4.6), dpi=150)
    for i, _id in enumerate(leaf_ids):
        d = eff.query("leaf_id == @_id")
        front = d.query("dea_efficiency > 0.99").sort_values("retio_1")
        fx = [0] + front["retio_1"].tolist() + [front.iloc[-1]["retio_1"]]
        fy = [front.iloc[0]["retio_2"]] + front["retio_2"].tolist() + [0]
        plt.plot(fx, fy, alpha=.7, color=pal[i], label=f"leaf {i + 1}: local DEA frontier")
        plt.scatter(front.retio_1, front.retio_2, s=110, marker="*", c=[pal[i]])
        ineff = d.query("dea_efficiency < 0.99")
        plt.scatter(ineff.retio_1, ineff.retio_2, alpha=.5, s=10, c=[pal[i]])
    plt.xlim(0, 1.8); plt.ylim(0, 2.0)
    plt.xlabel("Sales / #employees"); plt.ylabel("Profit / #employees")
    plt.legend(loc="lower left", fontsize=8); plt.grid(True)
    fig.tight_layout()
    fig.savefig(f"{TEXDIR}/multi-leaf.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote multi-leaf.pdf")

    # ---- single-leaf ----
    _id = leaf_ids[-1]
    d = eff.query("leaf_id == @_id")
    front = d.query("dea_efficiency > 0.99").sort_values("retio_1")
    # close the frontier with the left horizontal cap and the right vertical drop,
    # matching the multi-leaf panel (free-disposal boundary)
    fx = [0] + front["retio_1"].tolist() + [front["retio_1"].iloc[-1]]
    fy = [front.iloc[0]["retio_2"]] + front["retio_2"].tolist() + [0]
    fig = plt.figure(figsize=(6.6, 4.6), dpi=150)
    plt.fill_between(fx, 0, fy, color="w", alpha=.7)
    plt.plot(fx, fy, color="r", lw=1.2, alpha=.8, label="Local DEA frontier")
    plt.scatter(front.retio_1, front.retio_2, s=120, marker="*", c="r", label="Reference set (efficient)")
    ineff = d.query("dea_efficiency < 0.99")
    sc = plt.scatter(ineff.retio_1, ineff.retio_2, alpha=.6, c=ineff.dea_efficiency,
                     cmap="bwr", edgecolors="black")
    cbar = plt.colorbar(sc); cbar.set_label("within-leaf efficiency score")
    plt.xlim(0, 1.8); plt.ylim(0, 2.0)
    plt.xlabel("Sales / #employees"); plt.ylabel("Profit / #employees")
    plt.legend(loc="upper left"); plt.grid(True)
    fig.tight_layout()
    fig.savefig(f"{TEXDIR}/single-leaf.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote single-leaf.pdf")


# --------------------------------------------------------------------------- #
# New Fig (stability): within-leaf correlation across 50 train/test splits
# --------------------------------------------------------------------------- #
def fig_stability():
    res, gcorr = run_stability_realdata(seeds=range(50))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    res["input_capital"].hist(bins=20, ax=ax[0], alpha=0.6)
    res["input_assets"].hist(bins=20, ax=ax[1], alpha=0.6)
    ax[0].axvline(gcorr["input_capital"], color="black", ls="dashed", label="Original correlation")
    ax[1].axvline(gcorr["input_assets"], color="black", ls="dashed", label="Original correlation")
    ax[0].axvline(res["input_capital"].mean(), color="red", label="Mean over 50 splits")
    ax[1].axvline(res["input_assets"].mean(), color="red", label="Mean over 50 splits")
    for a in ax:
        a.set_xlabel("mean within-leaf rank correlation"); a.set_ylabel("frequency (splits)")
    ax[0].set_title("(a) Capital stock"); ax[1].set_title("(b) Net assets")
    ax[1].legend(framealpha=0, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.tight_layout()
    fig.savefig(f"{TEXDIR}/stability-hist.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote stability-hist.pdf")


if __name__ == "__main__":
    print("Regenerating figures as vector PDF ->", TEXDIR)
    fig_sim_regplot()
    df_tree = fig_correlation_hist()
    fig_leafwise(df_tree)
    fig_stability()
    print("Done.")
