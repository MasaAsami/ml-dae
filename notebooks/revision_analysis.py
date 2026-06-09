"""
Revision analyses for "Tree-based DEA for Focal Input Efficiency" (JDS2507-013).

Implements the two reviewer-requested additional analyses:
  G   -- Monte-Carlo Bias / MSE of the focal-input slope tau^(m), global vs leaf-wise
          (Referee 2 #7, Associate Editor).
  D-2 -- Stability / sensitivity of the propensity-tree partition
          (Referee 1, Referee 2 #4/#5, Associate Editor):
            * reproducibility under fixed random_state,
            * robustness of within-leaf confounding reduction to the train/test split,
            * robustness of the DEA efficient set (Jaccard) and efficiency ranking,
            * hyper-parameter sweep (max_depth, min_samples_leaf).

The data-generating process and the propensity-tree / leaf-wise pipeline mirror
notebooks/tree_based_DEA.ipynb so the numbers are directly comparable with the paper.

Run:  python notebooks/revision_analysis.py
"""
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import fmin_slsqp
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

TRUE_TAU = 1.0  # E[alpha^(m)] in the DGP (slopes ~ N(1, 0.05^2))


# --------------------------------------------------------------------------- #
# Data-generating process (identical structure to tree_based_DEA.ipynb cell 3)
# --------------------------------------------------------------------------- #
def generate_sim_data(size=400, seed=2025):
    rng = np.random.RandomState(seed)
    X = rng.normal(loc=0, scale=3, size=(size, 2))
    df = pd.DataFrame(X, columns=[f"input_{i + 1}" for i in range(2)])
    df["group_1"] = pd.qcut(df["input_1"], 3, labels=[1, 2, 3]).astype(int)
    df["group_2"] = pd.qcut(df["input_2"], 2, labels=[1, 2]).astype(int)
    df["group_prod"] = df["group_1"] + df["group_2"]

    def generate_D(g):
        if g == 5:
            return rng.normal(loc=6, scale=2)
        elif g == 4:
            return rng.normal(loc=4, scale=1.5)
        elif g == 3:
            return rng.normal(loc=3, scale=1.5)
        else:
            return rng.normal(loc=0, scale=2)

    df["target_input"] = df["group_prod"].apply(generate_D)
    other_input_cols = [c for c in df.columns if "input_" in c]
    df += 20
    assert df.min().min() > 0

    beta1 = [-1.25, -0.75]
    beta2 = [-0.75, -1.25]
    tau1 = rng.normal(1.0, 0.05, size)   # true unit-level slopes for Y1
    tau2 = rng.normal(1.0, 0.05, size)   # true unit-level slopes for Y2

    df["output_1"] = (df[other_input_cols].values @ beta1
                      + df["target_input"].values * tau1 + 50 + rng.normal(0, 1, size))
    df["output_2"] = (df[other_input_cols].values @ beta2
                      + df["target_input"].values * tau2 + 50 + rng.normal(0, 1, size))
    df = df.drop(columns=["group_1", "group_2"])
    return df, other_input_cols


def slope(d, x="target_input", y="output_1"):
    return stats.linregress(d[x], d[y]).slope


def fit_partition(df, feat_cols, d_col="target_input", max_depth=3,
                  min_samples_leaf=30, split_seed=2025, tree_seed=2025):
    """Cross-fitted propensity-tree partition (notebook cells 11-18)."""
    df1, df2 = train_test_split(df, train_size=0.5, random_state=split_seed)
    dtr1 = DecisionTreeRegressor(max_depth=max_depth, random_state=tree_seed,
                                 min_samples_leaf=min_samples_leaf)
    dtr2 = DecisionTreeRegressor(max_depth=max_depth, random_state=tree_seed,
                                 min_samples_leaf=min_samples_leaf)
    dtr1.fit(df1[feat_cols], df1[d_col])
    dtr2.fit(df2[feat_cols], df2[d_col])
    df_tree = pd.concat([
        df2.assign(leaf_id=[f"tree1_{i}" for i in dtr1.apply(df2[feat_cols])],
                   pred_d=dtr1.predict(df2[feat_cols]), model_id=1),
        df1.assign(leaf_id=[f"tree2_{i}" for i in dtr2.apply(df1[feat_cols])],
                   pred_d=dtr2.predict(df1[feat_cols]), model_id=2),
    ])
    return df_tree


# --------------------------------------------------------------------------- #
# G -- Monte-Carlo Bias / MSE of tau^(m): global vs leaf-wise
# --------------------------------------------------------------------------- #
def leafwise_weighted_slope(df_tree, y):
    """Sample-size-weighted average of within-leaf OLS slopes of y on D."""
    num = 0.0
    den = 0.0
    for _id, g in df_tree.groupby("leaf_id"):
        if len(g) < 3 or g["target_input"].std() == 0:
            continue
        n = len(g)
        num += n * slope(g, y=y)
        den += n
    return num / den if den else np.nan


def run_bias_mse(reps=200, size=400, base_seed=10000):
    rows = {"global": {"output_1": [], "output_2": []},
            "leafwise": {"output_1": [], "output_2": []}}
    for r in range(reps):
        df, feats = generate_sim_data(size=size, seed=base_seed + r)
        df_tree = fit_partition(df, feats)
        for y in ("output_1", "output_2"):
            rows["global"][y].append(slope(df, y=y))
            rows["leafwise"][y].append(leafwise_weighted_slope(df_tree, y))
    recs = []
    for method in ("global", "leafwise"):
        for y in ("output_1", "output_2"):
            est = np.array(rows[method][y], dtype=float)
            est = est[~np.isnan(est)]
            recs.append({
                "method": method, "output": y, "reps": len(est),
                "mean": est.mean(), "bias": est.mean() - TRUE_TAU,
                "sd": est.std(ddof=1), "mse": np.mean((est - TRUE_TAU) ** 2),
                "rmse": np.sqrt(np.mean((est - TRUE_TAU) ** 2)),
            })
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# Serial DEA (output-oriented; mirrors notebook's DEA class, no multiprocessing)
# --------------------------------------------------------------------------- #
def dea_efficiency(inputs, outputs):
    n, m = inputs.shape
    r = outputs.shape[1]
    eff = np.zeros(n)
    for unit in range(n):
        def target(x):
            in_w, out_w = x[:m], x[m:m + r]
            return np.dot(outputs[unit], out_w) / np.dot(inputs[unit], in_w)

        def constraints(x):
            in_w, out_w, lam = x[:m], x[m:m + r], x[m + r:]
            t = target(x)
            c = []
            for i in range(m):
                c.append(t * inputs[unit, i] - np.dot(inputs[:, i], lam))
            for o in range(r):
                c.append(np.dot(outputs[:, o], lam) - outputs[unit, o])
            c.extend(lam.tolist())
            return np.array(c)

        x0 = np.random.RandomState(unit).rand(m + r + n) - 0.5
        res = fmin_slsqp(target, x0, f_ieqcons=constraints, disp=False)
        in_w, out_w = res[:m], res[m:m + r]
        eff[unit] = (outputs @ out_w / (inputs @ in_w))[unit]
    return eff


def efficient_units(df_tree, d_col="target_input", y_cols=("output_1", "output_2"),
                    thresh=0.99):
    """Return set of unit indices flagged efficient (theta>thresh) across all leaves."""
    eff_set = set()
    scores = {}
    for _id, g in df_tree.groupby("leaf_id"):
        if len(g) < 3:
            continue
        e = dea_efficiency(g[[d_col]].to_numpy(), g[list(y_cols)].to_numpy())
        for idx, val in zip(g.index, e):
            scores[idx] = val
            if val > thresh:
                eff_set.add(idx)
    return eff_set, scores


# --------------------------------------------------------------------------- #
# D-2 -- stability / sensitivity
# --------------------------------------------------------------------------- #
def within_leaf_mean_corr(df_tree, d_col, x_cols):
    """Mean (over leaves) within-leaf Spearman corr |D, x| for each covariate."""
    out = {}
    for xc in x_cols:
        corrs = []
        for _id, g in df_tree.groupby("leaf_id"):
            if len(g) < 5 or g[d_col].std() == 0 or g[xc].std() == 0:
                continue
            corrs.append(g[[d_col, xc]].corr(method="spearman").values[0][1])
        out[xc] = np.nanmean(corrs)
    return out


def run_stability_realdata(seeds=range(50)):
    d_col = "input_employee"
    x_cols = ["input_capital", "input_assets"]
    feat_cols = ["input_capital", "input_assets", "prefecture_others",
                 "prefecture_兵庫県", "prefecture_大阪府", "prefecture_愛知県",
                 "prefecture_東京都", "prefecture_神奈川県"]
    gbiz = (pd.read_csv("data/gbizinfo_df_japan2023.csv")
            .dropna(subset=[d_col])
            .query("input_employee > 0 and input_assets > 0 and output_profit > 0"))
    global_corr = {xc: gbiz[[d_col, xc]].corr(method="spearman").values[0][1]
                   for xc in x_cols}
    recs = []
    for s in seeds:
        df_tree = fit_partition(gbiz, feat_cols, d_col=d_col, max_depth=9,
                                min_samples_leaf=30, split_seed=int(s))
        mc = within_leaf_mean_corr(df_tree, d_col, x_cols)
        recs.append({"seed": int(s), "n_leaves": df_tree["leaf_id"].nunique(), **mc})
    res = pd.DataFrame(recs)
    return res, global_corr


def run_stability_sim(seeds=range(20)):
    """Same simulated dataset (seed=2025), vary only the train/test split seed.
    Track DEA efficient-set Jaccard and efficiency rank correlation across splits."""
    df, feats = generate_sim_data(size=400, seed=2025)
    eff_sets, score_series = [], []
    leaf_corr = []
    for s in seeds:
        df_tree = fit_partition(df, feats, split_seed=int(s))
        lc = within_leaf_mean_corr(df_tree, "target_input", feats)
        leaf_corr.append(np.nanmean(list(lc.values())))
        eset, scores = efficient_units(df_tree)
        eff_sets.append(eset)
        score_series.append(pd.Series(scores))
    # pairwise Jaccard of efficient sets
    jac = []
    for i in range(len(eff_sets)):
        for j in range(i + 1, len(eff_sets)):
            a, b = eff_sets[i], eff_sets[j]
            if a or b:
                jac.append(len(a & b) / len(a | b))
    # pairwise Spearman of efficiency scores on shared units
    rank = []
    for i in range(len(score_series)):
        for j in range(i + 1, len(score_series)):
            common = score_series[i].index.intersection(score_series[j].index)
            if len(common) > 10:
                rank.append(stats.spearmanr(score_series[i][common],
                                            score_series[j][common]).correlation)
    return {
        "n_seeds": len(list(seeds)),
        "mean_leaf_corr": float(np.mean(leaf_corr)),
        "sd_leaf_corr": float(np.std(leaf_corr, ddof=1)),
        "mean_jaccard": float(np.mean(jac)),
        "sd_jaccard": float(np.std(jac, ddof=1)),
        "mean_rank_corr": float(np.nanmean(rank)),
        "sd_rank_corr": float(np.nanstd(rank, ddof=1)),
    }


def run_hyperparam_sweep():
    d_col = "input_employee"
    x_cols = ["input_capital", "input_assets"]
    feat_cols = ["input_capital", "input_assets", "prefecture_others",
                 "prefecture_兵庫県", "prefecture_大阪府", "prefecture_愛知県",
                 "prefecture_東京都", "prefecture_神奈川県"]
    gbiz = (pd.read_csv("data/gbizinfo_df_japan2023.csv")
            .dropna(subset=[d_col])
            .query("input_employee > 0 and input_assets > 0 and output_profit > 0"))
    recs = []
    for md in (5, 7, 9, 11):
        for msl in (20, 30, 50, 100):
            df_tree = fit_partition(gbiz, feat_cols, d_col=d_col, max_depth=md,
                                    min_samples_leaf=msl, split_seed=2025)
            mc = within_leaf_mean_corr(df_tree, d_col, x_cols)
            recs.append({"max_depth": md, "min_samples_leaf": msl,
                         "n_leaves": df_tree["leaf_id"].nunique(),
                         "mean_corr": float(np.mean(list(mc.values())))})
    return pd.DataFrame(recs)


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    print("\n" + "=" * 70)
    print("G -- Bias / MSE of tau^(m): global vs leaf-wise (200 reps, true=1.0)")
    print("=" * 70)
    bm = run_bias_mse(reps=200)
    print(bm.to_string(index=False))
    bm.to_csv("notebooks/out_bias_mse.csv", index=False)

    print("\n" + "=" * 70)
    print("D-2a -- Real data: within-leaf confounding reduction over 50 splits")
    print("=" * 70)
    res, gcorr = run_stability_realdata(seeds=range(50))
    print("Global Spearman corr (D vs covariate):", {k: round(v, 4) for k, v in gcorr.items()})
    print(res.describe().loc[["mean", "std", "min", "max"]].to_string())
    res.to_csv("notebooks/out_stability_realdata.csv", index=False)

    print("\n" + "=" * 70)
    print("D-2b -- Simulation: DEA efficient-set / ranking stability over splits")
    print("=" * 70)
    sim = run_stability_sim(seeds=range(20))
    for k, v in sim.items():
        print(f"  {k:18s}: {v}")

    print("\n" + "=" * 70)
    print("D-2c -- Hyper-parameter sweep (real data): leaves vs mean within-leaf corr")
    print("=" * 70)
    hp = run_hyperparam_sweep()
    print(hp.to_string(index=False))
    hp.to_csv("notebooks/out_hyperparam_sweep.csv", index=False)
    print("\nDone.")
