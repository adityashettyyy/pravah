"""Advanced leg-delay model: tuned gradient boosting + stacking + calibration + conformal sets
+ conformalized quantile regression for delay minutes + SHAP.

Protocol (the test set is touched exactly once, at the end)
  train 70% | validation 15% (split by trip into val_a and val_b) | test 15%, split by trip
  * hyper-parameters: Optuna, 3-fold grouped CV *inside train*, encodings rebuilt inside every fold
  * stacking meta-learner: trained on out-of-fold predictions inside train
  * val_a: probability calibration (Platt scaling) and decision threshold
  * val_b: conformal calibration (class-conditional), coverage guarantee 90%
  * test: all reported numbers, with trip-level cluster bootstrap CIs
Extra checks: out-of-time split (train on the first 70% of the calendar, test on the last 30%),
cold-start legs (hubs with < 5 training legs).

Run from the repo root:  python -m advanced.train
"""
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

from paths import HUBS_PATH
from ml.problem import load_clean
from ml.features import get_xy, split_indices, build_preprocessor
from advanced.features import FeatureContext, FEATURES, Platt

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = os.path.dirname(os.path.abspath(__file__))
ART, REP = os.path.join(HERE, "artifacts"), os.path.join(HERE, "reports")
ALPHA = 0.10
SEED = 42
N_TRIALS = int(os.environ.get("PRAVAH_TRIALS", 30))
N_BOOT = int(os.environ.get("PRAVAH_BOOT", 500))


# ------------------------------------------------------------------ helpers
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
               for b in range(bins) if (idx == b).any())


def metrics(y, p, thr):
    pred = p >= thr
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "brier": brier_score_loss(y, p), "ece": ece(y, p),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred)}


def best_f1_threshold(y, p):
    grid = np.linspace(0.1, 0.8, 71)
    return float(grid[np.argmax([f1_score(y, p >= t) for t in grid])])


def cluster_boot(y, preds, groups, fn, n=N_BOOT, seed=0):
    """Resample whole trips. preds is a dict name -> probs. Returns CI per name and paired diffs."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    members = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    out = {k: [] for k in preds}
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        if y[idx].min() == y[idx].max():
            continue
        for k, p in preds.items():
            out[k].append(fn(y[idx], p[idx]))
    return {k: np.array(v) for k, v in out.items()}


def lgbm(params, seed=SEED):
    return LGBMClassifier(**params, random_state=seed, n_jobs=1, verbose=-1)


XGB_PARAMS = dict(n_estimators=500, max_depth=4, learning_rate=0.04, subsample=0.85,
                  colsample_bytree=0.8, min_child_weight=3, reg_lambda=2.0, n_jobs=1,
                  eval_metric="logloss", random_state=SEED)


def et_model():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("et", ExtraTreesClassifier(150, min_samples_leaf=6, max_features=0.5,
                                                 n_jobs=1, random_state=SEED))])


def baseline_model():
    """The original XGBoost pipeline of ml/train.py (one-hot top-40 hubs), retrained on train only."""
    from sklearn.pipeline import Pipeline as P
    clf = XGBClassifier(colsample_bytree=1.0, learning_rate=0.15, max_depth=5, min_child_weight=1,
                        n_estimators=800, reg_lambda=2.0, subsample=0.9, n_jobs=1,
                        eval_metric="logloss", random_state=SEED)
    return P([("pre", build_preprocessor(scale=False)), ("clf", clf)])


def build_fold_matrices(df, hubs, groups, n_splits=3):
    """Per-fold feature matrices with encodings fitted inside the fold (no leakage)."""
    folds = []
    skf = StratifiedGroupKFold(n_splits, shuffle=True, random_state=SEED)
    for tr, va in skf.split(df, df["is_delayed"], groups):
        ctx = FeatureContext(hubs).fit(df.iloc[tr])
        Xtr = ctx.transform(df.iloc[tr], train_oof_groups=groups.iloc[tr].to_numpy())
        Xva = ctx.transform(df.iloc[va])
        folds.append((tr, va, Xtr, df["is_delayed"].iloc[tr].to_numpy(), Xva,
                      df["is_delayed"].iloc[va].to_numpy()))
    return folds


def tune(folds):
    def objective(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators", 200, 900),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 8, 64, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 0.1, 20, log=True),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True))
        sc = []
        for _, _, Xtr, ytr, Xva, yva in folds:
            sc.append(average_precision_score(yva, lgbm(p).fit(Xtr, ytr).predict_proba(Xva)[:, 1]))
        return float(np.mean(sc))
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params, study.best_value


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    os.makedirs(ART, exist_ok=True)
    os.makedirs(REP, exist_ok=True)
    df = load_clean().reset_index(drop=True)
    hubs = pd.read_csv(HUBS_PATH)
    X0, y_all, groups_all = get_xy(df)
    tr, va, te = split_indices(groups_all)
    gva = groups_all.iloc[va]
    a_idx, b_idx = next(GroupShuffleSplit(1, test_size=0.5, random_state=SEED).split(va, groups=gva))
    va_a, va_b = va[a_idx], va[b_idx]
    d_tr, d_va_a, d_va_b, d_te = (df.iloc[i].reset_index(drop=True) for i in (tr, va_a, va_b, te))
    y_tr, y_a, y_b, y_te = (d["is_delayed"].to_numpy() for d in (d_tr, d_va_a, d_va_b, d_te))
    g_tr, g_te = groups_all.iloc[tr].reset_index(drop=True), groups_all.iloc[te].reset_index(drop=True)
    log = [f"train {len(tr):,} | val_a {len(va_a):,} | val_b {len(va_b):,} | test {len(te):,} legs "
           f"(split by trip; test delayed rate {y_te.mean():.1%})"]

    # --- features
    ctx = FeatureContext(hubs).fit(d_tr)
    X_tr = ctx.transform(d_tr, train_oof_groups=g_tr.to_numpy())
    X_a, X_b, X_te = ctx.transform(d_va_a), ctx.transform(d_va_b), ctx.transform(d_te)

    # --- tuning (CV inside train, encodings rebuilt per fold)
    folds = build_fold_matrices(d_tr, hubs, g_tr)
    best, cv_pr = tune(folds)
    log.append(f"Optuna {N_TRIALS} trials, 3-fold grouped CV inside train: best CV PR-AUC {cv_pr:.3f}")
    log.append("best LightGBM params: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                     for k, v in best.items()}))
    lgb_params = dict(best, subsample_freq=1)

    # --- out-of-fold predictions for the stacker (inside train)
    oof = np.zeros((len(d_tr), 3))
    for tr_i, va_i, Xf_tr, yf_tr, Xf_va, _ in folds:
        for j, m in enumerate([lgbm(lgb_params), XGBClassifier(**XGB_PARAMS), et_model()]):
            oof[va_i, j] = m.fit(Xf_tr, yf_tr).predict_proba(Xf_va)[:, 1]
    logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    meta = LogisticRegression(C=1.0).fit(logit(oof), y_tr)

    # --- final base models on full train
    base = {"LightGBM": lgbm(lgb_params).fit(X_tr, y_tr),
            "XGBoost": XGBClassifier(**XGB_PARAMS).fit(X_tr, y_tr),
            "ExtraTrees": et_model().fit(X_tr, y_tr)}
    names = list(base)

    def stack_p(X):
        return meta.predict_proba(logit(np.column_stack([base[n].predict_proba(X)[:, 1] for n in names])))[:, 1]

    def prob(name, X):
        return stack_p(X) if name == "Stack" else base[name].predict_proba(X)[:, 1]

    X_va = pd.concat([X_a, X_b])
    y_va = np.r_[y_a, y_b]
    val_pr = {n: average_precision_score(y_va, prob(n, X_va)) for n in names + ["Stack"]}
    final = "Stack" if val_pr["Stack"] > val_pr["LightGBM"] + 0.002 else "LightGBM"
    log.append("validation PR-AUC: " + ", ".join(f"{k} {v:.3f}" for k, v in val_pr.items())
               + f"  -> final model: {final}")

    # --- calibration + threshold on val_a
    calib = Platt().fit(prob(final, X_a), y_a)
    cal = lambda X: calib.predict(prob(final, X))
    thr = best_f1_threshold(y_a, cal(X_a))

    # --- conformal (class-conditional) on val_b
    p_b = cal(X_b)
    conf_q = {}
    for k in (0, 1):
        s = np.sort(1 - (p_b[y_b == k] if k == 1 else 1 - p_b[y_b == k]))
        n = len(s)
        conf_q[k] = float(s[min(n - 1, int(np.ceil((n + 1) * (1 - ALPHA))) - 1)])

    def conformal_sets(p):
        in1 = (1 - p) <= conf_q[1]
        in0 = (1 - (1 - p)) <= conf_q[0]
        return in0, in1

    # --- baseline on the same split
    X0_tr, X0_te = X0.iloc[tr], X0.iloc[te]
    base_xgb = baseline_model().fit(X0_tr, y_tr)
    p_base_a = base_xgb.predict_proba(X0.iloc[va_a])[:, 1]
    thr_base = best_f1_threshold(y_a, p_base_a)
    p_base = base_xgb.predict_proba(X0_te)[:, 1]

    # ================= TEST (once)
    p_raw = {n: prob(n, X_te) for n in names + ["Stack"]}
    p_final_raw, p_final = p_raw[final], cal(X_te)
    rows = {"Baseline XGBoost (one-hot hubs)": metrics(y_te, p_base, thr_base),
            **{f"{n} (raw)": metrics(y_te, p_raw[n], thr) for n in names + ["Stack"]},
            f"Final = {final} + Platt": metrics(y_te, p_final, thr)}
    cis = cluster_boot(y_te, {"base": p_base, "final": p_final}, g_te, average_precision_score)
    ci_roc = cluster_boot(y_te, {"base": p_base, "final": p_final}, g_te, roc_auc_score)
    diff = cis["final"] - cis["base"][:len(cis["final"])]
    log += ["", f"TEST ({len(y_te):,} legs)  threshold: advanced {thr:.2f}, baseline {thr_base:.2f} (both F1-optimal on val_a)",
            f"{'model':<34}{'PR-AUC':>8}{'ROC-AUC':>9}{'Brier':>8}{'ECE':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}"]
    for k, m in rows.items():
        log.append(f"{k:<34}{m['pr_auc']:>8.3f}{m['roc_auc']:>9.3f}{m['brier']:>8.3f}{m['ece']:>7.3f}"
                   f"{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}")
    ci = lambda a: f"[{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]"
    log += [f"\nPR-AUC 95% CI (trip-level bootstrap): baseline {ci(cis['base'])}, final {ci(cis['final'])}",
            f"ROC-AUC 95% CI: baseline {ci(ci_roc['base'])}, final {ci(ci_roc['final'])}",
            f"paired PR-AUC gain (final - baseline): {diff.mean():+.3f} {ci(diff)}  "
            f"-> {'significant' if np.percentile(diff, 2.5) > 0 else 'not significant'} at 95%"]

    # cold-start slice
    cold = (X_te["n_src"].to_numpy() < 5) | (X_te["n_dst"].to_numpy() < 5)
    if cold.sum() > 50 and y_te[cold].min() != y_te[cold].max():
        log.append(f"cold-start legs (a hub with <5 training legs, n={cold.sum()}): PR-AUC baseline "
                   f"{average_precision_score(y_te[cold], p_base[cold]):.3f} vs final "
                   f"{average_precision_score(y_te[cold], p_final[cold]):.3f} "
                   f"(base rate {y_te[cold].mean():.3f})")

    # conformal results on test
    in0, in1 = conformal_sets(p_final)
    cov1, cov0 = in1[y_te == 1].mean(), in0[y_te == 0].mean()
    single = in0 ^ in1
    both, none = in0 & in1, ~in0 & ~in1
    acc_single = ((in1 & ~in0) == (y_te == 1))[single].mean()
    log += ["", f"Conformal (class-conditional, target coverage {1 - ALPHA:.0%})",
            f"  coverage delayed legs {cov1:.1%}, on-time legs {cov0:.1%}",
            f"  confident (single label) {single.mean():.1%} of legs, accuracy on those {acc_single:.1%}; "
            f"uncertain (both labels) {both.mean():.1%}; empty {none.mean():.1%}"]

    # --- CQR on delay minutes (log ratio actual/osrm)
    r = lambda d: np.log(d["actual_time_min"].to_numpy() / d["osrm_time_min"].to_numpy())
    r_tr, r_b, r_te = r(d_tr), r(d_va_b), r(d_te)
    qp = dict(n_estimators=400, learning_rate=0.03, num_leaves=24, min_child_samples=30, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=1, verbose=-1)
    q = {a: LGBMRegressor(objective="quantile", alpha=a, **qp).fit(X_tr, r_tr) for a in (0.05, 0.5, 0.95)}
    lo_b, hi_b = q[0.05].predict(X_b), q[0.95].predict(X_b)
    scores = np.maximum(lo_b - r_b, r_b - hi_b)
    n = len(scores)
    qhat = float(np.sort(scores)[min(n - 1, int(np.ceil((n + 1) * (1 - ALPHA))) - 1)])
    lo, med, hi = (q[a].predict(X_te) for a in (0.05, 0.5, 0.95))
    cov_raw = ((r_te >= lo) & (r_te <= hi)).mean()
    cov_cqr = ((r_te >= lo - qhat) & (r_te <= hi + qhat)).mean()
    osrm = d_te["osrm_time_min"].to_numpy()
    mins = lambda z: osrm * (np.exp(z) - 1)
    width = (mins(hi + qhat) - mins(lo - qhat)).mean()
    mae_med = np.abs(mins(med) - (d_te["actual_time_min"].to_numpy() - osrm)).mean()
    naive = np.median(r_tr)
    mae_naive = np.abs(mins(np.full(len(osrm), naive)) - (d_te["actual_time_min"].to_numpy() - osrm)).mean()
    ratio_bins = pd.qcut(X_te["dist_km"], 4, labels=False)
    cov_by = [((r_te >= lo - qhat) & (r_te <= hi + qhat))[ratio_bins == b].mean() for b in range(4)]
    log += ["", "Delay minutes (extra time vs the routing estimate), conformalized quantile regression",
            f"  raw 5-95% quantile band covers {cov_raw:.1%}; after conformal correction {cov_cqr:.1%} (target {1 - ALPHA:.0%})",
            f"  mean interval width {width:.0f} min; median-prediction MAE {mae_med:.1f} min vs naive {mae_naive:.1f} min",
            "  coverage by distance quartile (short -> long): " + ", ".join(f"{c:.1%}" for c in cov_by)]

    # --- out-of-time check
    ts = pd.to_datetime(df["od_start_time"])
    cut = ts.quantile(0.70)
    o_tr, o_te = np.flatnonzero(ts <= cut), np.flatnonzero(ts > cut)
    o_tr = o_tr[~np.isin(groups_all.iloc[o_tr].to_numpy(), groups_all.iloc[o_te].to_numpy())]
    d_otr, d_ote = df.iloc[o_tr].reset_index(drop=True), df.iloc[o_te].reset_index(drop=True)
    cx = FeatureContext(hubs).fit(d_otr)
    Xo_tr = cx.transform(d_otr, train_oof_groups=groups_all.iloc[o_tr].to_numpy())
    Xo_te = cx.transform(d_ote)
    p_o = lgbm(lgb_params).fit(Xo_tr, d_otr["is_delayed"]).predict_proba(Xo_te)[:, 1]
    pb = baseline_model().fit(X0.iloc[o_tr], d_otr["is_delayed"]).predict_proba(X0.iloc[o_te])[:, 1]
    yo = d_ote["is_delayed"].to_numpy()
    oot = {"baseline": average_precision_score(yo, pb), "advanced": average_precision_score(yo, p_o)}
    log += ["", f"Out-of-time check: train on legs up to {cut:%d %b}, test on the later {len(o_te):,} legs "
            f"(base rate {yo.mean():.1%})",
            f"  PR-AUC baseline {oot['baseline']:.3f} vs advanced {oot['advanced']:.3f}"]

    # --- SHAP
    explainer = shap.TreeExplainer(base["LightGBM"])
    sample = X_te.sample(min(1500, len(X_te)), random_state=0)
    sv = explainer.shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)
    log += ["", "Top SHAP features (mean |SHAP|, LightGBM member): "
            + ", ".join(f"{k} {v:.3f}" for k, v in imp.head(8).items())]

    # --- figures
    figure(rows, cis, ci_roc, p_final_raw, p_final, y_te, in0, in1, imp, log_title=final)

    # --- bundle for the dashboard
    joblib.dump({"ctx": ctx, "base": base, "meta": meta, "names": names, "final": final, "calibrator": calib,
                 "threshold": thr, "conf_q": conf_q, "alpha": ALPHA, "cqr": q, "cqr_q": qhat,
                 "features": FEATURES, "lgb_params": lgb_params}, os.path.join(ART, "leg_model.pkl"), compress=3)
    summary = {"test": rows, "pr_auc_ci": {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                           for k, v in cis.items()},
               "paired_gain": [float(diff.mean()), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
               "conformal": {"coverage_delayed": float(cov1), "coverage_ontime": float(cov0),
                             "confident_share": float(single.mean()), "confident_acc": float(acc_single)},
               "cqr": {"coverage_raw": float(cov_raw), "coverage_conformal": float(cov_cqr),
                       "width_min": float(width), "mae_med": float(mae_med), "mae_naive": float(mae_naive)},
               "out_of_time": oot, "final": final, "threshold": thr,
               "shap_top": imp.head(10).to_dict()}
    json.dump(summary, open(os.path.join(ART, "advanced_metrics.json"), "w"), indent=1, default=float)
    log.append(f"\nfinished in {time.time() - t0:.0f}s")
    open(os.path.join(REP, "advanced_report.txt"), "w").write("\n".join(log) + "\n")
    print("\n".join(log))


def figure(rows, cis, ci_roc, p_raw, p_cal, y, in0, in1, imp, log_title):
    INK, AMBER, GREEN, SAND = "#1A1815", "#C4622A", "#2F6B4F", "#B8A88A"
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    names = list(rows)
    vals = [rows[n]["pr_auc"] for n in names]
    ax[0].barh([n.replace(" (raw)", "") for n in names], vals,
               color=[SAND] + [GREEN if "Final" in n else "#8FA9A3" for n in names[1:]])
    ax[0].invert_yaxis()
    ax[0].set_title("Test PR-AUC (higher is better)", fontsize=10)
    ax[0].set_xlim(0.5, max(vals) + 0.04)
    for i, v in enumerate(vals):
        ax[0].text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    for p, lab, c in ((p_raw, "raw", SAND), (p_cal, "Platt calibrated", GREEN)):
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        pts = [(p[idx == b].mean(), y[idx == b].mean()) for b in range(10) if (idx == b).sum() > 20]
        ax[1].plot(*zip(*pts), "o-", color=c, label=lab)
    ax[1].plot([0, 1], [0, 1], "--", color=INK, lw=0.8)
    ax[1].set_xlabel("predicted probability")
    ax[1].set_ylabel("observed delay rate")
    ax[1].set_title("Reliability: calibration on test", fontsize=10)
    ax[1].legend(frameon=False)
    top = imp.head(12)[::-1]
    ax[2].barh(top.index, top.values, color=AMBER)
    ax[2].set_title("SHAP: mean |impact| on delay probability", fontsize=10)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(REP, "fig6_advanced.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
