"""Trains and compares delay classifiers, tunes XGBoost, evaluates once on the test set.

Writes reports/model_report.txt, reports/fig4-5, artifacts/final_model.pkl
and artifacts/test_metrics.json.
"""
import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve,
)
from xgboost import XGBClassifier

from problem import load_clean, DELAY_THRESHOLD
from features import get_xy, split_indices, build_preprocessor, get_feature_names

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(HERE, "reports")
ARTIFACT_DIR = os.path.join(HERE, "artifacts")

SEED = 42
INK, AMBER, GREEN, SAND = "#1A1815", "#C4622A", "#2F6B4F", "#B8A88A"
HEADER = f"{'model':<22}{'accuracy':>9}{'precision':>10}{'recall':>8}{'f1':>7}{'roc_auc':>9}{'pr_auc':>8}"


def evaluate(model, X, y, name):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }


def fmt(m):
    return (f"{m['model']:<22}{m['accuracy']:>9.3f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
            f"{m['f1']:>7.3f}{m['roc_auc']:>9.3f}{m['pr_auc']:>8.3f}")


def pipeline(clf, scale=False):
    return Pipeline([("pre", build_preprocessor(scale=scale)), ("clf", clf)])


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    lines = []

    def w(s=""):
        lines.append(str(s))
        print(s)

    df = load_clean()
    X, y, groups = get_xy(df)
    tr, va, te = split_indices(groups, seed=SEED)
    X_train, y_train, g_train = X.iloc[tr], y.iloc[tr], groups.iloc[tr]
    X_val, y_val = X.iloc[va], y.iloc[va]
    X_test, y_test = X.iloc[te], y.iloc[te]

    w("Pravah delay model (non-graph baseline)")
    w(f"target: actual_time > {DELAY_THRESHOLD} x osrm_time, split by trip")
    w(f"train {len(tr):,} | val {len(va):,} | test {len(te):,}")
    w(f"delayed rate: train {y_train.mean():.1%} | val {y_val.mean():.1%} | test {y_test.mean():.1%}")

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "Dummy (majority)": pipeline(DummyClassifier(strategy="most_frequent"), scale=True),
        "Logistic Regression": pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
            scale=True),
        "Decision Tree": pipeline(
            DecisionTreeClassifier(max_depth=8, min_samples_leaf=20,
                                   class_weight="balanced", random_state=SEED)),
        "Random Forest": pipeline(
            RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=5,
                                   class_weight="balanced_subsample", n_jobs=-1,
                                   random_state=SEED)),
        "XGBoost": pipeline(
            XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9,
                          colsample_bytree=0.9, scale_pos_weight=pos_weight,
                          eval_metric="logloss", random_state=SEED, n_jobs=-1)),
    }
    for m in models.values():
        m.fit(X_train, y_train)

    w("\nValidation")
    w(HEADER)
    val_results = [evaluate(m, X_val, y_val, name) for name, m in models.items()]
    for r in val_results:
        w(fmt(r))

    # tuning: XGBoost only, CV on train with trips kept together
    search = RandomizedSearchCV(
        pipeline(XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss",
                               random_state=SEED, n_jobs=-1)),
        {
            "clf__n_estimators": [200, 300, 400, 600, 800],
            "clf__max_depth": [3, 4, 5, 6, 8, 10],
            "clf__learning_rate": [0.02, 0.05, 0.08, 0.1, 0.15],
            "clf__subsample": [0.7, 0.8, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "clf__min_child_weight": [1, 3, 5, 8],
            "clf__reg_lambda": [0.5, 1.0, 2.0, 5.0],
        },
        n_iter=15, scoring="average_precision",
        cv=StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED),
        random_state=SEED, n_jobs=-1,
    )
    search.fit(X_train, y_train, groups=g_train)

    tuned_val = evaluate(search.best_estimator_, X_val, y_val, "XGBoost (tuned)")
    default_val = next(r for r in val_results if r["model"] == "XGBoost")
    w("\nTuning: XGBoost, 15 random candidates, 4-fold CV grouped by trip")
    w(f"best CV PR-AUC {search.best_score_:.3f}")
    for k, v in sorted(search.best_params_.items()):
        w(f"  {k.replace('clf__', ''):<18} {v}")
    w(fmt(tuned_val))

    if tuned_val["pr_auc"] > default_val["pr_auc"]:
        chosen, chosen_name = search.best_estimator_, "XGBoost (tuned)"
    else:
        chosen, chosen_name = models["XGBoost"], "XGBoost (default)"
    w(f"validation PR-AUC: default {default_val['pr_auc']:.3f}, tuned {tuned_val['pr_auc']:.3f} "
      f"-> using {chosen_name}")

    # final model: refit on train + val, score on test once
    final = clone(chosen).fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    dummy = clone(models["Dummy (majority)"]).fit(pd.concat([X_train, X_val]),
                                                    pd.concat([y_train, y_val]))
    test_final = evaluate(final, X_test, y_test, chosen_name)
    test_dummy = evaluate(dummy, X_test, y_test, "Dummy (majority)")

    w(f"\nTest (n={len(te):,}, refit on train + val)")
    w(HEADER)
    w(fmt(test_dummy))
    w(fmt(test_final))

    pred = final.predict(X_test)
    proba = final.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    w("\nConfusion matrix at 0.5 cutoff")
    w(f"{'':<16}{'pred on time':>14}{'pred delayed':>14}")
    w(f"{'actual on time':<16}{tn:>14,}{fp:>14,}")
    w(f"{'actual delayed':<16}{fn:>14,}{tp:>14,}")
    w(f"flags {tp + fp:,} legs; {tp:,} of {tp + fn:,} delayed legs caught "
      f"(recall {tp / (tp + fn):.1%}, precision {tp / (tp + fp):.1%})")

    names = get_feature_names(final.named_steps["pre"])
    imp = pd.Series(final.named_steps["clf"].feature_importances_, index=names)
    w("\nTop features (XGBoost gain importance)")
    for name, val in imp.sort_values(ascending=False).head(10).items():
        w(f"  {name:<45} {val:.3f}")

    plot_comparison(val_results + [tuned_val])
    plot_final(y_test, proba, cm)

    joblib.dump(final, os.path.join(ARTIFACT_DIR, "final_model.pkl"))
    with open(os.path.join(ARTIFACT_DIR, "test_metrics.json"), "w") as f:
        json.dump({
            "model": chosen_name,
            "test": test_final,
            "test_dummy": test_dummy,
            "cv_pr_auc": search.best_score_,
            "best_params": search.best_params_,
            "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        }, f, indent=2)

    with open(os.path.join(REPORT_DIR, "model_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_comparison(results):
    df = pd.DataFrame(results).set_index("model")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, metric in zip(axes, ["pr_auc", "roc_auc", "f1", "recall"]):
        vals = df[metric].sort_values()
        colors = [AMBER if v == vals.max() else SAND for v in vals]
        ax.barh(vals.index, vals.values, color=colors)
        ax.set_title(metric.replace("_", "-").upper() + " (validation)", fontsize=11)
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "fig4_model_comparison.png"), dpi=120)
    plt.close()


def plot_final(y_test, proba, cm):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[0].plot(fpr, tpr, color=AMBER, lw=2, label=f"AUC = {roc_auc_score(y_test, proba):.3f}")
    axes[0].plot([0, 1], [0, 1], color=SAND, ls="--", lw=1)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC (test)")
    axes[0].legend(frameon=False)

    prec, rec, _ = precision_recall_curve(y_test, proba)
    axes[1].plot(rec, prec, color=GREEN, lw=2,
                 label=f"PR-AUC = {average_precision_score(y_test, proba):.3f}")
    axes[1].axhline(y_test.mean(), color=SAND, ls="--", lw=1,
                    label=f"base rate = {y_test.mean():.3f}")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall (test)")
    axes[1].legend(frameon=False)

    axes[2].imshow(cm, cmap="copper_r")
    axes[2].set_xticks([0, 1], ["on time", "delayed"])
    axes[2].set_yticks([0, 1], ["on time", "delayed"])
    axes[2].set_xlabel("predicted")
    axes[2].set_ylabel("actual")
    axes[2].set_title("Confusion matrix (test)")
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                         fontweight="bold", color="white" if cm[i, j] > cm.max() / 2 else INK)

    for ax in axes[:2]:
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "fig5_final_evaluation.png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
