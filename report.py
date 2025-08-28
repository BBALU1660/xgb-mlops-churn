import os, json, urllib.request
import numpy as np, pandas as pd
import mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, confusion_matrix, f1_score
)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def ensure_data(path: str):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading dataset to {path} ...")
    urllib.request.urlretrieve(DATA_URL, path)

def load_data(path: str):
    df = pd.read_csv(path)
    df = df.replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    y = (df["Churn"] == "Yes").astype(int).values
    X = df.drop(columns=["Churn", "customerID"])
    return X, y

def load_model(model_uri: str, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.sklearn.load_model(model_uri)

def save_roc_pr_calibration(y_true, proba):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")

    os.makedirs("docs/images", exist_ok=True)

    # ROC
    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC (AUC={auc:.3f})")
    plt.tight_layout(); plt.savefig("docs/images/roc_auc.png", dpi=160); plt.close()

    # PR
    plt.figure()
    plt.step(rec, prec, where="post")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR (AP={ap:.3f})")
    plt.tight_layout(); plt.savefig("docs/images/pr_curve.png", dpi=160); plt.close()

    # Calibration
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration curve")
    plt.tight_layout(); plt.savefig("docs/images/calibration_curve.png", dpi=160); plt.close()

    return float(auc), float(ap)

def save_confusion(y_true, proba):
    thresholds = np.linspace(0, 1, 101)
    f1s = []
    for t in thresholds:
        f1s.append(f1_score(y_true, (proba >= t).astype(int)))
    best_idx = int(np.nanargmax(f1s))
    best_t = float(thresholds[best_idx])
    cm = confusion_matrix(y_true, (proba >= best_t).astype(int))

    plt.figure()
    fig, ax = plt.subplots()
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No","Yes"]); ax.set_yticklabels(["No","Yes"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix @ t={best_t:.2f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig("docs/images/confusion_matrix.png", dpi=160); plt.close()

    return best_t, cm.tolist()

def save_feature_importance(model, X_sample, y_sample):
    # Try mapping XGBoost booster importances to transformed feature names
    try:
        pre = model.named_steps.get("pre")
        names = pre.get_feature_names_out() if hasattr(pre, "get_feature_names_out") else None
        booster = model.named_steps["clf"].get_booster()
        scores = booster.get_score(importance_type="gain")  # {'f0':score, ...}
        items = []
        for k, v in scores.items():
            try:
                idx = int(k[1:])
                feat = names[idx] if (names is not None and idx < len(names)) else k
            except Exception:
                feat = k
            # collapse to original column name (optional)
            if "__" in feat:  # e.g., "cat__Contract_Month-to-month"
                feat = feat.split("__", 1)[1]
            items.append((feat, float(v)))

        # Aggregate by base feature (group one-hots)
        import collections
        agg = collections.defaultdict(float)
        for n, val in items:
            base = n.split("_", 1)[0]  # rough collapse; good enough for overview
            agg[base] += val

        top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:20]
        labels = [t[0] for t in top]; vals = [t[1] for t in top]

        plt.figure(figsize=(8,6))
        y_pos = np.arange(len(labels))
        plt.barh(y_pos, vals)
        plt.yticks(y_pos, labels)
        plt.gca().invert_yaxis()
        plt.xlabel("Gain (aggregated)"); plt.title("Top feature importance (XGBoost gain)")
        plt.tight_layout(); plt.savefig("docs/images/feature_importance.png", dpi=160); plt.close()
        return True
    except Exception as e:
        # Fallback: permutation importance on a small sample
        try:
            from sklearn.inspection import permutation_importance
            res = permutation_importance(model, X_sample, y_sample, n_repeats=5,
                                         scoring="roc_auc", random_state=42, n_jobs=1)
            imp = res.importances_mean
            idxs = np.argsort(imp)[::-1][:20]
            labels = X_sample.columns[idxs]; vals = imp[idxs]
            plt.figure(figsize=(8,6))
            y_pos = np.arange(len(labels))
            plt.barh(y_pos, vals)
            plt.yticks(y_pos, labels)
            plt.gca().invert_yaxis()
            plt.xlabel("Permutation importance"); plt.title("Top feature importance (permutation)")
            plt.tight_layout(); plt.savefig("docs/images/feature_importance.png", dpi=160); plt.close()
            return True
        except Exception:
            return False

def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    model_uri = os.getenv("MODEL_URI", "models:/xgb_churn/Production")
    data_path = os.getenv("DATA_PATH", "data/raw/Telco-Customer-Churn.csv")

    ensure_data(data_path)
    X, y = load_data(data_path)
    model = load_model(model_uri, tracking_uri)

    # Split once for fair eval
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Predict proba
    proba = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else model.predict(Xte)

    auc, ap = save_roc_pr_calibration(yte, proba)
    best_t, cm = save_confusion(yte, proba)
    # small sample for permutation fallback
    _ = save_feature_importance(model, Xtr.iloc[:min(len(Xtr), 1000)], ytr[:min(len(ytr), 1000)])

    # Write summary
    os.makedirs("docs", exist_ok=True)
    summary = {
        "auc": round(auc, 6),
        "average_precision": round(ap, 6),
        "best_threshold": round(float(best_t), 4),
        "confusion_matrix": cm
    }
    with open("docs/perf_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open("docs/perf_summary.md", "w") as f:
        f.write(f"**ROC-AUC:** {auc:.3f}  •  **PR-AUC:** {ap:.3f}  •  **Best F1 Threshold:** {best_t:.2f}\n")

    print("Wrote plots to docs/images and metrics to docs/perf_summary.json")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
