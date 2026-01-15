from __future__ import annotations

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from xgboost import XGBClassifier
from logger import TensorboardWriter
from utils.util import extract_numpy, calculate_metrics, get_feature_names


def _compute_scale_pos_weight(y: np.ndarray) -> float:
    """n_negative / n_positive (common XGBoost heuristic for imbalanced binary classification)."""
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return float(n_neg) / float(n_pos)


def train_xgboost(config, dataloader_train, dataloader_val):
    """
    XGBoost baseline training (sklearn API) with:
      - same dataloader -> numpy extraction as RF
      - Tensorboard logging (train + val metrics)
      - confusion matrix + classification report
      - feature importance plot (top-k) saved to log_dir
      - model saved via joblib into log_dir
    """

    # Tensorboard + logger (same style as RF)
    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    writer = TensorboardWriter(config.log_dir, logger, config['trainer']['tensorboard'])

    model_id = config.log_dir.name

    # Data
    X_train, y_train = extract_numpy(dataloader_train)
    X_val, y_val = extract_numpy(dataloader_val)

    # Class imbalance handling (optional)
    args = config['model_args']

    scale_pos_weight = args.get('scale_pos_weight', None)
    if scale_pos_weight in (None, "auto", "AUTO"):
        scale_pos_weight = _compute_scale_pos_weight(y_train)
    # XGBoost params from config (with sane defaults)
    # (RF-style: config['model_args'] exists in your configs)

    early_stopping_rounds = args.get('early_stopping_rounds', 50)
    use_early_stopping = early_stopping_rounds is not None and int(early_stopping_rounds) > 0

    xgb = XGBClassifier(
        # core
        n_estimators=args.get('n_estimators', 800),
        max_depth=args.get('max_depth', 6),
        learning_rate=args.get('learning_rate', 0.05),
        subsample=args.get('subsample', 0.8),
        colsample_bytree=args.get('colsample_bytree', 0.8),
        min_child_weight=args.get('min_child_weight', 1.0),
        gamma=args.get('gamma', 0.0),
        reg_alpha=args.get('reg_alpha', 0.0),
        reg_lambda=args.get('reg_lambda', 1.0),

        # imbalanced
        scale_pos_weight=scale_pos_weight,

        # boilerplate
        objective='binary:logistic',
        eval_metric=args.get('eval_metric', 'logloss'),

        early_stopping_rounds=int(early_stopping_rounds) if use_early_stopping else None,

        random_state=config['seed'],
        n_jobs=args.get('n_jobs', -1),
        tree_method=args.get('tree_method', 'hist'),
    )

    # --- Train (with optional early stopping)
    writer.set_step(0, mode='train')

    early_stopping_rounds = args.get('early_stopping_rounds', 50)
    use_early_stopping = early_stopping_rounds is not None and int(early_stopping_rounds) > 0

    fit_kwargs = {}
    if use_early_stopping:
        fit_kwargs.update(
            dict(
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        )

    xgb.fit(X_train, y_train, **fit_kwargs)

    best_iter = getattr(xgb, "best_iteration", None)
    pred_kwargs = {"iteration_range": (0, best_iter + 1)} if best_iter is not None else {}

    # --- Train metrics
    #y_train_pred = xgb.predict(X_train)
    #y_train_proba = xgb.predict_proba(X_train)[:, 1]

    #y_pred = xgb.predict(X_val)
    #y_proba = xgb.predict_proba(X_val)[:, 1]

    y_train_pred = xgb.predict(X_train, **pred_kwargs)
    y_train_proba = xgb.predict_proba(X_train, **pred_kwargs)[:, 1]

    y_pred = xgb.predict(X_val, **pred_kwargs)
    y_proba = xgb.predict_proba(X_val, **pred_kwargs)[:, 1]

    acc_tr, prec_tr, rec_tr, f1_tr, auprc_tr = calculate_metrics(y_train, y_train_pred, y_train_proba)

    writer.add_scalar("metrics_train/accuracy", acc_tr)
    writer.add_scalar("metrics_train/precision", prec_tr)
    writer.add_scalar("metrics_train/recall", rec_tr)
    writer.add_scalar("metrics_train/f1_score", f1_tr)
    writer.add_scalar("metrics_train/auprc", auprc_tr)

    logger.info("XGBoost Training Evaluation:")
    logger.info(f"train_accuracy  : {acc_tr:.6f}")
    logger.info(f"train_precision : {prec_tr:.6f}")
    logger.info(f"train_recall    : {rec_tr:.6f}")
    logger.info(f"train_f1_score  : {f1_tr:.6f}")
    logger.info(f"train_auprc     : {auprc_tr:.6f}")

    # --- Validation metrics
    writer.set_step(0, mode='valid')

    acc, prec, rec, f1, auprc = calculate_metrics(y_val, y_pred, y_proba)

    writer.add_scalar("metrics/accuracy", acc)
    writer.add_scalar("metrics/precision", prec)
    writer.add_scalar("metrics/recall", rec)
    writer.add_scalar("metrics/f1_score", f1)
    writer.add_scalar("metrics/auprc", auprc)

    # --- Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)
    cm_path = os.path.join(config.log_dir, f"confusion_matrix_{model_id}.png")
    fig.savefig(cm_path)
    plt.close(fig)
    logger.info(f"Confusion matrix saved to: {os.path.abspath(cm_path)}")

    # --- Classification report
    report = classification_report(y_val, y_pred, digits=4)
    writer.add_text("classification_report", f"```\n{report}\n```")

    # --- Feature importances (top-k plot)
    try:
        feature_names = get_feature_names(config)
        importances = np.asarray(getattr(xgb, "feature_importances_", []), dtype=float)

        if importances.size == len(feature_names) and importances.size > 0:
            # log importances like RF
            sorted_idx = np.argsort(importances)[::-1]
            top_n = min(720, len(feature_names))

            logger.info("Top %d wichtigste Features:", top_n)
            for rank in range(top_n):
                fname = feature_names[sorted_idx[rank]]
                imp = importances[sorted_idx[rank]]
                logger.info(f"{fname:<24} → {imp:.4f}")
                writer.add_scalar(f"feature_importance/{fname}", float(imp))

            _plot_feature_importances(config, model_id, importances, feature_names, top_k=20)
        else:
            logger.warning(
                "Skipping feature-importance plot: feature_importances_ not available or length mismatch "
                f"(got {importances.size}, expected {len(feature_names)})."
            )
    except Exception as e:
        logger.warning(f"Feature importance extraction failed: {e}")

    # --- Log hyperparameters
    logger.info("Hyperparameters:")
    for k in [
        "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
        "min_child_weight", "gamma", "reg_alpha", "reg_lambda", "tree_method", "eval_metric",
        "early_stopping_rounds", "n_jobs"
    ]:
        if k in args:
            logger.info(f"{k:<18}: {args.get(k)}")
    logger.info(f"scale_pos_weight   : {scale_pos_weight}")
    logger.info(f"random_state       : {config['seed']}")
    logger.info("XGBoost Evaluation:")
    logger.info(f"accuracy     : {acc:.6f}")
    logger.info(f"precision    : {prec:.6f}")
    logger.info(f"recall       : {rec:.6f}")
    logger.info(f"f1_score     : {f1:.6f}")
    logger.info(f"auprc        : {auprc:.6f}")
    logger.info(f"Model ID used for saving: {model_id}")

    # --- Save model
    model_path = os.path.join(config.log_dir, f"xgboost_model_{model_id}.pkl")
    joblib.dump(xgb, model_path)
    logger.info(f"XGBoost model saved to: {os.path.abspath(model_path)}")

    return xgb


def _plot_feature_importances(config, model_id, importances, feature_names, top_k=20):
    top_idx = np.argsort(importances)[::-1][:top_k]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = [importances[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_k), top_vals[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importances")
    plt.tight_layout()

    fi_path = os.path.join(config.log_dir, f"feature_importance_top{top_k}_{model_id}.png")
    fig.savefig(fi_path)
    plt.close(fig)

    config.get_logger('trainer', config['trainer']['verbosity']).info(
        f"Feature importance plot saved to: {os.path.abspath(fi_path)}"
    )
