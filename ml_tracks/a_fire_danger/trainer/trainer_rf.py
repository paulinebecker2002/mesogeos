from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from logger import TensorboardWriter
from utils.util import extract_numpy, calculate_metrics, get_feature_names

def train_rf(config, dataloader_train, dataloader_val):

    #Tensorboard
    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    writer = TensorboardWriter(config.log_dir, logger, config['trainer']['tensorboard'])
    writer.set_step(0, mode='train')

    model_id = config.log_dir.name

    X_train, y_train = extract_numpy(dataloader_train)
    X_val, y_val = extract_numpy(dataloader_val)

    rf = RandomForestClassifier(
        n_estimators=config['model_args']['n_estimators'],
        max_depth=config['model_args']['max_depth'],
        min_samples_split=config['model_args']['min_samples_split'],
        min_samples_leaf=config['model_args']['min_samples_leaf'],
        max_features=config['model_args']['max_features'],
        class_weight=config['model_args']['class_weight'],
        random_state=config['seed']
    )

    rf.fit(X_train, y_train)

    # Evaluate on training set
    y_train_pred = rf.predict(X_train)
    y_train_proba = rf.predict_proba(X_train)[:, 1]

    acc_tr, prec_tr, rec_tr, f1_tr, auprc_tr = calculate_metrics(y_train, y_train_pred, y_train_proba)

    writer.add_scalar("metrics_train/accuracy", acc_tr)
    writer.add_scalar("metrics_train/precision", prec_tr)
    writer.add_scalar("metrics_train/recall", rec_tr)
    writer.add_scalar("metrics_train/f1_score", f1_tr)
    writer.add_scalar("metrics_train/auprc", auprc_tr)

    logger.info("Random Forest Training Evaluation:")
    logger.info(f"train_accuracy  : {acc_tr:.6f}")
    logger.info(f"train_precision : {prec_tr:.6f}")
    logger.info(f"train_recall    : {rec_tr:.6f}")
    logger.info(f"train_f1_score  : {f1_tr:.6f}")
    logger.info(f"train_auprc     : {auprc_tr:.6f}")

    # Evaluate on validation set
    writer.set_step(0, mode='valid')
    y_pred = rf.predict(X_val)
    y_proba = rf.predict_proba(X_val)[:, 1]

    acc, prec, rec, f1, auprc = calculate_metrics(y_val, y_pred, y_proba)

    lag = config["dataset"]["args"]["lag"]
    dyn_feats = config["features"]["dynamic"]
    stat_feats = config["features"]["static"]
    feature_names = get_feature_names(config)

    top_n = min(720, len(feature_names))
    logger.info("Top %d wichtigste Features:", top_n)
    logger.info("Sorted Feature Importances:")
    importances, sorted_idx = calculate_feature_importances(rf, feature_names, logger, writer, top_n=top_n)
    plot_feature_importances(config, model_id, importances, feature_names, top_k=20)


    writer.add_scalar("metrics/accuracy", acc)
    writer.add_scalar("metrics/precision", prec)
    writer.add_scalar("metrics/recall", rec)
    writer.add_scalar("metrics/f1_score", f1)
    writer.add_scalar("metrics/auprc", auprc)
    writer.add_scalar("rf/num_trees", len(rf.estimators_))
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)
    cm_path = os.path.join(config.log_dir, f"confusion_matrix_{model_id}.png")
    fig.savefig(cm_path)
    plt.close(fig)
    logger.info(f"Confusion matrix saved to: {os.path.abspath(cm_path)}")

    report = classification_report(y_val, y_pred, digits=4)
    writer.add_text("classification_report", f"```\n{report}\n```")

    actual_n_estimators = len(rf.estimators_)
    expected_n_estimators = rf.n_estimators

    if actual_n_estimators != expected_n_estimators:
        logger.warning(f"Only {actual_n_estimators} of {expected_n_estimators} trees were trained!")
    else:
        logger.info(f"All {expected_n_estimators} trees were successfully trained.")

    writer.add_scalar("rf/n_estimators_expected", expected_n_estimators)
    writer.add_scalar("rf/n_estimators_actual", actual_n_estimators)

    #Logger
    logger.info("Hyperparameters:")
    logger.info(f"n_estimators : {config['model_args']['n_estimators']}")
    logger.info(f"max_depth    : {config['model_args']['max_depth']}")
    logger.info(f"min_samples_split : {config['model_args']['min_samples_split']}")
    logger.info(f"min_samples_leaf : {config['model_args']['min_samples_leaf']}")
    logger.info(f"max_features : {config['model_args']['max_features']}")
    logger.info(f"class_weight : {config['model_args']['class_weight']}")
    logger.info(f"random_state : {config['seed']}")
    logger.info("Random Forest Evaluation:")
    logger.info(f"accuracy     : {acc:.6f}")
    logger.info(f"precision    : {prec:.6f}")
    logger.info(f"recall       : {rec:.6f}")
    logger.info(f"f1_score     : {f1:.6f}")
    logger.info(f"auprc        : {auprc:.6f}")
    logger.info(f"Model ID used for saving: {model_id}")


    model_path = os.path.join(config.log_dir, f"random_forest_model_{model_id}.pkl")
    joblib.dump(rf, model_path)
    logger.info(f"Random Forest model saved to: {os.path.abspath(model_path)}")

    return rf

def optuna_rf(trial, config, dataloader_train, dataloader_val):
    config['model_args']['n_estimators'] = trial.suggest_int('n_estimators', 200, 1000)
    config['model_args']['max_depth'] = trial.suggest_categorical('max_depth', [None] + list(range(1, 51)))
    config['model_args']['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 10)
    config['model_args']['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)
    config['model_args']['max_features'] = trial.suggest_categorical('max_features', ['auto', 'sqrt'])
    config['model_args']['class_weight'] = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])

    rf = train_rf(config, dataloader_train, dataloader_val)

    # Evaluate (F1-score or AUPRC as Optuna objective)
    X_val, y_val = extract_numpy(dataloader_val)
    y_pred = rf.predict(X_val)

    f1 = f1_score(y_val, y_pred)
    return f1

def calculate_feature_importances(rf, feature_names, logger, writer, top_n=20):
    importances = rf.feature_importances_
    assert len(feature_names) == len(importances), "Mismatch in feature name and importance lengths!"
    sorted_idx = np.argsort(importances)[::-1]
    for rank in range(top_n):
        fname = feature_names[sorted_idx[rank]]
        imp = importances[sorted_idx[rank]]
        logger.info(f"{fname:<24} → {imp:.4f}")
        writer.add_scalar(f"feature_importance/{fname}", imp)

    return importances, sorted_idx
def plot_feature_importances(config, model_id, importances, feature_names, top_k=20):
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

    # Speicherpfad für Feature Importance Plot
    fi_path = os.path.join(config.log_dir, f"feature_importance_top{top_k}_{model_id}.png")
    fig.savefig(fi_path)
    plt.close(fig)

    config.get_logger('trainer', config['trainer']['verbosity']).info(
        f"Feature importance plot saved to: {os.path.abspath(fi_path)}"
    )