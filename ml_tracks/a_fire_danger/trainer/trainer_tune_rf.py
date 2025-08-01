from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint
import numpy as np
import torch
import torchvision
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import PIL.Image
from logger import TensorboardWriter
from utils.util import extract_numpy, calculate_metrics

def train_rf(config, dataloader_train, dataloader_val):

    X_train, y_train = extract_numpy(dataloader_train)
    X_val, y_val = extract_numpy(dataloader_val)

    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    writer = TensorboardWriter(config.log_dir, logger, config['trainer']['tensorboard'])
    writer.set_step(0, mode='valid')

    lag = config["dataset"]["args"]["lag"]
    dyn_feats = config["features"]["dynamic"]
    stat_feats = config["features"]["static"]
    feature_names = [f"{name}_lag{t}" for t in range(lag) for name in dyn_feats + stat_feats]

    if config["finetune"]["sklearn_tune"] == "RandomizedSearch":
        param_dist = {
            "n_estimators": randint(100, 800),
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "class_weight": ["balanced"]
        }

        base_rf = RandomForestClassifier(random_state=config["seed"])
        search = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_dist,
            n_iter=30,
            scoring="f1",
            cv=3,
            verbose=1,
            random_state=config["seed"],
            n_jobs=-1
        )
        logger.info("Starting RandomizedSearchCV to tune hyperparameters...")

    elif config["finetune"]["sklearn_tune"] == "GridSearch":
        param_grid = {
            "n_estimators": [626, 628, 630, 632],
            "max_depth": [None],
            "min_samples_split": [4, 5, 6],
            "min_samples_leaf": [3],
            "max_features": ["sqrt"],
            "class_weight": ["balanced"]
        }

        base_rf = RandomForestClassifier(random_state=config["seed"])
        search = GridSearchCV(
            estimator=base_rf,
            param_grid=param_grid,
            scoring="f1",
            cv=3,
            verbose=1,
            n_jobs=-1
        )
        logger.info("Starting GridSearchCV to tune hyperparameters...")


    search.fit(X_train, y_train)
    logger.info("Best Parameters: %s", search.best_params_)

    results = search.cv_results_
    sorted_idx = np.argsort(results['mean_test_score'])[::-1]

    logger.info("All tested parameter combinations sorted by F1-score:\n")
    for rank, idx in enumerate(sorted_idx):
        mean_score = results['mean_test_score'][idx]
        std_score = results['std_test_score'][idx]
        params = results['params'][idx]
        logger.info(f"Rank {rank+1:2d}: F1 = {mean_score:.4f} (+/- {std_score:.4f}) | Params: {params}")

    rf = search.best_estimator_
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    y_proba = rf.predict_proba(X_val)[:, 1]

    # --- Metrics
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
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    confusionMatrix = PIL.Image.open(buf).convert('RGB')
    confusionMatrix = torchvision.transforms.ToTensor()(confusionMatrix)
    writer.add_image("confusion_matrix", confusionMatrix)
    plt.close()

    # --- Classification report
    report = classification_report(y_val, y_pred, digits=4)
    writer.add_text("classification_report", f"```\n{report}\n```")

    # --- Feature Importances
    importances = rf.feature_importances_
    assert len(feature_names) == len(importances), "Mismatch in feature name and importance lengths!"
    sorted_idx = np.argsort(importances)[::-1]
    top_n = min(720, len(feature_names))

    logger.info("Random Forest Evaluation:")
    logger.info(f"accuracy     : {acc:.6f}")
    logger.info(f"precision    : {prec:.6f}")
    logger.info(f"recall       : {rec:.6f}")
    logger.info(f"f1_score     : {f1:.6f}")
    logger.info(f"auprc        : {auprc:.6f}")
    logger.info("Top %d wichtigste Features:", top_n)
    #for rank in range(top_n):
     #   fname = feature_names[sorted_idx[rank]]
      #  imp = importances[sorted_idx[rank]]
       # logger.info(f"{fname:<24} → {imp:.4f}")
        #writer.add_scalar(f"feature_importance/{fname}", imp)

    #plot_feature_importances(writer, 20, importances, feature_names)

    return rf


def plot_feature_importances(writer, top_k, importances, feature_names):
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

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf).convert('RGB')
    image = torchvision.transforms.ToTensor()(image)
    writer.add_image("feature_importance/top20", image)
    plt.close()
