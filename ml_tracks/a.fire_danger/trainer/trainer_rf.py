from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import torchvision
import os
import joblib
import io
import matplotlib.pyplot as plt
import PIL.Image
from logger import TensorboardWriter
from utils.util import extract_numpy, calculate_metrics

def train_rf(config, dataloader_train, dataloader_val):

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
    y_pred = rf.predict(X_val)
    y_proba = rf.predict_proba(X_val)[:, 1]

    acc, prec, rec, f1, auprc = calculate_metrics(y_val, y_pred, y_proba)

    #Tensorboard
    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    writer = TensorboardWriter(config.log_dir, logger, config['trainer']['tensorboard'])
    writer.set_step(0, mode='valid')

    lag = config["dataset"]["args"]["lag"]
    dyn_feats = config["features"]["dynamic"]
    stat_feats = config["features"]["static"]
    feature_names = [f"{name}_lag{t}" for t in range(lag) for name in dyn_feats + stat_feats]

    top_n = min(720, len(feature_names))
    importances, sorted_idx = calculate_feature_importances(rf, feature_names, logger, writer, top_n=top_n)
    plot_feature_importances(writer, 20, importances, feature_names)


    writer.add_scalar("metrics/accuracy", acc)
    writer.add_scalar("metrics/precision", prec)
    writer.add_scalar("metrics/recall", rec)
    writer.add_scalar("metrics/f1_score", f1)
    writer.add_scalar("metrics/auprc", auprc)

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

    report = classification_report(y_val, y_pred, digits=4)
    writer.add_text("classification_report", f"```\n{report}\n```")

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
    logger.info("Top %d wichtigste Features:", top_n)
    logger.info("Sorted Feature Importances:")


    model_path = os.path.join(config.log_dir, f"random_forest_model_{model_id}.pkl")
    joblib.dump(rf, model_path)
    logger.info(f"Random Forest model saved to: {os.path.abspath(model_path)}")

    return rf

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

    return image