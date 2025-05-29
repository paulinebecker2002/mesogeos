import os
import joblib
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, average_precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torchvision
import io
import PIL.Image

import datasets.dataset as module_data
import dataloaders.dataloader as module_dataloader
from logger import TensorboardWriter
from utils.util import extract_numpy, calculate_metrics


def test_rf(config):
    logger = config.get_logger('test_rf')
    writer = TensorboardWriter(config.log_dir, logger, config['trainer']['tensorboard'])
    writer.set_step(0, mode='test')

    model_path = config["model_path"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info(f"Loading Random Forest model from: {model_path}")
    rf = joblib.load(model_path)

    # Load test data
    dataset = config.init_obj('dataset', module_data,
                              dynamic_features=config['features']['dynamic'],
                              static_features=config['features']['static'],
                              train_val_test='test')
    dataloader = config.init_obj('dataloader', module_dataloader, dataset=dataset).dataloader()

    X_test, y_test = extract_numpy(dataloader)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    acc, prec, rec, f1, auprc = calculate_metrics(y_test, y_pred, y_proba)

    writer.add_scalar("test/accuracy", acc)
    writer.add_scalar("test/precision", prec)
    writer.add_scalar("test/recall", rec)
    writer.add_scalar("test/f1_score", f1)
    writer.add_scalar("test/auprc", auprc)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf).convert('RGB')
    image = torchvision.transforms.ToTensor()(image)
    writer.add_image("test/confusion_matrix", image)
    plt.close()

    report = classification_report(y_test, y_pred, digits=4)
    writer.add_text("test/classification_report", f"```{report}```")

    logger.info("Random Forest Test Results:")
    logger.info(f"accuracy     : {acc:.6f}")
    logger.info(f"precision    : {prec:.6f}")
    logger.info(f"recall       : {rec:.6f}")
    logger.info(f"f1_score     : {f1:.6f}")
    logger.info(f"auprc        : {auprc:.6f}")
