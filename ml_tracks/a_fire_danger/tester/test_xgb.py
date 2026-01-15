import os
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import torchvision
import io
import PIL.Image
import pandas as pd
from pathlib import Path
import datasets.dataset as module_data
import dataloaders.dataloader as module_dataloader
from logger import TensorboardWriter
from utils.util import extract_numpy, calculate_metrics, grouped_classification_metrics
from PIL import Image


def test_xgb(config):
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.Resampling.LANCZOS

    logger = config.get_logger('test_xgb')
    writer = TensorboardWriter(config.log_dir, logger, config['trainer']['tensorboard'])
    writer.set_step(0, mode='test')

    model_path = config["model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info(f"Loading XGBoost model from: {model_path}")
    model = joblib.load(model_path)

    dataset = config.init_obj(
        'dataset', module_data,
        dynamic_features=config['features']['dynamic'],
        static_features=config['features']['static'],
        train_val_test='test'
    )
    dataloader = config.init_obj('dataloader', module_dataloader, dataset=dataset).dataloader()

    X_test, y_test, g_coastal = extract_numpy(dataloader, coastal=True)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc, prec, rec, f1, auprc = calculate_metrics(y_test, y_pred, y_proba)
    metrics = grouped_classification_metrics(y_test, y_pred, y_proba, g_coastal, positive_group=1)

    for name in ["coastal", "inland"]:
        m = metrics[name]
        logger.info(
            f"[{name.upper()}] acc={m['acc']:.4f} prec={m['precision']:.4f} "
            f"rec={m['recall']:.4f} f1={m['f1']:.4f} auprc={m['auprc']:.4f}"
        )

    summary_path = Path(config.save_dir) / f"group_metrics_xgb.csv"
    (pd.DataFrame(metrics).T).to_csv(summary_path)
    logger.info(f"Saved grouped metrics to: {summary_path}")

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

    logger.info("XGBoost Test Results:")
    logger.info(f"accuracy     : {acc:.6f}")
    logger.info(f"precision    : {prec:.6f}")
    logger.info(f"recall       : {rec:.6f}")
    logger.info(f"f1_score     : {f1:.6f}")
    logger.info(f"auprc        : {auprc:.6f}")
