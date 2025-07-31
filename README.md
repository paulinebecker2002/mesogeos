# Mesogeos Wildfire Prediction

This project focuses on wildfire danger forecasting using machine learning and explainable AI (XAI) methods, based on the **Mesogeos** dataset — a publicly available datacube covering the Mediterranean region from 2006 to 2022 at 1 km × 1 km × 1 day resolution.

---

## Data Repository

The dataset originates from **Mesogeos**.  
You can download it from the following Google Drive folder:

 **[Mesogeos Data Repository](https://drive.google.com/drive/folders/1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9)**

This folder includes:
- **`mesogeos_cube.zarr/`**: The full Mesogeos datacube  
- **`ml_tracks/`**: Contains the pre‑extracted datasets from Mesogeos that serve as the foundation of this project. We build upon these datasets by implementing additional models and extending the pipeline with explainable AI (XAI) analyses.
- **`notebooks/`**: Jupyter notebook demonstrating how to access and process the Mesogeos datacube

---

## Repository Structure

```
mesogeos/
├── ml_tracks/
│   └── a_fire_danger/
│       ├── a_danger_forecasting/     # Contains classification data files (e.g., negatives.csv and positives.csv)
│       ├── configs/                  # Model-specific configs (MLP, CNN, LSTM, etc.)
│       ├── dataloaders/              # Data loading utilities
│       ├── datasets/                 # Dataset definitions
│       ├── integrated_gradients/     # IG computation & plotting
│       ├── models/                   # Model architectures & metrics
│       ├── saved/                    # Stores all model outputs and artifacts
│       │   └── ale/                  # Stores first-order and second-order ALE plots for feature effect analysis
│       │   └── ig/                   # Stores computed Integrated Gradients (CSV/NPZ) and their corresponding plots
│       │   ├── model/                # Stores trained model checkpoints (e.g., model_best.pth)
│       │   ├── log/                  # Contains log files and the used config files for each training/testing run
│       │   ├── shap_plot/            # Stores computed SHAP values (CSV/NPZ) and their corresponding plots
│       ├── shap_local/               # SHAP computation & plotting
│       ├── trainer/                  # Training scripts & utilities
│       ├── tester/                   # Evaluation and testing scripts
│       └── utils/                    # Helper utilities
│       └── train.py                  # Entry point for training
│       └── test.py                   # Entry point for testing
|
├── notebooks/                        # Jupyter notebooks for exploring the Mesogeos datacube and analyzing raw variables
├── outputs/                          # Analysis outputs and notebooks for evaluating ML and XAI results and generating plots
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## Setup Environment

Follow these steps to set up your Python environment:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/paulinebecker2002/mesogeos.git
   cd mesogeos
   ```

2. **Create and activate a Python environment (e.g., venv):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Training a Model

To train a model (e.g., MLP), run:
```bash
python train.py --config configs/config_<model_name>/config_train.py
```

This will train the model and save the best checkpoint under saved/models as:
```
model_best.pth
```

---

## 🧪 Testing / Evaluation

1. Open the corresponding `config_test.py` file and update the `model_path` entry to point to the `model_best.pth` checkpoint.
2. Run:
   ```bash
   python test.py --config configs/config_<model_name>/config_test.py
   ```

---

## 🔍 Explainable AI (XAI)

We use **SHAP (SHapley Additive Explanations)** and **Integrated Gradients (IG)** to interpret the predictions of trained models.

### Compute SHAP:
```bash
python shap_local/compute_shap.py --config configs/config_<model_name>/config_train.py
```
### Compute IG:
python integrated_gradients/compute_ig.py --config configs/config_<model_name>/config_train.py

### Important config keys:
- **`checkpoint_path`**: Path to the trained model checkpoint (e.g., `model_best.pth`)  
- **`shap_path`**: Directory where SHAP and IG explanation outputs will be saved  

These scripts will generate feature attribution values for interpretability and store them for visualization.

---

## Visualization and Interpretation with Jupyter Notebooks

The `outputs/` folder contains Jupyter notebooks specifically designed to **analyze, interpret, and visualize the results of trained machine learning (ML) models** and their explainable AI (XAI) outputs. These notebooks collectively provide a comprehensive framework to interpret ML models applied to wildfire forecasting, evaluate their performance, and connect their predictions to physically meaningful insights.

### Included Notebooks
- **`big_fires.ipynb`**: Analysis of large fire events and their associated predictors.  
- **`compare_models.ipynb`**: Comparison of ML model performance (metrics, curves, and results).  
- **`confusion_matrix_analysis.ipynb`**: In-depth analysis of false positives and false negatives.  
- **`csv_timeseries_before_fire_plot.ipynb`**: Visualization of time-series data leading up to fire events.  
- **`evaluate_cv_results.ipynb`**: Cross-validation evaluation across different models and configurations.  
- **`F1_Score_Comparison.ipynb`**: Direct comparison of F1 scores across multiple trained models.  
- **`physical_interpretability.ipynb`**: Relating SHAP/IG attributions to environmental and physical drivers.  
- **`RF_FeatureImportance_Top24.ipynb`**: Feature importance analysis for Random Forest baselines.  
- **`shap_analysis.ipynb`**: SHAP value computation and visualization for feature impact assessment.  
- **`shap_clustering.ipynb`**: Clustering of SHAP patterns to identify groups of similar model explanations.  
- **`shap_ig_plots.ipynb`**: Combined visualization of SHAP and Integrated Gradients explanations.  
- **`softmax_outputs_plots.ipynb`**: Plotting softmax output distributions and model confidence levels.  
- **`Timelag_Analysis.ipynb`**: Investigation of time lags and their influence on model predictions.  



