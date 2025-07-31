# Mesogeos Wildfire Prediction

This project focuses on wildfire danger forecasting using machine learning and explainable AI (XAI) methods, based on the **Mesogeos** dataset — a publicly available datacube covering the Mediterranean region from 2006 to 2022 at 1 km × 1 km × 1 day resolution.

---

## Data Repository

The dataset originates from **Mesogeos**.  
You can download it from the following Google Drive folder:

 **[Mesogeos Data Repository](https://drive.google.com/drive/folders/1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9)**

This folder includes:
- **`mesogeos_cube.zarr/`**: The full Mesogeos datacube  
- **`ml_tracks/`**: Pre‑extracted datasets for machine learning tracks  
- **`notebooks/`**: Jupyter notebooks demonstrating how to access and process the Mesogeos cubes  

---

## Repository Structure

```
mesogeos/
├── ml_tracks/
│   └── a_fire_danger/
│       ├── configs/                  # Model-specific configs (MLP, CNN, LSTM, etc.)
│       ├── dataloaders/              # Data loading utilities
│       ├── datasets/                 # Dataset definitions
│       ├── integrated_gradients/     # IG computation & plotting
│       ├── models/                   # Model architectures & metrics
│       ├── shap_local/               # SHAP computation & plotting
│       ├── trainer/                  # Training scripts & utilities
│       ├── tester/                   # Evaluation and testing scripts
│       └── utils/                    # Helper utilities
│
├── notebooks/                        # Jupyter notebooks for exploration & visualization
├── outputs/                          # Analysis outputs & comparison notebooks
├── requirements.txt                  # Python dependencies
├── train.py                          # Entry point for training
├── test.py                           # Entry point for testing
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

## Visualization with Notebooks

The `notebooks/` folder contains Jupyter notebooks designed for:
- Exploring the Mesogeos datacube  
- Visualizing input features and data distributions  
- Plotting model predictions and evaluation metrics  
- Displaying SHAP and Integrated Gradients explanation plots  

---
