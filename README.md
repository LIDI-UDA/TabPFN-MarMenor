# TabPFN-MarMenor
Few-shot TabPFN framework for high-fidelity multivariate water quality prediction (Chl-a &amp; Turbidity) in the Mar Menor lagoon.
# 🌊 Mar Menor Water Quality Prediction with TabPFN

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)

> **Comparative analysis of TabPFN and traditional ML models for predicting Chlorophyll (µg/L) and Turbidity (NTU) in the Mar Menor coastal lagoon (Spain).**

This repository contains the complete pipeline for ecological water quality classification and regression using state-of-the-art transformer-based tabular models (TabPFN) alongside classical machine learning baselines.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Figures Generated](#-figures-generated)
- [Citation](#-citation)
- [License](#-license)

---

## 🔬 Overview

The Mar Menor is the largest saltwater lagoon in Europe, located in the Region of Murcia, Spain. This project implements:

- **Classification**: Ecological categorization of water quality states based on scientific thresholds (Pérez-Martín et al., 2023)
- **Regression**: Continuous prediction of Chlorophyll and Turbidity values
- **Model Comparison**: TabPFN vs. Random Forest, Gradient Boosting, SVM, KNN, Ridge, Linear Regression, SVR
- **Computational Metrics**: Training time, prediction time, memory usage, NLL, ECE
- **Scientific Figures**: Learning curves, temporal decomposition, feature importance, calibration curves, radar charts

### Ecological Thresholds (Chlorophyll)
| Category | Range (µg/L) | Ecological State |
|----------|-------------|------------------|
| Optimal | < 0.5 | Robust ecosystem |
| Surveillance | 0.5 – 1.0 | Early warning |
| Precaution | 1.0 – 5.0 | Risk increasing |
| High Mortality Risk | > 5.0 | Critical state |

### Ecological Thresholds (Turbidity)
| Category | Range (NTU) | Ecological State |
|----------|------------|------------------|
| Optimal | < 1.0 | Clear water |
| Surveillance | 1.0 – 3.0 | Moderate turbidity |
| Precaution | 3.0 – 5.0 | High turbidity |
| High Mortality Risk | > 5.0 | Critical turbidity |

---

## 📊 Dataset

Place your CSV files in the `data/raw/` directory:

| File | Target Variable | Description |
|------|----------------|-------------|
| `dataset_MarMenor-chl.csv` | `Mean_Chl_ugl` | Chlorophyll concentration (µg/L) |
| `dataset_MarMenor-Tub.csv` | `Mean_Turb_NTU` | Turbidity (NTU) |

### Expected Columns
- `TIMESTAMP`: Datetime of measurement
- `Air_Temp_HS_Avg`: Air temperature
- `RelHumidity_Avg`: Relative humidity
- `WS_ms_Avg`: Wind speed (m/s)
- `SDI_Temp_3m`: Water temperature at 3m depth
- `O2_sat2_Avg`: Oxygen saturation (%)
- `SDI_TempCorrCond_3m`: Temperature-corrected conductivity
- Target columns: `Mean_Chl_ugl` or `Mean_Turb_NTU`

> **Note**: The scripts handle negative values (common in sensor data) with 5 different correction strategies.

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/mar-menor-tabpfn-analysis.git
cd mar-menor-tabpfn-analysis
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Hugging Face Authentication (for TabPFN)
TabPFN models are downloaded from Hugging Face. You may need to login:
```python
from huggingface_hub import login
login()  # Enter your token or use: huggingface-cli login
```

---

## 🎯 Usage

### Quick Start - Chlorophyll Analysis
```bash
python src/run_chlorophyll.py
```

### Quick Start - Turbidity Analysis
```bash
python src/run_turbidity.py
```

### Interactive Mode (Negative Value Correction)
Both scripts support interactive selection of correction strategies for negative values:
1. Replace with minimum positive value (recommended)
2. Log transformation
3. Remove negative values
4. KNN imputation
5. Censored variable modeling (Tobit)

Press `Enter` to use the recommended strategy (1).

### Output
Results are saved in:
- `results/resultados_marMenor-Chl_tabPFN/`
- `results/resultados_marMenor-Turb_tabPFN/`

---

## 🏗️ Project Structure

```
mar-menor-tabpfn-analysis/
├── data/               # Input datasets (not tracked by git)
│   ├── raw/
│   │   ├── dataset_MarMenor-chl.csv
│   │   └── dataset_MarMenor-Tub.csv
│   └── processed/
├── src/                # Source code
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── metrics.py
│   ├── visualizations.py
│   ├── utils.py
│   ├── run_chlorophyll.py
│   └── run_turbidity.py
├── notebooks/          # Jupyter notebooks exploratorios
│   ├── 01_eda_chlorophyll.ipynb
│   └── 02_eda_turbidity.ipynb
├── results/            # Generated figures and models
│   ├── resultados_marMenor-Chl_tabPFN/
│   └── resultados_marMenor-Turb_tabPFN/
├── docs/               # Documentation and images
├── requirements.txt    # Python dependencies
├── .gitignore
├── LICENSE
├── CITATION.cff
└── README.md           # This file
```

---

## 🧪 Methodology

### Models Evaluated

#### Classification
| Model | Type | Key Hyperparameters |
|-------|------|---------------------|
| **TabPFN** | Transformer (Prior-Data Fitted Network) | `n_estimators=4`, `fit_mode='batched'` |
| Random Forest | Ensemble | `n_estimators=100/200`, `max_depth=15` |
| Gradient Boosting | Ensemble | `n_estimators=100/150`, `lr=0.1` |
| SVM | Kernel | `RBF kernel`, `C=1.0/10.0` |
| KNN | Instance-based | `k=5`, `weights='distance'` |

#### Regression
| Model | Type |
|-------|------|
| **TabPFN Regressor** | Transformer |
| Random Forest Regressor | Ensemble |
| Ridge Regression | Linear (L2) |
| Linear Regression | Linear |
| SVR | Kernel |

### Metrics Computed
- **Performance**: Accuracy, Precision, Recall, F1-Score, R², MSE, MAE
- **Probabilistic**: Negative Log-Likelihood (NLL), Expected Calibration Error (ECE)
- **Computational**: Training time, Prediction time, Memory usage (RAM)
- **Visual**: Confusion matrices, scatter plots, residual analysis, learning curves

### Data Split
- Training: **80%**
- Validation: **10%**
- Test: **10%**
- Stratified split for classification

---

## 📈 Results

### Best Models (Example Output)

| Model | Type | Accuracy | F1 (Weighted) | Train Time | Memory |
|-------|------|----------|---------------|------------|--------|
| TabPFN_Clas | Classification | ~0.98 | ~0.98 | ~45s | ~2048 MB |
| RandomForest_Optimized | Classification | ~0.95 | ~0.95 | ~12s | ~512 MB |
| TabPFN_Reg | Regression | R² ~0.92 | - | ~50s | ~2048 MB |

> Actual results depend on your specific dataset and hardware.

---

## 🎨 Figures Generated

Each execution generates **18+ publication-ready figures**:

| Figure | Filename | Description |
|--------|----------|-------------|
| EDA | `eda_mar_menor.pdf` / `eda_mar_menor_turb.pdf` | 9-panel exploratory analysis |
| Negative Values | `distribution_negative_values.pdf` | Histogram of sensor errors |
| Correction | `correction_comparison_strategy_*.pdf` | Before/after correction |
| Categories | `categorization_comparison_mar_menor.pdf` | Ecological vs statistical |
| **Learning Curves** | `learning_curve_*_chl/turb.pdf` | Individual model learning curves |
| **Temporal Decomposition** | `figure4c_temporal_decomposition_*.pdf` | Seasonal & yearly patterns |
| **Feature Importance** | `figure5_feature_importance_*.pdf` | Top 15 features + cumulative |
| **Calibration** | `figure4_calibration_curves_*.pdf` | Reliability diagrams |
| **Time-Accuracy** | `figure5_time_accuracy_tradeoff_*.pdf` | Pareto frontier analysis |
| **Radar Chart** | `figure6_radar_chart_*.pdf` | Top 5 models multi-metric |
| Confusion Matrix | `confusion_matrix_*.pdf` | Per-model confusion matrices |
| Per-Class Metrics | `perclass_metrics_*.pdf` | Accuracy/Precision/Recall/F1 per class |
| Regression Scatter | `regression_scatter_*.pdf` | Predicted vs Actual |
| Residuals | `residuals_plot_*.pdf` | Residual analysis |
| SMOTE | `smote_balancing_*.pdf` | Class balancing visualization |
| Comparison | `model_comparison_*.pdf` | All models compared |
| Computational | `computational_metrics_*.pdf` | 6-panel metrics dashboard |
| Tradeoff | `time_accuracy_tradeoff_*.pdf` | Scatter plot analysis |

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{mar_menor_tabpfn_2024,
  author = {Your Name},
  title = {Mar Menor Water Quality Prediction with TabPFN},
  year = {2024},
  url = {https://github.com/yourusername/mar-menor-tabpfn-analysis}
}
```

Based on:
- Pérez-Martín et al. (2023). *MMag Model of Mar Menor*. [Add full citation]

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [TabPFN](https://github.com/PriorLabs/TabPFN) team for the transformer-based tabular model
- [scikit-learn](https://scikit-learn.org/) community
- Mar Menor monitoring stations for the environmental data

---

## 📧 Contact

For questions or collaborations, please open an [Issue](https://github.com/yourusername/mar-menor-tabpfn-analysis/issues) or contact: your.email@institution.es

---
*Last updated: August 2026*
