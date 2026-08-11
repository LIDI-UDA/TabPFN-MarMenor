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
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Results & Figures](#-results--figures)
- [Citation](#-citation)
- [License](#-license)

---

## 🔬 Overview

The Mar Menor is the largest saltwater lagoon in Europe, located in the Region of Murcia, Spain. This project implements:

- **Classification**: Ecological categorization of water quality states based on scientific thresholds (Pérez-Martín et al., 2023)
- **Regression**: Continuous prediction of Chlorophyll and Turbidity values
- **Model Comparison**: TabPFN vs. Random Forest, Gradient Boosting, SVM, KNN, Ridge, Linear Regression, SVR
- **Computational Metrics**: Training time, prediction time, memory usage, NLL, ECE
- **Scientific Figures**: Learning curves, temporal decomposition, feature importance, calibration curves, radar charts, residual analysis

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

Place your CSV files in the `dataset_examples/` directory (as referenced by the scripts):

| File | Target Variable | Description |
|------|----------------|-------------|
| `dataset_examples/dataset_MarMenor-chl.csv` | `Mean_Chl_ugl` | Chlorophyll concentration (µg/L) |
| `dataset_examples/dataset_MarMenor-Tub.csv` | `Mean_Turb_NTU` | Turbidity (NTU) |

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

## 📁 Project Structure

```
mar-menor-tabpfn-analysis/
│
├── 📂 dataset_examples/              # Input datasets (create this folder)
│   ├── dataset_MarMenor-chl.csv      # Chlorophyll dataset
│   └── dataset_MarMenor-Tub.csv      # Turbidity dataset
│
├── 📂 resultados_marMenor-Chl_tabPFN/   # Auto-generated: Chlorophyll results
│   ├── eda_mar_menor.pdf
│   ├── distribution_negative_values.pdf
│   ├── chlorophyll_value_analysis.pdf
│   ├── correction_comparison_strategy_*.pdf
│   ├── categorization_comparison_mar_menor.pdf
│   ├── computational_metrics_mar_menor.pdf
│   ├── metrics_summary_mar_menor.csv
│   ├── model_comparison_mar_menor.pdf
│   ├── model_comparison_classification_mar_menor.pdf
│   ├── model_comparison_regression_mar_menor.pdf
│   ├── time_accuracy_tradeoff_mar_menor.pdf
│   ├── smote_balancing_mar_menor.pdf
│   ├── improvement_comparison_classification.pdf
│   ├── improvement_comparison_regression.pdf
│   ├── figure4_calibration_curves_chl.pdf
│   ├── figure5_time_accuracy_tradeoff_chl.pdf
│   ├── figure6_radar_chart_chl.pdf
│   ├── figure4c_temporal_decomposition_chl.pdf
│   ├── figure5_feature_importance_chl.pdf
│   ├── figure5_feature_importance_tabpfn_chl.pdf
│   ├── learning_curve_*_chl.pdf
│   ├── learning_curve_regression_*_chl.pdf
│   ├── regression_scatter_*.pdf
│   ├── residuals_plot_*.pdf
│   ├── confusion_matrix_*_mar_menor.pdf
│   ├── perclass_metrics_*_mar_menor.pdf
│   ├── complete_results_mar_menor.pkl
│   ├── tabpfn_model_mar_menor.pkl
│   └── tabpfn_reg_model_mar_menor.pkl
│
├── 📂 resultados_marMenor-Turb_tabPFN/  # Auto-generated: Turbidity results
│   ├── eda_mar_menor_turb.pdf
│   ├── distribution_negative_values_turb.pdf
│   ├── turbidity_value_analysis.pdf
│   ├── correction_comparison_turb_strategy_*.pdf
│   ├── categorization_comparison_mar_menor_turb.pdf
│   ├── computational_metrics_mar_menor_turb.pdf
│   ├── metrics_summary_mar_menor_turb.csv
│   ├── model_comparison_mar_menor_turb.pdf
│   ├── model_comparison_classification_mar_menor_turb.pdf
│   ├── model_comparison_regression_mar_menor_turb.pdf
│   ├── time_accuracy_tradeoff_mar_menor_turb.pdf
│   ├── smote_balancing_mar_menor_turb.pdf
│   ├── improvement_comparison_classification_turb.pdf
│   ├── improvement_comparison_regression_turb.pdf
│   ├── figure4_calibration_curves_turb.pdf
│   ├── figure5_time_accuracy_tradeoff_turb.pdf
│   ├── figure6_radar_chart_turb.pdf
│   ├── figure4c_temporal_decomposition_turb.pdf
│   ├── figure5_feature_importance_turb.pdf
│   ├── figure5_feature_importance_tabpfn_turb.pdf
│   ├── learning_curve_*_turb.pdf
│   ├── regression_scatter_*.pdf
│   ├── residuals_plot_*.pdf
│   ├── confusion_matrix_*_mar_menor_turb.pdf
│   ├── perclass_metrics_*_mar_menor_turb.pdf
│   ├── complete_results_mar_menor_turb.pkl
│   ├── tabpfn_model_mar_menor_turb.pkl
│   └── tabpfn_reg_model_mar_menor_turb.pkl
│
├── 📄 usetFPN_v4-metrics-Chl.py      # Main script: Chlorophyll analysis
├── 📄 usetFPN_v4-metrics-Turb.py     # Main script: Turbidity analysis
├── 📄 requirements.txt               # Python dependencies
├── 📄 .gitignore                     # Git ignore rules
├── 📄 LICENSE                        # License file
└── 📄 README.md                      # This file
```

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

### Prepare folders
Before running, ensure the input folder exists and contains your datasets:
```bash
mkdir -p dataset_examples
# Place dataset_MarMenor-chl.csv and dataset_MarMenor-Tub.csv inside
```

### Run Chlorophyll Analysis
```bash
python usetFPN_v4-metrics-Chl.py
```

### Run Turbidity Analysis
```bash
python usetFPN_v4-metrics-Turb.py
```

### Interactive Mode (Negative Value Correction)
Both scripts support interactive selection of correction strategies for negative values:
1. **Replace with minimum positive value** (recommended default)
2. Log transformation `log(x + offset)`
3. Remove negative values
4. KNN imputation
5. Censored variable modeling (Tobit)

Press `Enter` to use the recommended strategy (1).

### Output
Results are automatically saved in:
- `resultados_marMenor-Chl_tabPFN/`
- `resultados_marMenor-Turb_tabPFN/`

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

## 📈 Results & Figures

### Best Models (Example Output)

| Model | Type | Accuracy | F1 (Weighted) | Train Time | Memory |
|-------|------|----------|---------------|------------|--------|
| TabPFN_Clas | Classification | ~0.98 | ~0.98 | ~45s | ~2048 MB |
| RandomForest_Optimized | Classification | ~0.95 | ~0.95 | ~12s | ~512 MB |
| TabPFN_Reg | Regression | R² ~0.92 | - | ~50s | ~2048 MB |

> Actual results depend on your specific dataset and hardware.

### Figures Generated

Each execution generates **18+ publication-ready figures** per variable:

| # | Figure | Filename Pattern | Description |
|---|--------|------------------|-------------|
| 1 | EDA | `eda_mar_menor*.pdf` | 9-panel exploratory analysis |
| 2 | Negative Values | `distribution_negative_values*.pdf` | Histogram of sensor errors |
| 3 | Correction | `correction_comparison*.pdf` | Before/after correction |
| 4 | Categories | `categorization_comparison*.pdf` | Ecological vs statistical |
| 5 | **Learning Curves** | `learning_curve_*_chl/turb.pdf` | Individual model learning curves |
| 6 | **Temporal Decomposition** | `figure4c_temporal_decomposition_*.pdf` | Seasonal & yearly patterns |
| 7 | **Feature Importance** | `figure5_feature_importance_*.pdf` | Top 15 features + cumulative |
| 8 | **Calibration** | `figure4_calibration_curves_*.pdf` | Reliability diagrams |
| 9 | **Time-Accuracy** | `figure5_time_accuracy_tradeoff_*.pdf` | Pareto frontier analysis |
| 10 | **Radar Chart** | `figure6_radar_chart_*.pdf` | Top 5 models multi-metric |
| 11 | Confusion Matrix | `confusion_matrix_*.pdf` | Per-model confusion matrices |
| 12 | Per-Class Metrics | `perclass_metrics_*.pdf` | Accuracy/Precision/Recall/F1 per class |
| 13 | Regression Scatter | `regression_scatter_*.pdf` | Predicted vs Actual |
| 14 | Residuals | `residuals_plot_*.pdf` | Residual analysis |
| 15 | SMOTE | `smote_balancing*.pdf` | Class balancing visualization |
| 16 | Comparison | `model_comparison*.pdf` | All models compared |
| 17 | Computational | `computational_metrics*.pdf` | 6-panel metrics dashboard |
| 18 | Tradeoff | `time_accuracy_tradeoff*.pdf` | Scatter plot analysis |

### Serialized Outputs
| File | Description |
|------|-------------|
| `complete_results_mar_menor*.pkl` | Complete results dictionary |
| `tabpfn_model_mar_menor*.pkl` | Saved TabPFN classifier |
| `tabpfn_reg_model_mar_menor*.pkl` | Saved TabPFN regressor |
| `metrics_summary_mar_menor*.csv` | Metrics summary table |

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
