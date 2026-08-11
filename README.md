# TabPFN-MarMenor
Few-shot TabPFN framework for high-fidelity multivariate water quality prediction (Chl-a &amp; Turbidity) in the Mar Menor lagoon.

# 🌊 Mar Menor Water Quality Prediction with TabPFN

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)

&gt; **Comparative analysis of TabPFN and traditional ML models for predicting Chlorophyll (µg/L) and Turbidity (NTU) in the Mar Menor coastal lagoon (Spain).**

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
| Optimal | &lt; 0.5 | Robust ecosystem |
| Surveillance | 0.5 – 1.0 | Early warning |
| Precaution | 1.0 – 5.0 | Risk increasing |
| High Mortality Risk | &gt; 5.0 | Critical state |

### Ecological Thresholds (Turbidity)
| Category | Range (NTU) | Ecological State |
|----------|------------|------------------|
| Optimal | &lt; 1.0 | Clear water |
| Surveillance | 1.0 – 3.0 | Moderate turbidity |
| Precaution | 3.0 – 5.0 | High turbidity |
| High Mortality Risk | &gt; 5.0 | Critical turbidity |

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

&gt; **Note**: The scripts handle negative values (common in sensor data) with 5 different correction strategies.

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/mar-menor-tabpfn-analysis.git
cd mar-menor-tabpfn-analysis
