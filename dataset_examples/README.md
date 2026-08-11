# 📁 dataset_examples

This folder is intended to hold the raw environmental monitoring datasets for the Mar Menor coastal lagoon analysis.

## Expected Files

| File | Target Variable | Description |
|------|----------------|-------------|
| `dataset_MarMenor-chl.csv` | `Mean_Chl_ugl` | Chlorophyll concentration (µg/L) |
| `dataset_MarMenor-Tub.csv` | `Mean_Turb_NTU` | Turbidity (NTU) |

## Data Columns

The input CSVs should contain at minimum the following columns:
- `TIMESTAMP` — datetime of measurement
- `Air_Temp_HS_Avg` — air temperature
- `RelHumidity_Avg` — relative humidity
- `WS_ms_Avg` — wind speed (m/s)
- `SDI_Temp_3m` — water temperature at 3 m depth
- `O2_sat2_Avg` — oxygen saturation (%)
- `SDI_TempCorrCond_3m` — temperature-corrected conductivity
- `Mean_Chl_ugl` or `Mean_Turb_NTU` — target variable

> ⚠️ **Note:** These datasets are **not included** in this repository. 
> 
> **To access the data, please contact the authors.**
> 
> *For data access requests, collaboration inquiries, or further information, please reach out to the corresponding author(s).* 

---

## 📧 Contact

For data access requests, please contact the authors or open an issue in the main repository.
