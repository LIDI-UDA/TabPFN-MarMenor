from huggingface_hub import login
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score, precision_score, recall_score, f1_score, mean_absolute_error
from sklearn.calibration import calibration_curve
import joblib
import pickle
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time
import psutil
import gc
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance

# Configuración global de estilo R
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = 'black'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 0.3
plt.rcParams['grid.alpha'] = 1.0
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

# Filter specific warnings
warnings.filterwarnings('ignore', category=UserWarning, message="X does not have valid feature names")

# 1. Load data
path_output = 'resultados_marMenor-Chl_tabPFN'
# Ensure directory exists
os.makedirs(path_output, exist_ok=True)

df = pd.read_csv('dataset_examples/dataset_MarMenor-chl.csv')
print("Dataset loaded. Dimensions:", df.shape)

# Check dataset structure
print("\nFirst rows of dataset:")
print(df.head())
print("\nAvailable columns:")
print(df.columns.tolist())

# Clean possible spaces in column names
df.columns = df.columns.str.strip()

# Check if target column exists
target_column = 'Mean_Chl_ugl'
if target_column not in df.columns:
    print(f"\nERROR: Target column '{target_column}' not found.")
    print("Available columns:", df.columns.tolist())
    # Search for possible target columns
    possible_targets = ['Mean_Chl_ugl', 'Chl', 'chlorophyll', 'Chlorophyll']
    for col in possible_targets:
        if col in df.columns:
            target_column = col
            print(f"Using '{col}' as target column.")
            break
        
# ====================================================================
# ANALYSIS AND CORRECTION OF NEGATIVE CHLOROPHYLL VALUES
# ====================================================================

print("\n" + "="*80)
print("ANALYSIS OF NEGATIVE CHLOROPHYLL VALUES")
print("="*80)

# Initial analysis of negative values
negative_values = df[target_column] < 0
num_negatives = negative_values.sum()
percent_negatives = (num_negatives / len(df)) * 100

print(f"Total rows: {len(df)}")
print(f"Negative chlorophyll values: {num_negatives} ({percent_negatives:.1f}%)")
print(f"Non-negative values: {len(df) - num_negatives} ({100 - percent_negatives:.1f}%)")

# Statistics of negative values
if num_negatives > 0:
    neg_stats = df[negative_values][target_column].describe()
    print("\nStatistics of negative values:")
    print(f"  Minimum: {neg_stats['min']:.4f}")
    print(f"  Maximum: {neg_stats['max']:.4f}")
    print(f"  Mean: {neg_stats['mean']:.4f}")
    print(f"  Median: {neg_stats['50%']:.4f}")
    
    # Distribution of negative values
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_facecolor('white')
    plt.hist(df[negative_values][target_column], bins=30, color='red', alpha=0.7, edgecolor='black', linewidth=0.8)
    plt.title(f'Distribution of Negative Chlorophyll Values ({num_negatives} samples)', fontsize=14, fontweight='bold')
    plt.xlabel('Chlorophyll Value ( µg/L)', fontsize=14, fontweight='normal')
    plt.ylabel('Frequency', fontsize=14, fontweight='normal')
    plt.grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    ax.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    plt.tight_layout()
    plt.savefig(f'{path_output}/distribution_negative_values.pdf', dpi=150, bbox_inches='tight')
    plt.close()

# Complete distribution analysis
print("\nComplete statistics of target variable:")
print(df[target_column].describe())

# Complete distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
# Before
axes[0].set_facecolor('white')
axes[0].hist(df[target_column], bins=50, color='lightblue', alpha=0.7, edgecolor='black', linewidth=0.8)
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
axes[0].set_title('Complete Chlorophyll Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Chlorophyll (µg/L)', fontsize=14, fontweight='normal')
axes[0].set_ylabel('Frequency', fontsize=14, fontweight='normal')
axes[0].grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
for spine in ['top', 'bottom', 'left', 'right']:
    axes[0].spines[spine].set_color('black')
    axes[0].spines[spine].set_linewidth(0.8)
axes[0].tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
axes[0].tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)

# After
zoom_range = (-2, 5)  # µg/L
zoom_mask = (df[target_column] >= zoom_range[0]) & (df[target_column] <= zoom_range[1])
axes[1].set_facecolor('white')
axes[1].hist(df[zoom_mask][target_column], bins=50, color='orange', alpha=0.7, edgecolor='black', linewidth=0.8)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_title(f'Zoom near zero [{zoom_range[0]}, {zoom_range[1]}] µg/L', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Chlorophyll (µg/L)', fontsize=14, fontweight='normal')
axes[1].set_ylabel('Frequency', fontsize=14, fontweight='normal')
axes[1].grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
for spine in ['top', 'bottom', 'left', 'right']:
    axes[1].spines[spine].set_color('black')
    axes[1].spines[spine].set_linewidth(0.8)
axes[1].tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
axes[1].tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)

plt.suptitle('ANALYSIS OF CHLOROPHYLL VALUES - MAR MENOR', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{path_output}/chlorophyll_value_analysis.pdf', dpi=150, bbox_inches='tight')
plt.close()

# ====================================================================
# STRATEGIES FOR HANDLING NEGATIVE VALUES
# ====================================================================

print("\n" + "="*80)
print("STRATEGIES FOR HANDLING NEGATIVE VALUES")
print("="*80)

print("Available strategies:")
print("1. Replace with minimum positive value (0.01 µg/L)")
print("2. Log transformation log(x + offset)")
print("3. Remove negative values")
print("4. KNN imputation")
print("5. Model as censored variable")

print("\nSelect a strategy (1-5):")
print("Recommendation: For 1974 negative values (23.3%), use strategy 1 or 2")
print("Press Enter to use recommended strategy (1):")

try:
    estrategia = input().strip()
    if estrategia == "":
        estrategia = "1"
    estrategia = int(estrategia)
    if estrategia not in [1, 2, 3, 4, 5]:
        print("Invalid option. Using strategy 1 by default.")
        estrategia = 1
except:
    print("Invalid input. Using strategy 1 by default.")
    estrategia = 1

# Apply selected strategy
df_original = df.copy()  # Save original data
df_corrected = df.copy()
original_target_column = target_column  # Save original name

print(f"\nApplying strategy {estrategia}...")

if estrategia == 1:  # Replace with minimum positive value
    min_positive = df[df[original_target_column] > 0][original_target_column].min()
    replacement_value = min(0.01, min_positive)  # Use 0.01 or the minimum positive, whichever is smaller
    print(f"Replacing negative values with {replacement_value:.4f} µg/L")
    df_corrected.loc[df_corrected[original_target_column] < 0, original_target_column] = replacement_value
    print(f"Corrected values: {num_negatives}")
    
    # Keep same column name
    target_column_corrected = original_target_column

elif estrategia == 2:  # Logarithmic transformation with offset
    offset = abs(df[df[original_target_column] < 0][original_target_column].min()) + 0.01
    print(f"Applying transformation log(x + {offset:.4f})")
    
    # Create new column with different name
    new_column_name = f"log_{original_target_column}"
    df_corrected[new_column_name] = np.log(df_corrected[original_target_column] + offset)
    
    # Update target variable to use new column
    target_column_corrected = new_column_name
    print("NOTE: Target variable is now in logarithmic scale")
    
    # Keep original column if needed
    df_corrected[original_target_column] = df[original_target_column]

elif estrategia == 3:  # Remove negative values
    print(f"Removing {num_negatives} rows with negative values")
    df_corrected = df_corrected[df_corrected[original_target_column] >= 0].copy()
    print(f"Dataset after removal: {len(df_corrected)} rows")
    target_column_corrected = original_target_column

elif estrategia == 4:  # KNN imputation
    print("Imputing negative values using KNN...")
    from sklearn.impute import KNNImputer
    
    # Create copy for imputation
    df_impute = df.copy()
    
    # Mark negative values as NaN
    df_impute.loc[df_impute[original_target_column] < 0, original_target_column] = np.nan
    
    # Separate features and target
    numeric_cols = df_impute.select_dtypes(include=[np.number]).columns.tolist()
    if original_target_column in numeric_cols:
        # Use KNN for imputation
        imputer = KNNImputer(n_neighbors=5)
        df_imputed_array = imputer.fit_transform(df_impute[numeric_cols])
        df_imputed = pd.DataFrame(df_imputed_array, columns=numeric_cols)
        
        # Replace target column
        df_corrected[original_target_column] = df_imputed[original_target_column]
        
        # Ensure no negative values remain
        df_corrected.loc[df_corrected[original_target_column] < 0, original_target_column] = 0.01
        print(f"Imputed values: {num_negatives}")
    
    target_column_corrected = original_target_column

elif estrategia == 5:  # Model as censored variable
    print("Preparing data for censored modeling (Tobit)...")
    df_corrected['censored'] = (df_corrected[original_target_column] < 0).astype(int)
    df_corrected.loc[df_corrected[original_target_column] < 0, original_target_column] = 0
    print(f"Censored values (Tobit): {num_negatives}")
    target_column_corrected = original_target_column

# Verify results
print(f"\nAfter applying strategy {estrategia}:")
print(f"Remaining negative values: {(df_corrected[target_column_corrected] < 0).sum()}")
print(f"Minimum value: {df_corrected[target_column_corrected].min():.4f}")
print(f"Maximum value: {df_corrected[target_column_corrected].max():.4f}")

# Comparative before/after plot
if estrategia in [1, 3, 4]:  # Only for strategies that maintain original scale
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before
    axes[0].set_facecolor('white')
    axes[0].hist(df_original[original_target_column], bins=50, color='red', alpha=0.5, edgecolor='black', linewidth=0.8, label='Original')
    axes[0].set_title('Before Correction', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Chlorophyll (µg/L)', fontsize=14, fontweight='normal')
    axes[0].set_ylabel('Frequency', fontsize=14, fontweight='normal')
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    axes[0].grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    for spine in ['top', 'bottom', 'left', 'right']:
        axes[0].spines[spine].set_color('black')
        axes[0].spines[spine].set_linewidth(0.8)
    axes[0].tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    axes[0].tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    # After
    axes[1].set_facecolor('white')
    axes[1].hist(df_corrected[original_target_column], bins=50, color='green', alpha=0.5, edgecolor='black', linewidth=0.8, label='Corrected')
    axes[1].set_title('After Correction', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Chlorophyll (µg/L)', fontsize=14, fontweight='normal')
    axes[1].set_ylabel('Frequency', fontsize=14, fontweight='normal')
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    axes[1].grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    for spine in ['top', 'bottom', 'left', 'right']:
        axes[1].spines[spine].set_color('black')
        axes[1].spines[spine].set_linewidth(0.8)
    axes[1].tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    axes[1].tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    plt.suptitle(f'Before/After Comparison - Strategy {estrategia}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{path_output}/correction_comparison_strategy_{estrategia}.pdf', dpi=150, bbox_inches='tight')
    plt.close()

# Use corrected dataset
df = df_corrected

# Update target column name if changed
target_column = target_column_corrected

# Check null values
print("\nNull values by column:")
print(df.isnull().sum())

# Remove rows with null values if any
initial_rows = len(df)
df = df.dropna()
print(f"\nRows after removing null values: {len(df)} ({initial_rows - len(df)} removed)")

# Convert TIMESTAMP to datetime
if 'TIMESTAMP' in df.columns:
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
    print(f"\nTemporal range of dataset:")
    print(f"  Start: {df['TIMESTAMP'].min()}")
    print(f"  End: {df['TIMESTAMP'].max()}")
    print(f"  Total hours: {len(df)}")
    
    # Extract temporal features
    df['hour'] = df['TIMESTAMP'].dt.hour
    df['day'] = df['TIMESTAMP'].dt.day
    df['month'] = df['TIMESTAMP'].dt.month
    df['year'] = df['TIMESTAMP'].dt.year
    df['dayofyear'] = df['TIMESTAMP'].dt.dayofyear

# Analysis of target variable
print(f"\nAnalysis of target variable '{target_column}':")
print(df[target_column].describe())

# Create categories for classification
# ====================================================================
# ECOLOGICAL CATEGORIZATION BASED ON ACTUAL SCIENCE - MAR MENOR
# ====================================================================
print("\n" + "="*80)
print("ECOLOGICAL CATEGORIZATION BASED ON SCIENCE")
print("="*80)
print("Based on: Pérez-Martín et al. (2023) - MMag Model of Mar Menor")
print("• Optimal/Robust State: < 1 µg/L")
print("• High Risk of Mass Mortality: > 5 µg/L")

# Ecological thresholds based on scientific literature of Mar Menor
umbrales_mar_menor = [0.5, 1.0, 5.0]  # Values in µg/L according to Pérez-Martín et al. (2023)
etiquetas_ecologicas = ['Optimal', 'Surveillance', 'Precaution', 'High Mortality Risk']

# Create ecological categories
df['Chl_category_ecologica'] = pd.cut(df[target_column],
                                     bins=[-np.inf] + umbrales_mar_menor + [np.inf],
                                     labels=etiquetas_ecologicas)

# Use quartiles to create balanced statistical categories
try:
    # Check if there are enough unique values
    if df[target_column].nunique() > 10:
        df['Chl_category_estadistica'] = pd.qcut(df[target_column], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    else:
        # If few unique values, use threshold-based categories
        thresholds = np.percentile(df[target_column], [25, 50, 75])
        df['Chl_category_estadistica'] = pd.cut(df[target_column], 
                                   bins=[-np.inf, thresholds[0], thresholds[1], thresholds[2], np.inf],
                                   labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    
    # COMPARATIVE ANALYSIS OF BOTH CLASSIFICATIONS
    print("\n" + "="*80)
    print("COMPARISON: STATISTICAL vs ECOLOGICAL CATEGORIES")
    print("="*80)

    print("\n1. DISTRIBUTION BY ECOLOGICAL CATEGORY (Actual Science):")
    dist_eco = df['Chl_category_ecologica'].value_counts().sort_index()
    for categoria, count in dist_eco.items():
        porcentaje = (count / len(df)) * 100
        print(f"  {categoria}: {count} samples ({porcentaje:.1f}%)")
        
        # Show value ranges for each ecological category
        categoria_min = df[df['Chl_category_ecologica'] == categoria][target_column].min()
        categoria_max = df[df['Chl_category_ecologica'] == categoria][target_column].max()
        print(f"    Range: [{categoria_min:.2f} - {categoria_max:.2f}] µg/L")

    print("\n2. DISTRIBUTION BY STATISTICAL CATEGORY (Quartiles):")
    if 'Chl_category_estadistica' in df.columns:
        dist_est = df['Chl_category_estadistica'].value_counts().sort_index()
        for categoria, count in dist_est.items():
            porcentaje = (count / len(df)) * 100
            print(f"  {categoria}: {count} samples ({porcentaje:.1f}%)")

    # VERIFICATION OF CRITICAL THRESHOLDS
    print("\n" + "="*80)
    print("VERIFICATION OF CRITICAL STATES - MAR MENOR")
    print("="*80)

    # Count samples in high risk state (> 5 µg/L)
    alto_riesgo_count = (df[target_column] > 5.0).sum()
    alto_riesgo_porcentaje = (alto_riesgo_count / len(df)) * 100
    print(f"Samples in HIGH MORTALITY RISK (> 5 µg/L): {alto_riesgo_count} ({alto_riesgo_porcentaje:.1f}%)")

    # Count samples in optimal state (< 1 µg/L)
    optimo_count = (df[target_column] < 1.0).sum()
    optimo_porcentaje = (optimo_count / len(df)) * 100
    print(f"Samples in OPTIMAL/ROBUST state (< 1 µg/L): {optimo_count} ({optimo_porcentaje:.1f}%)")

    # COMPARATIVE VISUALIZATION
    print("\nGenerating comparative visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Ecological distribution
    if 'Chl_category_ecologica' in df.columns:
        colors_eco = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']  # Green to Red (ecological)
        dist_eco = df['Chl_category_ecologica'].value_counts().sort_index()
        bars1 = axes[0].bar(dist_eco.index.astype(str), dist_eco.values, 
                           color=colors_eco, edgecolor='black', linewidth=0.8, width=0.7)
        axes[0].set_xlabel('Ecological Classification (Scientific Thresholds)', fontsize=14, fontweight='normal')
        axes[0].set_ylabel('Number of Observations', fontsize=14, fontweight='normal')
        axes[0].set_facecolor('white')
        axes[0].grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
        axes[0].set_axisbelow(True)
        for spine in ['top', 'bottom', 'left', 'right']:
            axes[0].spines[spine].set_color('black')
            axes[0].spines[spine].set_linewidth(0.8)
        axes[0].tick_params(axis='x', rotation=45, length=4, width=0.8, color='black', labelsize=10)
        axes[0].tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
        
        y_max = max(dist_eco.values)
        axes[0].set_ylim(0, y_max * 1.15)

        for bar, count in zip(bars1, dist_eco.values):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + (y_max * 0.02),
                        f'{count}', ha='center', va='bottom', fontsize=12, fontweight='normal')

    # Plot 2: Statistical distribution
    if 'Chl_category_estadistica' in df.columns:
        colors_est = ['#4DAF4A', '#377EB8', '#FF7F00', '#E41A1C']  # R ColorBrewer palette
        dist_est = df['Chl_category_estadistica'].value_counts().sort_index()
        bars2 = axes[1].bar(dist_est.index.astype(str), dist_est.values, 
                           color=colors_est, edgecolor='black', linewidth=0.8, width=0.7)
        axes[1].set_xlabel('Statistical Classification', fontsize=14, fontweight='normal')
        axes[1].set_ylabel('Number of Observations', fontsize=14, fontweight='normal')
        axes[1].set_facecolor('white')
        axes[1].grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
        axes[1].set_axisbelow(True)
        for spine in ['top', 'bottom', 'left', 'right']:
            axes[1].spines[spine].set_color('black')
            axes[1].spines[spine].set_linewidth(0.8)
        axes[1].tick_params(axis='x', rotation=45, length=4, width=0.8, color='black', labelsize=10)
        axes[1].tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
        
        y_max = max(dist_est.values)
        axes[1].set_ylim(0, y_max * 1.15)

        for bar, count in zip(bars2, dist_est.values):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + (y_max * 0.02),
                        f'{count}', ha='center', va='bottom', fontsize=12, fontweight='normal')

    plt.tight_layout()
    plt.savefig(f'{path_output}/categorization_comparison_mar_menor.pdf', dpi=150, bbox_inches='tight')
    plt.close()

    # USE ECOLOGICAL CATEGORY FOR MODELING
    print("\n" + "="*80)
    print("MODELING CONFIGURATION: Using ecological categories")
    print("="*80)
    df['Chl_category'] = df['Chl_category_ecologica']  # Use ecological for modeling

    print("\nChlorophyll category distribution:")
    print(df['Chl_category'].value_counts())

    # Check balance
    category_counts = df['Chl_category'].value_counts()
    balance_ratio = category_counts.min() / category_counts.max()
    print(f"Balance ratio (min/max): {balance_ratio:.3f}")
    
    if balance_ratio < 0.3:
        print("WARNING: Categories are highly imbalanced. Consider using more/fewer categories.")
        print("Consider adjusting thresholds or using class balancing techniques.")
except Exception as e:
    print(f"Error creating categories: {e}")
    print("Using binary classification (high/low) instead...")
    median_val = df[target_column].median()
    df['Chl_category'] = np.where(df[target_column] > median_val, 'High', 'Low')
    print("Binary category distribution:")
    print(df['Chl_category'].value_counts())

# ====================================================================
# NEW FUNCTIONS FOR FIGURES 3b, 4c, AND 5
# ====================================================================

def crear_learning_curves_individual(modelo_nombre, modelo, X_train, y_train, X_val, y_val, path_output, variable_name="Chlorophyll"):
    """
    Creates individual learning curve for a single model.
    Saves a separate figure for each model.
    
    Args:
        modelo_nombre: Name of the model
        modelo: Model object
        X_train, y_train: Training data
        X_val, y_val: Validation data
        path_output: Output directory
        variable_name: Name of target variable
    """
    print(f"\n  Generating learning curve for {modelo_nombre}...")
    
    # Determine model type
    is_classification = hasattr(modelo, 'predict_proba') or (hasattr(modelo, 'predict') and len(np.unique(y_train)) <= 10)
    
    # Training sizes (logarithmic spacing)
    train_sizes = np.linspace(0.1, 1.0, 8)  # 8 points from 10% to 100%
    train_sizes_abs = np.round(train_sizes * len(X_train)).astype(int)
    train_sizes_abs = train_sizes_abs[train_sizes_abs > 0]
    
    # Store scores
    train_scores = []
    val_scores = []
    train_std = []
    val_std = []
    
    # Calculate learning curves with multiple repetitions for stability
    n_repeats = 3
    
    for train_size in train_sizes_abs:
        train_scores_size = []
        val_scores_size = []
        
        for rep in range(n_repeats):
            # Sample training data
            indices = np.random.choice(len(X_train), min(train_size, len(X_train)), replace=False)
            X_train_sub = X_train[indices]
            y_train_sub = y_train[indices]
            
            # Train model - handle different model types
            try:
                # Check if model is TabPFN (which has different API)
                if 'TabPFN' in str(type(modelo)):
                    if hasattr(modelo, 'fit'):
                        # Create new instance for TabPFN
                        if hasattr(modelo, 'get_params'):
                            modelo_clone = modelo.__class__(**modelo.get_params())
                        else:
                            modelo_clone = modelo.__class__()
                        modelo_clone.fit(X_train_sub, y_train_sub)
                    else:
                        continue
                else:
                    # Standard sklearn models
                    modelo_clone = modelo.__class__(**modelo.get_params())
                    modelo_clone.fit(X_train_sub, y_train_sub)
                
                # Evaluate on training
                if is_classification:
                    train_pred = modelo_clone.predict(X_train_sub)
                    train_acc = accuracy_score(y_train_sub, train_pred)
                else:
                    train_pred = modelo_clone.predict(X_train_sub)
                    train_acc = r2_score(y_train_sub, train_pred)
                
                # Evaluate on validation
                if is_classification:
                    val_pred = modelo_clone.predict(X_val)
                    val_acc = accuracy_score(y_val, val_pred)
                else:
                    val_pred = modelo_clone.predict(X_val)
                    val_acc = r2_score(y_val, val_pred)
                
                train_scores_size.append(train_acc)
                val_scores_size.append(val_acc)
                
            except Exception as e:
                print(f"      Warning: Error in {modelo_nombre} with size {train_size}, rep {rep}: {e}")
                continue
        
        if train_scores_size:
            train_scores.append(np.mean(train_scores_size))
            train_std.append(np.std(train_scores_size) if len(train_scores_size) > 1 else 0)
            val_scores.append(np.mean(val_scores_size))
            val_std.append(np.std(val_scores_size) if len(val_scores_size) > 1 else 0)
        else:
            train_scores.append(0)
            train_std.append(0)
            val_scores.append(0)
            val_std.append(0)
    
    # Use actual training sizes that succeeded
    actual_sizes = train_sizes_abs[:len(train_scores)]
    
    # Create figure for this model
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('white')
    
    # Plot learning curves
    ax.plot(actual_sizes, train_scores, 'o-', color='#377EB8', 
            linewidth=2, markersize=8, label='Training Score', 
            markeredgecolor='black', markeredgewidth=0.5)
    ax.plot(actual_sizes, val_scores, 's-', color='#E41A1C', 
            linewidth=2, markersize=8, label='Validation Score',
            markeredgecolor='black', markeredgewidth=0.5)
    
    # Fill between to show variance (only if std > 0)
    if any(train_std) > 0:
        ax.fill_between(actual_sizes, 
                        np.array(train_scores) - np.array(train_std), 
                        np.array(train_scores) + np.array(train_std), 
                        alpha=0.2, color='#377EB8')
    if any(val_std) > 0:
        ax.fill_between(actual_sizes, 
                        np.array(val_scores) - np.array(val_std), 
                        np.array(val_scores) + np.array(val_std), 
                        alpha=0.2, color='#E41A1C')
    
    # Clean up model name for display
    display_name = modelo_nombre.replace('_Clas', '').replace('_Optimized', '').replace('_Reg', '')
    
    ax.set_xlabel('Training Set Size', fontsize=14, fontweight='normal')
    ax.set_ylabel('Score', fontsize=14, fontweight='normal')
    #ax.set_title(f'Learning Curve - {display_name}', fontsize=14, fontweight='bold')
    ax.grid(True, axis='both', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=True, facecolor='#f0f0f0', edgecolor='black', fontsize=12)
    
    # Add final performance annotation
    if len(train_scores) > 0 and len(val_scores) > 0:
        final_train = train_scores[-1]
        final_val = val_scores[-1]
        gap = final_train - final_val
        
        textstr = f'Final Training Score: {final_train:.4f}\nFinal Validation Score: {final_val:.4f}\nGap: {gap:.4f}'
        props = dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='black', alpha=0.9)
        ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='bottom', bbox=props)
    
    # Set y-axis limits based on data
    all_scores = train_scores + val_scores
    if all_scores and max(all_scores) > 0:
        y_min = max(0, min(all_scores) - 0.1)
        y_max = min(1.05, max(all_scores) + 0.05)
        ax.set_ylim([y_min, y_max])
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    ax.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    #plt.suptitle(f'LEARNING CURVE - {variable_name}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Create a clean filename for this model
    clean_name = modelo_nombre.replace('_Clas', '').replace('_Optimized', '').replace('_Reg', '').replace(' ', '_').lower()
    if variable_name.lower() == 'turbidity':
        filename = f'{path_output}/learning_curve_{clean_name}_turb.pdf'
    else:
        filename = f'{path_output}/learning_curve_{clean_name}_chl.pdf'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Learning curve saved as '{filename}'")
    
    return fig, train_scores, val_scores

def crear_learning_curves_all_models(modelos_dict, X_train, y_train, X_val, y_val, path_output, variable_name="Chlorophyll"):
    """
    Creates individual learning curves for each model in the dictionary.
    Saves a separate figure for each model.
    
    Args:
        modelos_dict: Dictionary of {model_name: model_object} for trained models
        X_train, y_train: Training data
        X_val, y_val: Validation data
        path_output: Output directory
        variable_name: Name of target variable
    """
    print(f"\n" + "="*80)
    print(f"GENERATING INDIVIDUAL LEARNING CURVES")
    print("="*80)
    
    if not modelos_dict:
        print("  No models provided for learning curves.")
        return {}
    
    results = {}
    
    for modelo_nombre, modelo in modelos_dict.items():
        try:
            fig, train_scores, val_scores = crear_learning_curves_individual(
                modelo_nombre, modelo, X_train, y_train, X_val, y_val, path_output, variable_name
            )
            results[modelo_nombre] = {
                'fig': fig,
                'train_scores': train_scores,
                'val_scores': val_scores
            }
        except Exception as e:
            print(f"  ✗ Error generating learning curve for {modelo_nombre}: {e}")
            continue
    
    print(f"\n✓ Generated {len(results)} individual learning curves")
    
    return results

def crear_learning_curves_individual_regression(modelo_nombre, modelo, X_train, y_train, X_val, y_val, path_output, variable_name="Chlorophyll"):
    """
    Creates individual learning curve for a single regression model.
    Saves a separate figure for each model.
    
    Args:
        modelo_nombre: Name of the model
        modelo: Model object
        X_train, y_train: Training data
        X_val, y_val: Validation data
        path_output: Output directory
        variable_name: Name of target variable
    """
    print(f"\n  Generating learning curve for regression model: {modelo_nombre}...")
    
    # Determine if classification or regression (always regression here)
    is_classification = False
    
    # Training sizes (logarithmic spacing)
    train_sizes = np.linspace(0.1, 1.0, 8)  # 8 points from 10% to 100%
    train_sizes_abs = np.round(train_sizes * len(X_train)).astype(int)
    train_sizes_abs = train_sizes_abs[train_sizes_abs > 0]
    
    # Store scores (using R² for regression)
    train_scores = []
    val_scores = []
    train_std = []
    val_std = []
    
    # Calculate learning curves with multiple repetitions for stability
    n_repeats = 3
    
    for train_size in train_sizes_abs:
        train_scores_size = []
        val_scores_size = []
        
        for rep in range(n_repeats):
            # Sample training data
            indices = np.random.choice(len(X_train), min(train_size, len(X_train)), replace=False)
            X_train_sub = X_train[indices]
            y_train_sub = y_train[indices]
            
            # Train model - handle different model types
            try:
                # Check if model is TabPFN (which has different API)
                if 'TabPFN' in str(type(modelo)):
                    if hasattr(modelo, 'fit'):
                        # Create new instance for TabPFN
                        if hasattr(modelo, 'get_params'):
                            modelo_clone = modelo.__class__(**modelo.get_params())
                        else:
                            modelo_clone = modelo.__class__()
                        modelo_clone.fit(X_train_sub, y_train_sub)
                    else:
                        continue
                else:
                    # Standard sklearn models
                    modelo_clone = modelo.__class__(**modelo.get_params())
                    modelo_clone.fit(X_train_sub, y_train_sub)
                
                # Evaluate on training (using R² for regression)
                train_pred = modelo_clone.predict(X_train_sub)
                train_r2 = r2_score(y_train_sub, train_pred)
                
                # Evaluate on validation
                val_pred = modelo_clone.predict(X_val)
                val_r2 = r2_score(y_val, val_pred)
                
                train_scores_size.append(train_r2)
                val_scores_size.append(val_r2)
                
            except Exception as e:
                print(f"      Warning: Error in {modelo_nombre} with size {train_size}, rep {rep}: {e}")
                continue
        
        if train_scores_size:
            train_scores.append(np.mean(train_scores_size))
            train_std.append(np.std(train_scores_size) if len(train_scores_size) > 1 else 0)
            val_scores.append(np.mean(val_scores_size))
            val_std.append(np.std(val_scores_size) if len(val_scores_size) > 1 else 0)
        else:
            train_scores.append(0)
            train_std.append(0)
            val_scores.append(0)
            val_std.append(0)
    
    # Use actual training sizes that succeeded
    actual_sizes = train_sizes_abs[:len(train_scores)]
    
    # Create figure for this model
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('white')
    
    # Plot learning curves
    ax.plot(actual_sizes, train_scores, 'o-', color='#377EB8', 
            linewidth=2, markersize=8, label='Training R²', 
            markeredgecolor='black', markeredgewidth=0.5)
    ax.plot(actual_sizes, val_scores, 's-', color='#E41A1C', 
            linewidth=2, markersize=8, label='Validation R²',
            markeredgecolor='black', markeredgewidth=0.5)
    
    # Fill between to show variance (only if std > 0)
    if any(train_std) > 0:
        ax.fill_between(actual_sizes, 
                        np.array(train_scores) - np.array(train_std), 
                        np.array(train_scores) + np.array(train_std), 
                        alpha=0.2, color='#377EB8')
    if any(val_std) > 0:
        ax.fill_between(actual_sizes, 
                        np.array(val_scores) - np.array(val_std), 
                        np.array(val_scores) + np.array(val_std), 
                        alpha=0.2, color='#E41A1C')
    
    # Clean up model name for display
    display_name = modelo_nombre.replace('_Reg', '').replace('_Optimized', '').replace('_Reg', '')
    
    ax.set_xlabel('Training Set Size', fontsize=14, fontweight='normal')
    ax.set_ylabel('R² Score', fontsize=14, fontweight='normal')
    #ax.set_title(f'Learning Curve - {display_name} (Regression)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='both', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=True, facecolor='#f0f0f0', edgecolor='black', fontsize=12)
    
    # Add final performance annotation
    if len(train_scores) > 0 and len(val_scores) > 0:
        final_train = train_scores[-1]
        final_val = val_scores[-1]
        gap = final_train - final_val
        
        textstr = f'Final Training R²: {final_train:.4f}\nFinal Validation R²: {final_val:.4f}\nGap: {gap:.4f}'
        props = dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='black', alpha=0.9)
        ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='bottom', bbox=props)
    
    # Set y-axis limits based on data
    all_scores = train_scores + val_scores
    if all_scores and max(all_scores) > 0:
        y_min = max(-0.2, min(all_scores) - 0.1)
        y_max = min(1.05, max(all_scores) + 0.05)
        ax.set_ylim([y_min, y_max])
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    ax.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    #plt.suptitle(f'LEARNING CURVE - REGRESSION - {variable_name}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Create a clean filename for this model
    clean_name = modelo_nombre.replace('_Reg', '').replace('_Optimized', '').replace(' ', '_').lower()
    if variable_name.lower() == 'turbidity':
        filename = f'{path_output}/learning_curve_regression_{clean_name}_turb.pdf'
    else:
        filename = f'{path_output}/learning_curve_regression_{clean_name}_chl.pdf'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Regression learning curve saved as '{filename}'")
    
    return fig, train_scores, val_scores

def crear_learning_curves_all_models_regression(modelos_dict, X_train, y_train, X_val, y_val, path_output, variable_name="Chlorophyll"):
    """
    Creates individual learning curves for each regression model in the dictionary.
    Saves a separate figure for each model.
    
    Args:
        modelos_dict: Dictionary of {model_name: model_object} for trained regression models
        X_train, y_train: Training data
        X_val, y_val: Validation data
        path_output: Output directory
        variable_name: Name of target variable
    """
    print(f"\n" + "="*80)
    print(f"GENERATING INDIVIDUAL LEARNING CURVES - REGRESSION MODELS")
    print("="*80)
    
    if not modelos_dict:
        print("  No regression models provided for learning curves.")
        return {}
    
    results = {}
    
    for modelo_nombre, modelo in modelos_dict.items():
        try:
            fig, train_scores, val_scores = crear_learning_curves_individual_regression(
                modelo_nombre, modelo, X_train, y_train, X_val, y_val, path_output, variable_name
            )
            results[modelo_nombre] = {
                'fig': fig,
                'train_scores': train_scores,
                'val_scores': val_scores
            }
        except Exception as e:
            print(f"  ✗ Error generating regression learning curve for {modelo_nombre}: {e}")
            continue
    
    print(f"\n✓ Generated {len(results)} individual regression learning curves")
    
    return results

def crear_feature_importance(modelo, X_train, y_train, feature_names, path_output, modelo_nombre, variable_name="Chlorophyll"):
    """
    Creates feature importance plot for tree-based models.
    Figure 5: Feature importance analysis.
    """
    print(f"\nGenerating feature importance for {modelo_nombre}...")
    
    # Check if we have valid feature names and data
    if len(feature_names) == 0:
        print("  No features available for importance analysis.")
        return None, None
    
    # Check if model has feature_importances_ attribute
    if hasattr(modelo, 'feature_importances_'):
        importances = modelo.feature_importances_
        method = "Gini Importance"
    elif hasattr(modelo, 'coef_'):
        # For linear models, use coefficients
        importances = np.abs(modelo.coef_)
        if len(importances.shape) > 1:
            importances = np.mean(importances, axis=0)
        method = "Coefficient Magnitude"
    else:
        # Use permutation importance
        print("  Model doesn't have built-in feature importance. Using permutation importance...")
        try:
            # Determine model type
            is_classification = hasattr(modelo, 'predict_proba') or (len(np.unique(y_train)) <= 10)
            
            # Calculate permutation importance (use a subset of data for speed)
            n_samples = min(1000, len(X_train))
            indices = np.random.choice(len(X_train), n_samples, replace=False)
            X_sample = X_train[indices]
            y_sample = y_train[indices]
            
            result = permutation_importance(modelo, X_sample, y_sample, 
                                          n_repeats=5, random_state=42, 
                                          scoring='accuracy' if is_classification else 'r2')
            importances = result.importances_mean
            method = "Permutation Importance"
        except Exception as e:
            print(f"  Could not calculate permutation importance: {e}")
            return None, None
    
    # Ensure importances is 1D
    if len(importances.shape) > 1:
        importances = np.mean(importances, axis=0)
    
    # Create DataFrame for plotting
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    # Filter out features with zero importance
    feature_importance_df = feature_importance_df[feature_importance_df['importance'] > 0]
    
    if len(feature_importance_df) == 0:
        print("  No features with positive importance found.")
        return None, None
    
    # Select top 15 features (or all if less)
    top_n = min(15, len(feature_importance_df))
    top_features = feature_importance_df.tail(top_n)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Plot 1: Horizontal bar chart (top features)
    ax1.set_facecolor('white')
    bars = ax1.barh(top_features['feature'], top_features['importance'], 
                    color='#377EB8', edgecolor='black', linewidth=0.8)
    ax1.set_xlabel('Importance', fontsize=14, fontweight='normal')
    ax1.set_title(f'Feature Importance - {modelo_nombre}', fontsize=14, fontweight='bold')
    ax1.grid(True, axis='x', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    ax1.set_axisbelow(True)
    
    # Add value labels
    max_imp = top_features['importance'].max()
    for bar, val in zip(bars, top_features['importance'].values):
        width = bar.get_width()
        ax1.text(width + (max_imp * 0.01), 
                bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', ha='left', va='center', fontsize=9)
    
    # Plot 2: Cumulative importance
    ax2.set_facecolor('white')
    sorted_importance = feature_importance_df.sort_values('importance', ascending=False)
    cumulative = np.cumsum(sorted_importance['importance'].values) / np.sum(sorted_importance['importance'].values)
    
    ax2.plot(range(1, len(cumulative) + 1), cumulative, 'o-', color='#E41A1C', 
            linewidth=2, markersize=6, markeredgecolor='black', markeredgewidth=0.5)
    ax2.axhline(y=0.8, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='80% threshold')
    ax2.axhline(y=0.9, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='90% threshold')
    ax2.fill_between(range(1, len(cumulative) + 1), cumulative, alpha=0.2, color='#E41A1C')
    
    # Find number of features for 80% and 90% cumulative importance
    n_80 = np.where(cumulative >= 0.8)[0]
    n_90 = np.where(cumulative >= 0.9)[0]
    n_80 = n_80[0] + 1 if len(n_80) > 0 else len(cumulative)
    n_90 = n_90[0] + 1 if len(n_90) > 0 else len(cumulative)
    
    ax2.axvline(x=n_80, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x=n_90, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Number of Features', fontsize=14, fontweight='normal')
    ax2.set_ylabel('Cumulative Importance', fontsize=14, fontweight='normal')
    ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='both', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    ax2.set_axisbelow(True)
    ax2.legend(loc='lower right', frameon=True, facecolor='#f0f0f0', edgecolor='black', fontsize=10)
    
    # Add annotation
    ax2.text(0.05, 0.95, f'80% variance: {n_80} features\n90% variance: {n_90} features',
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='black', alpha=0.8))
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax2.spines[spine].set_color('black')
        ax2.spines[spine].set_linewidth(0.8)
    ax2.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    ax2.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    #plt.suptitle(f'FEATURE IMPORTANCE ANALYSIS - {variable_name}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    if variable_name.lower() == 'turbidity':
        filename = f'{path_output}/figure5_feature_importance_turb.pdf'
    else:
        filename = f'{path_output}/figure5_feature_importance_chl.pdf'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure 5 saved as '{filename}'")
    print(f"  Importance method: {method}")
    print(f"  Top 5 features: {top_features.tail(5)['feature'].tolist()[-5:]}")
    
    return fig, feature_importance_df

def crear_feature_importance_tabpfn(modelo, X_train, y_train, feature_names, path_output, modelo_nombre, variable_name="Chlorophyll", n_repeats=5, n_samples=1000):
    """
    Creates feature importance plot for TabPFN using permutation importance.
    This method is model-agnostic and works for any classifier.
    
    Args:
        modelo: Trained TabPFN model
        X_train: Training features
        y_train: Training labels
        feature_names: List of feature names
        path_output: Output directory
        modelo_nombre: Name of the model
        variable_name: Name of target variable
        n_repeats: Number of permutations for each feature
        n_samples: Number of samples to use for permutation importance (for speed)
    """
    print(f"\nGenerating feature importance for {modelo_nombre} using permutation importance...")
    
    # Check if we have valid feature names and data
    if len(feature_names) == 0:
        print("  No features available for importance analysis.")
        return None, None
    
    # Use a subset of data for faster computation
    n_samples_use = min(n_samples, len(X_train))
    indices = np.random.choice(len(X_train), n_samples_use, replace=False)
    X_sample = X_train[indices]
    y_sample = y_train[indices]
    
    # Determine if classification or regression
    is_classification = hasattr(modelo, 'predict_proba') or (len(np.unique(y_sample)) <= 10)
    
    # Calculate baseline score
    if is_classification:
        y_pred = modelo.predict(X_sample)
        baseline_score = accuracy_score(y_sample, y_pred)
        scoring = 'accuracy'
    else:
        y_pred = modelo.predict(X_sample)
        baseline_score = r2_score(y_sample, y_pred)
        scoring = 'r2'
    
    print(f"  Baseline {scoring}: {baseline_score:.4f}")
    
    # Calculate permutation importance for each feature
    importances = []
    importance_stds = []
    
    for idx, feature in enumerate(feature_names):
        print(f"  Computing importance for feature {idx+1}/{len(feature_names)}: {feature}...")
        
        # Store scores for this feature across repeats
        scores = []
        
        for repeat in range(n_repeats):
            # Permute the feature
            X_permuted = X_sample.copy()
            X_permuted[:, idx] = np.random.permutation(X_permuted[:, idx])
            
            # Predict with permuted data
            if is_classification:
                y_pred_perm = modelo.predict(X_permuted)
                score = accuracy_score(y_sample, y_pred_perm)
            else:
                y_pred_perm = modelo.predict(X_permuted)
                score = r2_score(y_sample, y_pred_perm)
            
            scores.append(score)
        
        # Importance is the drop in performance
        importance = baseline_score - np.mean(scores)
        importance_std = np.std(scores)
        
        importances.append(importance)
        importance_stds.append(importance_std)
        
        print(f"    Importance: {importance:.4f} ± {importance_std:.4f}")
    
    # Create DataFrame for plotting
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'std': importance_stds
    }).sort_values('importance', ascending=True)
    
    # Filter out features with zero or negative importance
    feature_importance_df = feature_importance_df[feature_importance_df['importance'] > 0]
    
    if len(feature_importance_df) == 0:
        print("  No features with positive importance found.")
        return None, None
    
    # Select top 15 features (or all if less)
    top_n = min(15, len(feature_importance_df))
    top_features = feature_importance_df.tail(top_n)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Plot 1: Horizontal bar chart with error bars
    ax1.set_facecolor('white')
    y_pos = np.arange(len(top_features))
    bars = ax1.barh(y_pos, top_features['importance'].values, 
                    xerr=top_features['std'].values,
                    color='#377EB8', edgecolor='black', linewidth=0.8,
                    capsize=3, error_kw={'linewidth': 1.5, 'color': 'black'})
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_features['feature'].values, fontsize=10)
    ax1.set_xlabel('Permutation Importance (Drop in Accuracy)', fontsize=14, fontweight='normal')
    ax1.set_title(f'Feature Importance - {modelo_nombre}\n(Permutation Importance)', fontsize=12, fontweight='bold')
    ax1.grid(True, axis='x', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    ax1.set_axisbelow(True)
    
    # Add value labels
    max_imp = top_features['importance'].max()
    for bar, val, std in zip(bars, top_features['importance'].values, top_features['std'].values):
        width = bar.get_width()
        ax1.text(width + (max_imp * 0.02), 
                bar.get_y() + bar.get_height()/2,
                f'{val:.4f} ± {std:.4f}', ha='left', va='center', fontsize=8)
    
    # Plot 2: Cumulative importance
    ax2.set_facecolor('white')
    sorted_importance = feature_importance_df.sort_values('importance', ascending=False)
    cumulative = np.cumsum(sorted_importance['importance'].values) / np.sum(sorted_importance['importance'].values)
    
    ax2.plot(range(1, len(cumulative) + 1), cumulative, 'o-', color='#E41A1C', 
            linewidth=2, markersize=6, markeredgecolor='black', markeredgewidth=0.5)
    ax2.axhline(y=0.8, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='80% threshold')
    ax2.axhline(y=0.9, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='90% threshold')
    ax2.fill_between(range(1, len(cumulative) + 1), cumulative, alpha=0.2, color='#E41A1C')
    
    # Find number of features for 80% and 90% cumulative importance
    n_80 = np.where(cumulative >= 0.8)[0]
    n_90 = np.where(cumulative >= 0.9)[0]
    n_80 = n_80[0] + 1 if len(n_80) > 0 else len(cumulative)
    n_90 = n_90[0] + 1 if len(n_90) > 0 else len(cumulative)
    
    ax2.axvline(x=n_80, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x=n_90, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Number of Features', fontsize=14, fontweight='normal')
    ax2.set_ylabel('Cumulative Importance', fontsize=14, fontweight='normal')
    ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='both', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    ax2.set_axisbelow(True)
    ax2.legend(loc='lower right', frameon=True, facecolor='#f0f0f0', edgecolor='black', fontsize=10)
    
    # Add annotation
    ax2.text(0.05, 0.95, f'80% variance: {n_80} features\n90% variance: {n_90} features',
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='black', alpha=0.8))
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax2.spines[spine].set_color('black')
        ax2.spines[spine].set_linewidth(0.8)
    ax2.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    ax2.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    #plt.suptitle(f'FEATURE IMPORTANCE ANALYSIS WITH TABPFN - {variable_name}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    if variable_name.lower() == 'turbidity':
        filename = f'{path_output}/figure5_feature_importance_tabpfn_turb.pdf'
    else:
        filename = f'{path_output}/figure5_feature_importance_tabpfn_chl.pdf'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure 5 (TabPFN) saved as '{filename}'")
    print(f"  Importance method: Permutation Importance")
    print(f"  Top 5 features: {top_features.tail(5)['feature'].tolist()[-5:]}")
    
    return fig, feature_importance_df

def crear_temporal_decomposition(df, target_column, path_output, variable_name="Chlorophyll"):
    """
    Creates temporal decomposition plot showing seasonal and annual patterns.
    Figure 4c: Temporal decomposition of chlorophyll time series.
    """
    print("\nGenerating temporal decomposition...")
    
    if 'TIMESTAMP' not in df.columns:
        print("TIMESTAMP column not found. Cannot generate temporal decomposition.")
        return None, None, None, None
    
    # Sort by timestamp
    df_sorted = df.sort_values('TIMESTAMP').copy()
    
    # Create time-based aggregations
    df_sorted['month'] = df_sorted['TIMESTAMP'].dt.month
    df_sorted['year'] = df_sorted['TIMESTAMP'].dt.year
    df_sorted['dayofyear'] = df_sorted['TIMESTAMP'].dt.dayofyear
    
    # Calculate daily, monthly, and yearly averages
    daily_avg = df_sorted.groupby('dayofyear')[target_column].mean()
    monthly_avg = df_sorted.groupby('month')[target_column].mean()
    
    # Handle yearly aggregation - remove NaN and ensure we have data
    yearly_data = df_sorted.groupby('year')[target_column].mean()
    yearly_avg = yearly_data.dropna()
    
    # Check if we have enough data for yearly analysis
    has_yearly = len(yearly_avg) >= 2
    
    # Create figure with subplots
    if has_yearly:
        fig = plt.figure(figsize=(14, 10))
        ax1 = plt.subplot(3, 1, 1)
        ax2 = plt.subplot(3, 1, 2)
        ax3 = plt.subplot(3, 1, 3)
    else:
        fig = plt.figure(figsize=(14, 8))
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        ax3 = None
    
    # Plot 1: Time series with trend
    ax1.set_facecolor('white')
    
    # Filter out any inf or NaN values for plotting
    valid_mask = np.isfinite(df_sorted[target_column].values)
    valid_timestamps = df_sorted['TIMESTAMP'].values[valid_mask]
    valid_values = df_sorted[target_column].values[valid_mask]
    
    ax1.plot(valid_timestamps, valid_values, 
            color='gray', linewidth=0.8, alpha=0.5, label='Original')
    
    # Add rolling average (only if enough data)
    if len(valid_values) > 30:
        rolling_avg = pd.Series(valid_values).rolling(window=30, center=True).mean().values
        ax1.plot(valid_timestamps, rolling_avg, 
                color='#E41A1C', linewidth=2, label='30-day Rolling Average')
    
    # Add trend line (only if enough data)
    if len(valid_values) > 1:
        x_numeric = np.arange(len(valid_values))
        # Remove any NaN or inf values for polyfit
        valid_idx = np.isfinite(valid_values)
        if np.sum(valid_idx) > 1:
            x_clean = x_numeric[valid_idx]
            y_clean = valid_values[valid_idx]
            try:
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                ax1.plot(valid_timestamps[valid_idx], p(x_clean), 
                        'k--', linewidth=1.5, label=f'Trend (slope: {z[0]:.4f})')
            except Exception as e:
                print(f"  Warning: Could not fit trend line: {e}")
    
    ax1.set_xlabel('Date', fontsize=14, fontweight='normal')
    ax1.set_ylabel('Chlorophyll (µg/L)', fontsize=14, fontweight='normal')
    ax1.set_title('A) Time Series with Trend', fontsize=14, fontweight='bold', loc='left')
    ax1.grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper left', frameon=True, facecolor='#f0f0f0', edgecolor='black', fontsize=10)
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax1.spines[spine].set_color('black')
        ax1.spines[spine].set_linewidth(0.8)
    ax1.tick_params(axis='x', rotation=45, length=4, width=0.8, color='black', labelsize=9)
    ax1.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    # Plot 2: Seasonal pattern (monthly)
    ax2.set_facecolor('white')
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Ensure we have data for all months (fill NaN with 0)
    monthly_values = []
    for m in range(1, 13):
        val = monthly_avg.get(m, 0)
        if np.isnan(val) or np.isinf(val):
            val = 0
        monthly_values.append(val)
    
    bar_colors = ['#2ecc71' if m in [3,4,5] else 
                  '#f1c40f' if m in [6,7,8] else 
                  '#e67e22' if m in [9,10,11] else 
                  '#3498db' for m in range(1, 13)]
    
    bars = ax2.bar(months, monthly_values, color=bar_colors, 
                   edgecolor='black', linewidth=0.8, width=0.7)
    ax2.set_xlabel('Month', fontsize=14, fontweight='normal')
    ax2.set_ylabel('Average Chlorophyll (µg/L)', fontsize=14, fontweight='normal')
    ax2.set_title('B) Seasonal Pattern', fontsize=14, fontweight='bold', loc='left')
    ax2.grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    ax2.set_axisbelow(True)
    
    # Add value labels only for non-zero values
    max_val = max(monthly_values) if monthly_values else 1
    for bar, val in zip(bars, monthly_values):
        if val > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (max_val * 0.02),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Find seasonal peaks
    if monthly_values:
        seasonal_max_month = np.argmax(monthly_values) + 1
        seasonal_min_month = np.argmin(monthly_values) + 1
        ax2.text(0.98, 0.95, f'Peak: Month {seasonal_max_month}\nMinimum: Month {seasonal_min_month}',
                transform=ax2.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='black', alpha=0.8))
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax2.spines[spine].set_color('black')
        ax2.spines[spine].set_linewidth(0.8)
    ax2.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    ax2.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    # Plot 3: Yearly variation (if we have data)
    if has_yearly and ax3 is not None:
        ax3.set_facecolor('white')
        
        years = [str(int(y)) for y in yearly_avg.index]
        yearly_values = yearly_avg.values
        
        # Replace any NaN/inf values
        yearly_values = np.nan_to_num(yearly_values, nan=0, posinf=0, neginf=0)
        
        if len(years) > 0 and np.sum(yearly_values) > 0:
            bars3 = ax3.bar(years, yearly_values, color='#377EB8', 
                            edgecolor='black', linewidth=0.8, width=0.7)
            ax3.set_xlabel('Year', fontsize=14, fontweight='normal')
            ax3.set_ylabel('Annual Average Chlorophyll (µg/L)', fontsize=14, fontweight='normal')
            ax3.set_title('C) Yearly Variation', fontsize=14, fontweight='bold', loc='left')
            ax3.grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
            ax3.set_axisbelow(True)
            
            # Add value labels
            max_yearly = max(yearly_values) if len(yearly_values) > 0 else 1
            for bar, val in zip(bars3, yearly_values):
                if val > 0:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + (max_yearly * 0.02),
                            f'{val:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Add trend annotation (only if enough points)
            if len(yearly_values) >= 2:
                try:
                    y_years = np.arange(len(yearly_values))
                    # Filter out any invalid values
                    valid_idx = np.isfinite(yearly_values)
                    if np.sum(valid_idx) >= 2:
                        z_yearly = np.polyfit(y_years[valid_idx], yearly_values[valid_idx], 1)
                        trend_text = f'Annual Trend: {z_yearly[0]:.3f} µg/L/year'
                        ax3.text(0.02, 0.95, trend_text, transform=ax3.transAxes, fontsize=10,
                                verticalalignment='top', 
                                bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='black', alpha=0.8))
                except Exception as e:
                    print(f"  Warning: Could not calculate yearly trend: {e}")
        
        for spine in ['top', 'bottom', 'left', 'right']:
            ax3.spines[spine].set_color('black')
            ax3.spines[spine].set_linewidth(0.8)
        ax3.tick_params(axis='x', rotation=45, length=4, width=0.8, color='black', labelsize=10)
        ax3.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    elif not has_yearly:
        print("  Note: Not enough yearly data for yearly variation plot")
    
    #plt.suptitle(f'TEMPORAL DECOMPOSITION - {variable_name}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    if variable_name.lower() == 'turbidity':
        filename = f'{path_output}/figure4c_temporal_decomposition_turb.pdf'
    else:
        filename = f'{path_output}/figure4c_temporal_decomposition_chl.pdf'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure 4c saved as '{filename}'")
    
    return fig, daily_avg, monthly_avg, yearly_avg

# ====================================================================
# VISUALIZATION FUNCTIONS AND COMPUTATIONAL METRICS
# ====================================================================

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculates Expected Calibration Error (ECE) for multiclass classification.
    """
    # Convert y_true to one-hot encoding if necessary
    if len(y_prob.shape) == 1 or y_prob.shape[1] == 1:
        # Binary
        y_prob = np.column_stack([1 - y_prob, y_prob])
    
    n_classes = y_prob.shape[1]
    ece = 0.0
    for c in range(n_classes):
        prob_c = y_prob[:, c]
        true_c = (y_true == c).astype(int)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (prob_c > bin_lower) & (prob_c <= bin_upper)
            if np.sum(in_bin) > 0:
                avg_prob = np.mean(prob_c[in_bin])
                avg_acc = np.mean(true_c[in_bin])
                ece += np.abs(avg_acc - avg_prob) * np.sum(in_bin) / len(y_true)
    
    return ece

def negative_log_likelihood(y_true, y_prob):
    """
    Calculates Negative Log-Likelihood (NLL) for classification.
    """
    # Convert y_true to one-hot encoding if necessary
    if len(y_prob.shape) == 1 or y_prob.shape[1] == 1:
        # Binary
        y_prob = np.column_stack([1 - y_prob, y_prob])
    
    n_samples = len(y_true)
    log_likelihood = 0.0
    for i in range(n_samples):
        true_class = y_true[i]
        prob_true_class = y_prob[i, true_class]
        log_likelihood += np.log(max(prob_true_class, 1e-15))  # Avoid log(0)
    
    return -log_likelihood / n_samples

def obtener_metricas_computacionales(modelo, X_train, y_train, X_test, y_test, modelo_nombre, tipo_modelo='auto'):
    """
    Calculates computational and performance metrics for a model.
    
    Args:
        tipo_modelo: 'classification', 'regression', or 'auto' (automatically detects)
    """
    metricas = {}
    
    # 1. Detect model type if 'auto'
    if tipo_modelo == 'auto':
        # Check if it's a regression model by name or type
        model_name = str(type(modelo)).lower()
        if any(term in model_name for term in ['regressor', 'regression', 'svr', 'ridge']):
            tipo_modelo = 'regression'
        elif hasattr(modelo, 'predict_proba'):
            tipo_modelo = 'classification'
        else:
            # Default to classification
            tipo_modelo = 'classification'
    
    # 2. Training time
    inicio_entrenamiento = time.time()
    modelo.fit(X_train, y_train)
    metricas['tiempo_entrenamiento'] = time.time() - inicio_entrenamiento
    
    # 3. Prediction time
    inicio_prediccion = time.time()
    y_pred = modelo.predict(X_test)
    metricas['tiempo_prediccion'] = time.time() - inicio_prediccion
    
    # 4. Model-specific metrics
    metricas['tipo_modelo'] = tipo_modelo
    metricas['nombre'] = modelo_nombre
    metricas['modelo'] = modelo
    metricas['y_pred'] = y_pred
    
    if tipo_modelo == 'classification':
        # CLASSIFICATION METRICS
        metricas['accuracy'] = accuracy_score(y_test, y_pred)
        
        # Precision, Recall, F1
        try:
            metricas['precision_macro'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
            metricas['recall_macro'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
            metricas['f1_macro'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            metricas['precision_weighted'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metricas['recall_weighted'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metricas['f1_weighted'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        except Exception as e:
            print(f"    Warning in classification metrics: {e}")
            metricas['precision_macro'] = metricas['recall_macro'] = metricas['f1_macro'] = 0
            metricas['precision_weighted'] = metricas['recall_weighted'] = metricas['f1_weighted'] = 0
        
        # Negative Log-Likelihood (NLL)
        if hasattr(modelo, 'predict_proba'):
            y_prob = modelo.predict_proba(X_test)
            metricas['nll'] = negative_log_likelihood(y_test, y_prob)
        else:
            metricas['nll'] = np.nan
        
        # Expected Calibration Error (ECE)
        if hasattr(modelo, 'predict_proba'):
            y_prob = modelo.predict_proba(X_test)
            metricas['ece'] = expected_calibration_error(y_test, y_prob)
        else:
            metricas['ece'] = np.nan
        
        # Confusion matrix
        metricas['matriz_confusion'] = confusion_matrix(y_test, y_pred)
        
        # For regression: None values
        metricas['mse'] = None
        metricas['mae'] = None
        metricas['r2_score'] = None
        metricas['y_pred_redondeado'] = y_pred  # Already integers
        
    else:  # REGRESSION
        # REGRESSION METRICS
        metricas['mse'] = mean_squared_error(y_test, y_pred)
        metricas['mae'] = mean_absolute_error(y_test, y_pred)
        metricas['r2_score'] = r2_score(y_test, y_pred)
        
        # NLL and ECE not applicable for regression
        metricas['nll'] = np.nan
        metricas['ece'] = np.nan
        
        # For classification: calculate accuracy of rounded values
        y_pred_redondeado = np.round(y_pred).astype(int)
        # Ensure rounded values are in original range
        y_pred_redondeado = np.clip(y_pred_redondeado, int(min(y_test)), int(max(y_test)))
        
        # Also round y_test for confusion matrix
        y_test_redondeado = np.round(y_test).astype(int)
        y_test_redondeado = np.clip(y_test_redondeado, int(min(y_test)), int(max(y_test)))
        
        metricas['y_pred_redondeado'] = y_pred_redondeado
        try:
            metricas['accuracy'] = accuracy_score(y_test_redondeado, y_pred_redondeado)
        except:
            metricas['accuracy'] = 0
        
        # For classification metrics: use rounded values
        try:
            metricas['precision_weighted'] = precision_score(y_test_redondeado, y_pred_redondeado, average='weighted', zero_division=0)
            metricas['recall_weighted'] = recall_score(y_test_redondeado, y_pred_redondeado, average='weighted', zero_division=0)
            metricas['f1_weighted'] = f1_score(y_test_redondeado, y_pred_redondeado, average='weighted', zero_division=0)
            metricas['precision_macro'] = precision_score(y_test_redondeado, y_pred_redondeado, average='macro', zero_division=0)
            metricas['recall_macro'] = recall_score(y_test_redondeado, y_pred_redondeado, average='macro', zero_division=0)
            metricas['f1_macro'] = f1_score(y_test_redondeado, y_pred_redondeado, average='macro', zero_division=0)
        except:
            metricas['precision_weighted'] = metricas['recall_weighted'] = metricas['f1_weighted'] = 0
            metricas['precision_macro'] = metricas['recall_macro'] = metricas['f1_macro'] = 0
        
        # Confusion matrix using rounded values (both rounded)
        try:
            metricas['matriz_confusion'] = confusion_matrix(y_test_redondeado, y_pred_redondeado)
        except Exception as e:
            print(f"    Error calculating confusion matrix for regression: {e}")
            metricas['matriz_confusion'] = None
    
    # 5. Memory usage
    proceso = psutil.Process()
    metricas['memoria_usada_mb'] = proceso.memory_info().rss / 1024 / 1024
    
    # Clean memory
    gc.collect()
    
    return metricas, y_pred

def crear_scatter_regresion(y_true, y_pred, modelo_nombre, target_name):
    """
    Creates separate scatter plots of predicted vs actual values for regression models in R style
    Returns two separate figures: one for predicted vs actual, one for residuals
    """
    
    # Calculate metrics for annotation
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    residuals = y_true - y_pred
    std_res = np.std(residuals)
    
    # =========================================================================
    # FIGURE 1: Scatter plot with identity line
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.set_facecolor('white')
    
    # Create scatter plot
    scatter = ax1.scatter(y_true, y_pred, alpha=0.6, s=50, 
                         c='#377EB8', edgecolor='black', linewidth=0.5)
    
    # Add identity line (y = x)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             label='Perfect Prediction (y = x)')
    
    # Add confidence bands (±20% deviation)
    ax1.fill_between([min_val, max_val], 
                     [min_val*0.8, max_val*0.8], 
                     [min_val*1.2, max_val*1.2], 
                     alpha=0.2, color='gray', label='±20% deviation')
    
    ax1.set_xlabel('Actual Values', fontsize=14, fontweight='normal')
    ax1.set_ylabel('Predicted Values', fontsize=14, fontweight='normal')
    ax1.grid(True, axis='both', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    ax1.set_axisbelow(True)
    ax1.legend(loc='lower right', frameon=True, facecolor='#f0f0f0', edgecolor='black')
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax1.spines[spine].set_color('black')
        ax1.spines[spine].set_linewidth(0.8)
    ax1.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    ax1.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    # Add metrics as text box
    textstr = f'R² = {r2:.4f}\nMSE = {mse:.4f}\nMAE = {mae:.4f}'
    props = dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='black', alpha=0.9)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save figure 1
    filename1 = f'regression_scatter_{modelo_nombre.replace(" ", "_").lower()}.pdf'
    plt.savefig(f'{path_output}/{filename1}', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    print(f"\nRegression scatter plot saved as '{filename1}'")
    
    # =========================================================================
    # FIGURE 2: Residuals plot
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.set_facecolor('white')
    
    # Create residuals scatter
    scatter2 = ax2.scatter(y_pred, residuals, alpha=0.6, s=50,
                          c='#E41A1C', edgecolor='black', linewidth=0.5)
    
    # Add zero line
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero residual')
    
    # Add confidence bands (±2σ)
    ax2.axhline(y=2*std_res, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label=f'±2σ ({2*std_res:.3f})')
    ax2.axhline(y=-2*std_res, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax2.set_xlabel('Predicted Values', fontsize=14, fontweight='normal')
    ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=14, fontweight='normal')
    ax2.set_title(f'{modelo_nombre} - Residuals Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='both', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper right', frameon=True, facecolor='#f0f0f0', edgecolor='black')
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax2.spines[spine].set_color('black')
        ax2.spines[spine].set_linewidth(0.8)
    ax2.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    ax2.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    # Add residual statistics
    res_text = f'Mean Residual: {np.mean(residuals):.4f}\nStd Residual: {std_res:.4f}'
    props2 = dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='black', alpha=0.9)
    ax2.text(0.05, 0.95, res_text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', bbox=props2)
    
    plt.suptitle(f'RESIDUAL ANALYSIS - {target_name}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure 2
    filename2 = f'residuals_plot_{modelo_nombre.replace(" ", "_").lower()}.pdf'
    plt.savefig(f'{path_output}/{filename2}', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"Residuals plot saved as '{filename2}'")
    
    return fig1, fig2

def crear_visualizaciones_eda(df, target_column='Mean_Chl_ugl'):
    """Creates exploratory visualizations of Mar Menor dataset in R style"""
    print("\nGenerating EDA visualizations for Mar Menor...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Temporal distribution of chlorophyll
    ax1 = plt.subplot(3, 3, 1)
    if 'TIMESTAMP' in df.columns:
        df_sorted = df.sort_values('TIMESTAMP')
        ax1.set_facecolor('white')
        ax1.plot(df_sorted['TIMESTAMP'], df_sorted[target_column], 
                color='green', linewidth=1.5, alpha=0.7)
        ax1.set_title('Temporal Evolution of Chlorophyll', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=14, fontweight='normal')
        ax1.set_ylabel('Chlorophyll (µg/L)', fontsize=14, fontweight='normal')
        ax1.grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
        for spine in ['top', 'bottom', 'left', 'right']:
            ax1.spines[spine].set_color('black')
            ax1.spines[spine].set_linewidth(0.8)
        ax1.tick_params(axis='x', rotation=45, length=4, width=0.8, color='black', labelsize=10)
        ax1.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    # 2. Chlorophyll distribution
    ax2 = plt.subplot(3, 3, 2)
    ax2.set_facecolor('white')
    ax2.hist(df[target_column], bins=30, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=0.8)
    ax2.set_title('Chlorophyll Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Chlorophyll (µg/L)', fontsize=14, fontweight='normal')
    ax2.set_ylabel('Frequency', fontsize=14, fontweight='normal')
    ax2.grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax2.spines[spine].set_color('black')
        ax2.spines[spine].set_linewidth(0.8)
    ax2.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    ax2.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    # 3. Correlation matrix (only numeric variables)
    ax3 = plt.subplot(3, 3, 3)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        corr_matrix = df[numeric_cols].corr()
        im = ax3.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        ax3.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(corr_matrix.columns)))
        ax3.set_yticks(range(len(corr_matrix.columns)))
        ax3.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
        ax3.set_yticklabels(corr_matrix.columns, fontsize=8)
        for spine in ['top', 'bottom', 'left', 'right']:
            ax3.spines[spine].set_color('black')
            ax3.spines[spine].set_linewidth(0.8)
        ax3.tick_params(axis='x', length=4, width=0.8, color='black')
        ax3.tick_params(axis='y', length=4, width=0.8, color='black')
        plt.colorbar(im, ax=ax3)
    
    # 4. Relationship between water temperature and chlorophyll
    ax4 = plt.subplot(3, 3, 4)
    if 'SDI_Temp_3m' in df.columns:
        ax4.set_facecolor('white')
        scatter = ax4.scatter(df['SDI_Temp_3m'], df[target_column], 
                             c=df['SDI_Temp_3m'], cmap='plasma', alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
        ax4.set_title('Water Temperature vs Chlorophyll', fontsize=14, fontweight='normal')
        ax4.set_xlabel('Water Temperature (°C)', fontsize=11)
        ax4.set_ylabel('Chlorophyll (µg/L)', fontsize=11)
        ax4.grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
        for spine in ['top', 'bottom', 'left', 'right']:
            ax4.spines[spine].set_color('black')
            ax4.spines[spine].set_linewidth(0.8)
        ax4.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
        ax4.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
        plt.colorbar(scatter, ax=ax4, label='Water Temp (°C)')
    
    # 5. Relationship between oxygen and chlorophyll
    ax5 = plt.subplot(3, 3, 5)
    if 'O2_sat2_Avg' in df.columns:
        ax5.set_facecolor('white')
        scatter = ax5.scatter(df['O2_sat2_Avg'], df[target_column], 
                             c=df['O2_sat2_Avg'], cmap='viridis', alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
        ax5.set_title('Dissolved Oxygen vs Chlorophyll', fontsize=14, fontweight='normal')
        ax5.set_xlabel('Oxygen Saturation (%)', fontsize=11)
        ax5.set_ylabel('Chlorophyll (µg/L)', fontsize=11)
        ax5.grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
        for spine in ['top', 'bottom', 'left', 'right']:
            ax5.spines[spine].set_color('black')
            ax5.spines[spine].set_linewidth(0.8)
        ax5.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
        ax5.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
        plt.colorbar(scatter, ax=ax5, label='O₂ (%)')
    
    # 6. Relationship between conductivity and chlorophyll
    ax6 = plt.subplot(3, 3, 6)
    if 'SDI_TempCorrCond_3m' in df.columns:
        ax6.set_facecolor('white')
        scatter = ax6.scatter(df['SDI_TempCorrCond_3m'], df[target_column], 
                             c=df['SDI_TempCorrCond_3m'], cmap='summer', alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
        ax6.set_title('Conductivity vs Chlorophyll', fontsize=14, fontweight='normal')
        ax6.set_xlabel('Corrected Conductivity', fontsize=11)
        ax6.set_ylabel('Chlorophyll (µg/L)', fontsize=11)
        ax6.grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
        for spine in ['top', 'bottom', 'left', 'right']:
            ax6.spines[spine].set_color('black')
            ax6.spines[spine].set_linewidth(0.8)
        ax6.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
        ax6.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
        plt.colorbar(scatter, ax=ax6, label='Conductivity')
    
    # 7. Relationship between wind speed and chlorophyll
    ax7 = plt.subplot(3, 3, 7)
    if 'WS_ms_Avg' in df.columns:
        ax7.set_facecolor('white')
        scatter = ax7.scatter(df['WS_ms_Avg'], df[target_column], 
                             c=df['WS_ms_Avg'], cmap='winter', alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
        ax7.set_title('Wind Speed vs Chlorophyll', fontsize=14, fontweight='normal')
        ax7.set_xlabel('Wind Speed (m/s)', fontsize=11)
        ax7.set_ylabel('Chlorophyll (µg/L)', fontsize=11)
        ax7.grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
        for spine in ['top', 'bottom', 'left', 'right']:
            ax7.spines[spine].set_color('black')
            ax7.spines[spine].set_linewidth(0.8)
        ax7.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
        ax7.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
        plt.colorbar(scatter, ax=ax7, label='Wind (m/s)')
    
    # 8. Daily variation of chlorophyll
    ax8 = plt.subplot(3, 3, 8)
    if 'hour' in df.columns:
        ax8.set_facecolor('white')
        hourly_avg = df.groupby('hour')[target_column].mean()
        ax8.plot(hourly_avg.index, hourly_avg.values, 
                color='darkgreen', linewidth=2, marker='o', markersize=6, markerfacecolor='white', markeredgecolor='black', markeredgewidth=0.8)
        ax8.fill_between(hourly_avg.index, hourly_avg.values, 
                        alpha=0.3, color='lightgreen')
        ax8.set_title('Hourly Chlorophyll Variation', fontsize=14, fontweight='normal')
        ax8.set_xlabel('Hour of Day', fontsize=11)
        ax8.set_ylabel('Average Chlorophyll (µg/L)', fontsize=11)
        ax8.set_xticks(range(0, 24, 3))
        ax8.grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
        for spine in ['top', 'bottom', 'left', 'right']:
            ax8.spines[spine].set_color('black')
            ax8.spines[spine].set_linewidth(0.8)
        ax8.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
        ax8.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    # 9. Chlorophyll categories
    ax9 = plt.subplot(3, 3, 9)
    if 'Chl_category' in df.columns:
        ax9.set_facecolor('white')
        counts = df['Chl_category'].value_counts()
        colors = ['#4DAF4A', '#377EB8', '#FF7F00', '#E41A1C']  # R ColorBrewer palette
        bars = ax9.bar(counts.index.astype(str), counts.values, 
                      color=colors, edgecolor='black', linewidth=0.8, width=0.7)
        ax9.set_title('Chlorophyll Categories', fontsize=14, fontweight='normal')
        ax9.set_xlabel('Category', fontsize=11)
        ax9.set_ylabel('Frequency', fontsize=11)
        ax9.grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
        ax9.set_axisbelow(True)
        for spine in ['top', 'bottom', 'left', 'right']:
            ax9.spines[spine].set_color('black')
            ax9.spines[spine].set_linewidth(0.8)
        ax9.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
        ax9.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
        
        y_max = max(counts.values)
        ax9.set_ylim(0, y_max * 1.15)
        
        for bar, count in zip(bars, counts.values):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height + (y_max * 0.02),
                    f'{int(height)}', ha='center', va='bottom', fontsize=12)
    
    plt.suptitle('EXPLORATORY DATA ANALYSIS - MAR MENOR (Chlorophyll)', 
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{path_output}/eda_mar_menor.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig

def visualizar_metricas_computacionales(metricas_modelos):
    """
    Creates visualizations for computational metrics in R style
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    nombres = [m['nombre'] for m in metricas_modelos]
    tipos = [m.get('tipo_modelo', 'classification') for m in metricas_modelos]
    
    # 1. Training time
    tiempos_ent = [m['tiempo_entrenamiento'] for m in metricas_modelos]
    axes[0, 0].set_facecolor('white')
    bars1 = axes[0, 0].bar(nombres, tiempos_ent, 
                          color=['skyblue' if t == 'classification' else 'lightcoral' for t in tipos],
                          edgecolor='black', linewidth=0.8, width=0.7)
    axes[0, 0].set_title('Training Time (seconds)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Seconds', fontsize=14, fontweight='normal')
    axes[0, 0].grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    axes[0, 0].set_axisbelow(True)
    for spine in ['top', 'bottom', 'left', 'right']:
        axes[0, 0].spines[spine].set_color('black')
        axes[0, 0].spines[spine].set_linewidth(0.8)
    axes[0, 0].tick_params(axis='x', rotation=45, length=4, width=0.8, color='black', labelsize=10)
    axes[0, 0].tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    y_max = max(tiempos_ent)
    for bar, tiempo in zip(bars1, tiempos_ent):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + (y_max * 0.02),
                       f'{tiempo:.2f}s', ha='center', va='bottom', fontsize=9)
    
    # 2. Accuracy or R² according to model type
    metricas_principal = []
    for m in metricas_modelos:
        if m.get('tipo_modelo') == 'regression':
            val = m.get('r2_score', 0)
        else:
            val = m.get('accuracy', 0)
        # Handle None values
        if val is None:
            val = 0
        metricas_principal.append(val)
    
    colors_principal = ['gold' if t == 'classification' else 'lightgreen' for t in tipos]
    
    axes[0, 1].set_facecolor('white')
    bars2 = axes[0, 1].bar(nombres, metricas_principal, color=colors_principal,
                          edgecolor='black', linewidth=0.8, width=0.7)
    axes[0, 1].set_title('Main Metric by Model', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Value', fontsize=14, fontweight='normal')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    axes[0, 1].set_axisbelow(True)
    for spine in ['top', 'bottom', 'left', 'right']:
        axes[0, 1].spines[spine].set_color('black')
        axes[0, 1].spines[spine].set_linewidth(0.8)
    axes[0, 1].tick_params(axis='x', rotation=45, length=4, width=0.8, color='black', labelsize=10)
    axes[0, 1].tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    # Legend for model types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gold', edgecolor='black', label='Classification (Accuracy)'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Regression (R²)')
    ]
    axes[0, 1].legend(handles=legend_elements, loc='upper right', frameon=True, edgecolor='black')
    
    for bar, valor in zip(bars2, metricas_principal):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{valor:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Negative Log-Likelihood (NLL) - only for classification
    nlls = []
    for m in metricas_modelos:
        val = m.get('nll', np.nan)
        if val is None:
            val = np.nan
        nlls.append(val)
    nlls_filtered = [n if not np.isnan(n) else 0 for n in nlls]
    
    axes[1, 0].set_facecolor('white')
    bars3 = axes[1, 0].bar(nombres, nlls_filtered, 
                          color=['skyblue' if t == 'classification' else 'lightgray' for t in tipos],
                          edgecolor='black', linewidth=0.8, width=0.7)
    axes[1, 0].set_title('Negative Log-Likelihood (NLL)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('NLL', fontsize=14, fontweight='normal')
    axes[1, 0].grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    axes[1, 0].set_axisbelow(True)
    for spine in ['top', 'bottom', 'left', 'right']:
        axes[1, 0].spines[spine].set_color('black')
        axes[1, 0].spines[spine].set_linewidth(0.8)
    axes[1, 0].tick_params(axis='x', rotation=45, length=4, width=0.8, color='black', labelsize=10)
    axes[1, 0].tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    valid_nlls = [n for n in nlls if not np.isnan(n)]
    nll_max = max(valid_nlls) if valid_nlls else 0
    for bar, nll, tipo in zip(bars3, nlls, tipos):
        if tipo == 'classification' and not np.isnan(nll):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + (nll_max * 0.02),
                           f'{nll:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Expected Calibration Error (ECE) - only for classification
    eces = []
    for m in metricas_modelos:
        val = m.get('ece', np.nan)
        if val is None:
            val = np.nan
        eces.append(val)
    eces_filtered = [e if not np.isnan(e) else 0 for e in eces]
    
    axes[1, 1].set_facecolor('white')
    bars4 = axes[1, 1].bar(nombres, eces_filtered, 
                          color=['skyblue' if t == 'classification' else 'lightgray' for t in tipos],
                          edgecolor='black', linewidth=0.8, width=0.7)
    axes[1, 1].set_title('Expected Calibration Error (ECE)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('ECE', fontsize=14, fontweight='normal')
    axes[1, 1].grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    axes[1, 1].set_axisbelow(True)
    for spine in ['top', 'bottom', 'left', 'right']:
        axes[1, 1].spines[spine].set_color('black')
        axes[1, 1].spines[spine].set_linewidth(0.8)
    axes[1, 1].tick_params(axis='x', rotation=45, length=4, width=0.8, color='black', labelsize=10)
    axes[1, 1].tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    valid_eces = [e for e in eces if not np.isnan(e)]
    ece_max = max(valid_eces) if valid_eces else 0
    for bar, ece, tipo in zip(bars4, eces, tipos):
        if tipo == 'classification' and not np.isnan(ece):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + (ece_max * 0.02),
                           f'{ece:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Memory usage
    memorias = [m['memoria_usada_mb'] for m in metricas_modelos]
    axes[1, 2].set_facecolor('white')
    bars5 = axes[1, 2].bar(nombres, memorias, color='lightblue',
                          edgecolor='black', linewidth=0.8, width=0.7)
    axes[1, 2].set_title('Memory Usage (MB)', fontsize=14, fontweight='bold')
    axes[1, 2].set_ylabel('Megabytes (MB)', fontsize=14, fontweight='normal')
    axes[1, 2].grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    axes[1, 2].set_axisbelow(True)
    for spine in ['top', 'bottom', 'left', 'right']:
        axes[1, 2].spines[spine].set_color('black')
        axes[1, 2].spines[spine].set_linewidth(0.8)
    axes[1, 2].tick_params(axis='x', rotation=45, length=4, width=0.8, color='black', labelsize=10)
    axes[1, 2].tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    mem_max = max(memorias)
    for bar, memoria in zip(bars5, memorias):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + (mem_max * 0.02),
                       f'{memoria:.0f}MB', ha='center', va='bottom', fontsize=9)
    
    # 6. Metrics comparison table
    axes[0, 2].axis('off')
    axes[0, 2].set_facecolor('white')
    
    # Create table with main metrics (CORREGIDO - manejo de valores None)
    tabla_data = []
    for m in metricas_modelos[:10]:  # Limit to 10 models for readability
        # Obtener valor principal con manejo de None
        main_val = m.get('r2_score', m.get('accuracy', 0))
        if main_val is None:
            main_val = 0
        main_str = f"{main_val:.3f}"
        
        # Obtener NLL con manejo de None
        nll_val = m.get('nll', np.nan)
        if nll_val is None:
            nll_val = np.nan
        nll_str = f"{nll_val:.3f}" if not np.isnan(nll_val) else 'N/A'
        
        # Obtener ECE con manejo de None
        ece_val = m.get('ece', np.nan)
        if ece_val is None:
            ece_val = np.nan
        ece_str = f"{ece_val:.3f}" if not np.isnan(ece_val) else 'N/A'
        
        row = [
            m['nombre'][:15],
            m.get('tipo_modelo', 'classification')[:4],
            main_str,
            nll_str,
            ece_str
        ]
        tabla_data.append(row)
    
    col_labels = ['Model', 'Type', 'Main', 'NLL', 'ECE']
    tabla = axes[0, 2].table(cellText=tabla_data, colLabels=col_labels,
                            cellLoc='center', loc='center',
                            colWidths=[0.15, 0.08, 0.1, 0.1, 0.1])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1.2, 1.5)
    for (i, j), cell in tabla.get_celld().items():
        if i == 0:  # Header
            cell.set_facecolor('#f0f0f0')
        cell.set_edgecolor('black')
        cell.set_linewidth(0.8)
    axes[0, 2].set_title('Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('COMPUTATIONAL METRICS - MAR MENOR', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{path_output}/computational_metrics_mar_menor.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig

def crear_tabla_resumen_metricas(metricas_modelos):
    """
    Creates a summary table with all metrics
    """
    print("\n" + "="*80)
    print("COMPUTATIONAL METRICS SUMMARY - MAR MENOR")
    print("="*80)
    
    # Create DataFrame
    datos = []
    for m in metricas_modelos:
        tipo = m.get('tipo_modelo', 'classification')
        if tipo == 'regression':
            r2_val = m.get('r2_score', 0)
            if r2_val is None:
                r2_val = 0
            metrica_principal = f"R²: {r2_val:.4f}"
        else:
            acc_val = m.get('accuracy', 0)
            if acc_val is None:
                acc_val = 0
            metrica_principal = f"Acc: {acc_val:.4f}"
        
        # Manejar valores None en todas las métricas
        accuracy = m.get('accuracy', 0)
        if accuracy is None:
            accuracy = 0
            
        f1_macro = m.get('f1_macro', 0)
        if f1_macro is None:
            f1_macro = 0
            
        f1_weighted = m.get('f1_weighted', 0)
        if f1_weighted is None:
            f1_weighted = 0
            
        nll = m.get('nll', np.nan)
        if nll is None:
            nll = np.nan
            
        ece = m.get('ece', np.nan)
        if ece is None:
            ece = np.nan
            
        mse = m.get('mse', None)
        mae = m.get('mae', None)
        
        datos.append({
            'Model': m['nombre'],
            'Type': tipo[:10],
            'Main Metric': metrica_principal,
            'Accuracy': f"{accuracy:.4f}",
            'F1 Macro': f"{f1_macro:.4f}",
            'F1 Weighted': f"{f1_weighted:.4f}",
            'NLL': f"{nll:.4f}" if not np.isnan(nll) else 'N/A',
            'ECE': f"{ece:.4f}" if not np.isnan(ece) else 'N/A',
            'MSE': f"{mse:.4f}" if mse is not None else 'N/A',
            'MAE': f"{mae:.4f}" if mae is not None else 'N/A',
            'Train Time (s)': f"{m['tiempo_entrenamiento']:.2f}",
            'Pred Time (s)': f"{m['tiempo_prediccion']:.3f}",
            'Memory (MB)': f"{m['memoria_usada_mb']:.0f}"
        })
    
    df_resumen = pd.DataFrame(datos)
    print(df_resumen.to_string(index=False))
    
    # Save to CSV
    df_resumen.to_csv(f'{path_output}/metrics_summary_mar_menor.csv', index=False)
    print(f"\n✓ Summary saved to 'metrics_summary_mar_menor.csv'")
    
    return df_resumen

def crear_grafico_comparativo_modelos(resultados_modelos):
    """Creates comparative plot of all models in R style"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Data for plot
    modelos = list(resultados_modelos.keys())
    accuracies = []
    for m in modelos:
        acc = resultados_modelos[m]['accuracy']
        if acc is None:
            acc = 0
        accuracies.append(acc)
    
    # Bar plot (horizontal)
    ax1.set_facecolor('white')
    bars = ax1.barh(modelos, accuracies, color='lightblue', edgecolor='black', linewidth=0.8)
    ax1.set_xlabel('Accuracy', fontsize=14, fontweight='normal')
    ax1.set_title('Accuracy Comparison Between Models', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 1])
    ax1.grid(True, axis='x', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    ax1.set_axisbelow(True)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax1.spines[spine].set_color('black')
        ax1.spines[spine].set_linewidth(0.8)
    ax1.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    ax1.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    # Add values on bars
    x_max = max(accuracies)
    for bar, acc in zip(bars, accuracies):
        width = bar.get_width()
        ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{acc:.4f}', ha='left', va='center', fontsize=12)
    
    # Radar plot for multiple metrics
    if len(modelos) >= 3:
        # Additional metrics
        metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Calibration (1-ECE)']
        angles = np.linspace(0, 2*np.pi, len(metricas), endpoint=False).tolist()
        angles += angles[:1]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.set_facecolor('white')
        colors_radar = ['#E41A1C', '#377EB8', '#4DAF4A']  # R ColorBrewer palette
        
        for i, modelo in enumerate(modelos[:3]):  # Show only 3 models
            if 'metricas' in resultados_modelos[modelo]:
                m = resultados_modelos[modelo]['metricas']
                # Calibrate ECE (1 - ECE so higher values are better)
                ece_val = m.get('ece', 0)
                if ece_val is None or np.isnan(ece_val):
                    ece_val = 0
                cal_score = 1 - ece_val
                
                # Get metrics with None handling
                acc_val = resultados_modelos[modelo]['accuracy']
                if acc_val is None:
                    acc_val = 0
                    
                prec_val = m.get('precision_weighted', 0)
                if prec_val is None:
                    prec_val = 0
                    
                rec_val = m.get('recall_weighted', 0)
                if rec_val is None:
                    rec_val = 0
                    
                f1_val = m.get('f1_weighted', 0)
                if f1_val is None:
                    f1_val = 0
                
                valores = [
                    acc_val,
                    prec_val,
                    rec_val,
                    f1_val,
                    cal_score
                ]
            else:
                acc_val = resultados_modelos[modelo]['accuracy']
                if acc_val is None:
                    acc_val = 0
                valores = [acc_val, 0.7, 0.7, 0.7, 0.5]
            
            valores += valores[:1]
            ax2.plot(angles, valores, 'o-', linewidth=2, color=colors_radar[i], label=modelo)
            ax2.fill(angles, valores, alpha=0.1, color=colors_radar[i])
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metricas, fontsize=12)
        ax2.set_title('Metrics Comparison by Model', fontsize=14, fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), frameon=True, edgecolor='black')
        ax2.grid(True, color='black', linestyle=':', linewidth=0.3, alpha=1.0)
        ax2.tick_params(axis='both', length=4, width=0.8, color='black', labelsize=9)
    
    plt.suptitle('COMPARATIVE ANALYSIS OF MODELS - MAR MENOR', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{path_output}/model_comparison_mar_menor.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig

def crear_grafico_comparativo_modelos_clasificacion(resultados_modelos):
    """Creates comparative plot of classification models only in R style"""
    # Diccionario de mapeo para renombrar los modelos
    renombres_modelos = {
        'TabPFN_Clas': 'TabPFN',
        'RandomForest_Optimized': 'Random Forest Optimized',
        'GradientBoosting_Optimized': 'Gradient Boosting Optimized',
        'RandomForestClassifier_Clas': 'Random Forest Default',
        'KNeighborsClassifier_Clas': 'K Neighbors',
        'GradientBoostingClassifier_Clas': 'Gradient Boosting Default',
        'SVM_Optimized': 'SVM Optimized',
        'SVM_Clas': 'SVM Default'
    }
    
    # Filter only classification models
    modelos_clasificacion = {k: v for k, v in resultados_modelos.items() 
                            if v.get('tipo') == 'classification'}
    
    if not modelos_clasificacion:
        print("No classification models found to plot.")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Data for plot
    modelos = list(modelos_clasificacion.keys())
    accuracies = []
    for m in modelos:
        acc = modelos_clasificacion[m]['accuracy']
        if acc is None:
            acc = 0
        accuracies.append(acc)
    
    # Sort models by accuracy in descending order
    modelos_ordenados = sorted(zip(modelos, accuracies), key=lambda x: x[1], reverse=True)
    
    # Unzip the sorted data
    modelos_sorted = [m[0] for m in modelos_ordenados]
    accuracies_sorted = [m[1] for m in modelos_ordenados]
    
    # Crear lista de nombres personalizados manteniendo el orden
    modelos_sorted_custom = [renombres_modelos.get(modelo, modelo) for modelo in modelos_sorted]
    
    # Bar plot (horizontal) - All models sorted by accuracy
    ax1.set_facecolor('white')
    bars = ax1.barh(modelos_sorted_custom, accuracies_sorted, color='lightblue', edgecolor='black', linewidth=0.8)
    ax1.set_xlabel('Accuracy', fontsize=14, fontweight='normal')
    ax1.set_title('Accuracy Comparison - All Classification Models', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 1])
    ax1.grid(True, axis='x', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    ax1.set_axisbelow(True)
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax1.spines[spine].set_color('black')
        ax1.spines[spine].set_linewidth(0.8)
    
    ax1.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    ax1.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    # Add values on bars
    x_max = max(accuracies_sorted)
    for bar, acc in zip(bars, accuracies_sorted):
        width = bar.get_width()
        ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{acc:.4f}', ha='left', va='center', fontsize=12)
    
    # Radar plot for top 5 models based on accuracy
    top_5_modelos = modelos_ordenados[:5]
    top_5_nombres_originales = [m[0] for m in top_5_modelos]
    top_5_nombres_custom = [renombres_modelos.get(m[0], m[0]) for m in top_5_modelos]
    
    if len(top_5_modelos) >= 1:
        # Metrics for radar plot
        metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Calibration (1-ECE)']
        angles = np.linspace(0, 2*np.pi, len(metricas), endpoint=False).tolist()
        angles += angles[:1]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.set_facecolor('white')
        
        # Color palette for top 5 models
        colors_radar = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']
        
        # Store values for legend ordering
        legend_handles = []
        
        for i, (modelo_nombre_original, acc_val) in enumerate(top_5_modelos):
            modelo_nombre_custom = renombres_modelos.get(modelo_nombre_original, modelo_nombre_original)
            
            if 'metricas' in modelos_clasificacion[modelo_nombre_original]:
                m = modelos_clasificacion[modelo_nombre_original]['metricas']
                
                # Calibrate ECE (1 - ECE so higher values are better)
                ece_val = m.get('ece', 0)
                if ece_val is None or np.isnan(ece_val):
                    ece_val = 0
                cal_score = 1 - ece_val
                
                # Get metrics with None handling
                acc_val = modelos_clasificacion[modelo_nombre_original]['accuracy']
                if acc_val is None:
                    acc_val = 0
                    
                prec_val = m.get('precision_weighted', 0)
                if prec_val is None:
                    prec_val = 0
                    
                rec_val = m.get('recall_weighted', 0)
                if rec_val is None:
                    rec_val = 0
                    
                f1_val = m.get('f1_weighted', 0)
                if f1_val is None:
                    f1_val = 0
                
                valores = [
                    acc_val,
                    prec_val,
                    rec_val,
                    f1_val,
                    cal_score
                ]
            else:
                acc_val = modelos_clasificacion[modelo_nombre_original]['accuracy']
                if acc_val is None:
                    acc_val = 0
                valores = [acc_val, 0.7, 0.7, 0.7, 0.5]
            
            valores += valores[:1]
            
            # Plot line with marker
            line = ax2.plot(angles, valores, 'o-', linewidth=2, color=colors_radar[i % len(colors_radar)])
            ax2.fill(angles, valores, alpha=0.1, color=colors_radar[i % len(colors_radar)])
            
            # Highlight the best model (first one) with a thicker line
            if i == 0:
                ax2.plot(angles, valores, 'o-', linewidth=3, color=colors_radar[i])
        
        ax2.set_ylim([0.85, 1])
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metricas, fontsize=12)
        ax2.set_title(f'Top 5 Models - Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
        
        # Create legend with custom model names and their accuracy
        legend_labels = []
        for i, (modelo_nombre_original, acc_val) in enumerate(top_5_modelos):
            modelo_nombre_custom = renombres_modelos.get(modelo_nombre_original, modelo_nombre_original)
            # Acortar nombre si es muy largo para la leyenda
            short_name = modelo_nombre_custom[:25] + '...' if len(modelo_nombre_custom) > 25 else modelo_nombre_custom
            legend_labels.append(f"{i+1}. {short_name} (Acc:{acc_val:.3f})")
        
        ax2.legend(labels=legend_labels, loc='upper right', bbox_to_anchor=(1.4, 1.0), 
                  frameon=True, facecolor='#f0f0f0', edgecolor='black', fontsize=8)
        ax2.grid(True, color='black', linestyle=':', linewidth=0.3, alpha=1.0)
        ax2.tick_params(axis='both', length=4, width=0.8, color='black', labelsize=9)
        
        # Add a note about the metrics
        ax2.text(0, -0.15, 'Note: Higher values indicate better performance', 
                ha='center', va='center', fontsize=8, transform=ax2.transAxes,
                style='italic', bbox=dict(facecolor='#f0f0f0', edgecolor='black', boxstyle='round,pad=0.3'))
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'Insufficient models for radar plot', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes)
    
    plt.suptitle('COMPARATIVE ANALYSIS - CLASSIFICATION MODELS', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{path_output}/model_comparison_classification_mar_menor.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig

def crear_grafico_comparativo_modelos_regresion(resultados_modelos):
    """Creates comparative plot of regression models only in R style"""
    # Diccionario de mapeo para renombrar los modelos de regresión
    renombres_modelos_regresion = {
        'RidgeRegression_Reg': 'Ridge Regression',
        'LinearRegression_Reg': 'Linear Regression',
        'SVR_Reg': 'SVM Default',
        'SVR_Optimized': 'SVM Optimized',
        'RandomForestReg_Optimized': 'Random Forest Optimized',
        'RandomForestRegressor_Reg': 'Random Forest Default',
        'TabPFN_Reg': 'TabPFN'
    }
    
    # Filter only regression models
    modelos_regresion = {k: v for k, v in resultados_modelos.items() 
                        if v.get('tipo') == 'regression'}
    
    if not modelos_regresion:
        print("No regression models found to plot.")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Data for plot
    modelos = list(modelos_regresion.keys())
    accuracies = []
    r2_scores = []
    for m in modelos:
        acc = modelos_regresion[m]['accuracy']
        if acc is None:
            acc = 0
        accuracies.append(acc)
        
        r2 = modelos_regresion[m].get('r2_score', 0)
        if r2 is None:
            r2 = 0
        r2_scores.append(r2)
    
    # Sort models by R² score in descending order
    modelos_ordenados = sorted(zip(modelos, r2_scores, accuracies), key=lambda x: x[1], reverse=True)
    
    # Unzip the sorted data
    modelos_sorted = [m[0] for m in modelos_ordenados]
    r2_scores_sorted = [m[1] for m in modelos_ordenados]
    accuracies_sorted = [m[2] for m in modelos_ordenados]
    
    # Crear lista de nombres personalizados manteniendo el orden
    modelos_sorted_custom = [renombres_modelos_regresion.get(modelo, modelo) for modelo in modelos_sorted]
    
    # Bar plot (horizontal) - Using R² as main metric for regression (all models)
    ax1.set_facecolor('white')
    bars = ax1.barh(modelos_sorted_custom, r2_scores_sorted, color='lightcoral', edgecolor='black', linewidth=0.8)
    ax1.set_xlabel('R² Score', fontsize=14, fontweight='normal')
    ax1.set_title('R² Comparison - All Regression Models', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 1])
    ax1.grid(True, axis='x', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    ax1.set_axisbelow(True)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax1.spines[spine].set_color('black')
        ax1.spines[spine].set_linewidth(0.8)
    ax1.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    ax1.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    # Add values on bars
    x_max = max(r2_scores_sorted)
    for bar, r2 in zip(bars, r2_scores_sorted):
        width = bar.get_width()
        ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{r2:.4f}', ha='left', va='center', fontsize=12)
    
    # Bar plot for regression metrics (MSE, MAE, Accuracy rounded) - TOP 5 MODELS
    # Select top 5 models based on R² score
    top_5_modelos = modelos_ordenados[:5]
    top_5_nombres_originales = [m[0] for m in top_5_modelos]
    top_5_nombres_custom = [renombres_modelos_regresion.get(m[0], m[0]) for m in top_5_modelos]
    
    if len(top_5_modelos) >= 1:
        # Prepare data for grouped bar chart
        metric_names = ['MSE', 'MAE', 'Accuracy\n(rounded)']
        
        # Normalize metrics for better visualization
        mse_values = []
        mae_values = []
        acc_rounded_values = []
        
        for modelo_info in top_5_modelos:
            modelo_nombre_original = modelo_info[0]
            if 'metricas' in modelos_regresion[modelo_nombre_original]:
                m = modelos_regresion[modelo_nombre_original]['metricas']
                
                # Get MSE (normalized for visualization)
                mse = m.get('mse', 0)
                if mse is None or mse == 0:
                    mse_norm = 0
                else:
                    # Inverse normalization (lower MSE is better)
                    # Using min-max normalization across top models for better comparison
                    mse_norm = 1 / (1 + mse) if mse > 0 else 0
                mse_values.append(mse_norm)
                
                # Get MAE (normalized for visualization)
                mae = m.get('mae', 0)
                if mae is None or mae == 0:
                    mae_norm = 0
                else:
                    # Inverse normalization (lower MAE is better)
                    mae_norm = 1 / (1 + mae) if mae > 0 else 0
                mae_values.append(mae_norm)
                
                # Accuracy rounded
                acc_round = m.get('accuracy', 0)
                if acc_round is None:
                    acc_round = 0
                acc_rounded_values.append(acc_round)
            else:
                mse_values.append(0)
                mae_values.append(0)
                acc_rounded_values.append(0)
        
        # Further normalize MSE and MAE values to [0,1] range for better visualization
        if mse_values and max(mse_values) > 0:
            max_mse = max(mse_values)
            mse_values = [v / max_mse for v in mse_values]
        
        if mae_values and max(mae_values) > 0:
            max_mae = max(mae_values)
            mae_values = [v / max_mae for v in mae_values]
        
        # Set up bar positions
        x = np.arange(len(top_5_nombres_custom))
        width = 0.25
        
        ax2.set_facecolor('white')
        
        # Create bars for each metric
        bars_mse = ax2.bar(x - width, mse_values, width, 
                          label='MSE (normalized)', color='#E41A1C', edgecolor='black', linewidth=0.8)
        bars_mae = ax2.bar(x, mae_values, width, 
                          label='MAE (normalized)', color='#377EB8', edgecolor='black', linewidth=0.8)
        bars_acc = ax2.bar(x + width, acc_rounded_values, width, 
                          label='Accuracy', color='#4DAF4A', edgecolor='black', linewidth=0.8)
        
        # Highlight the best model (first one) with a different edge color or annotation
        bars_acc[0].set_edgecolor('gold')
        bars_acc[0].set_linewidth(1.5)
        bars_mse[0].set_edgecolor('gold')
        bars_mse[0].set_linewidth(1.5)
        bars_mae[0].set_edgecolor('gold')
        bars_mae[0].set_linewidth(1.5)
        
        ax2.set_title('Top 5 Regression Models - Metrics Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Model (sorted by R²)', fontsize=14, fontweight='normal')
        ax2.set_ylabel('Score (normalized)', fontsize=14, fontweight='normal')
        ax2.set_xticks(x)
        
        # Usar nombres personalizados para las etiquetas del eje x
        shortened_labels = []
        for nombre_custom in top_5_nombres_custom:
            if len(nombre_custom) > 15:
                # Acortar nombres largos si es necesario
                parts = nombre_custom.split()
                if len(parts) > 1:
                    abbr = parts[0][:8] + '...' + parts[-1][:4]
                else:
                    abbr = nombre_custom[:12] + '...'
                shortened_labels.append(abbr)
            else:
                shortened_labels.append(nombre_custom)
        
        ax2.set_xticklabels(shortened_labels, rotation=15, ha='right', fontsize=9)
        ax2.set_ylim([0, 1.15])
        ax2.grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
        ax2.set_axisbelow(True)
        ax2.legend(loc='upper right',  bbox_to_anchor=(0.95, 0.95), frameon=True, facecolor='#f0f0f0', edgecolor='black', fontsize=9)
        
        for spine in ['top', 'bottom', 'left', 'right']:
            ax2.spines[spine].set_color('black')
            ax2.spines[spine].set_linewidth(0.8)
        ax2.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=9)
        ax2.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
        
        # Add value labels on bars (only for Accuracy and optionally for others)
        for bar, val in zip(bars_acc, acc_rounded_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Add R² values as text above the bars (usando nombres personalizados en el texto si es necesario)
        for i, (nombre_original, nombre_custom, r2_val) in enumerate(zip(top_5_nombres_originales, top_5_nombres_custom, [m[1] for m in top_5_modelos])):
            ax2.text(i, 1.08, f'R²={r2_val:.3f}', ha='center', va='bottom', 
                    fontsize=8, fontweight='bold', rotation=0)
        
        # Add a note about the metrics
        ax2.text(0.5, -0.25, 
                'Note: MSE and MAE are normalized (higher values indicate better performance)',
                ha='center', va='center', fontsize=8, transform=ax2.transAxes,
                style='italic', bbox=dict(facecolor='#f0f0f0', edgecolor='black', boxstyle='round,pad=0.3'))
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'No regression models to display', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes)
    
    plt.suptitle('COMPARATIVE ANALYSIS - REGRESSION MODELS', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{path_output}/model_comparison_regression_mar_menor.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig

def mostrar_matriz_confusion(y_real, y_pred, titulo="Confusion Matrix", modelo_nombre=""):
    """Displays confusion matrix with enhanced visualizations in R style"""
    # Check if data is continuous and round if necessary
    if not np.issubdtype(y_real.dtype, np.integer) and not isinstance(y_real[0], (str, np.str_)):
        # If continuous, round for confusion matrix
        y_real_disp = np.round(y_real).astype(int)
        y_pred_disp = np.round(y_pred).astype(int)
    else:
        y_real_disp = y_real
        y_pred_disp = y_pred
    
    cm = confusion_matrix(y_real_disp, y_pred_disp)
    
    print(f"\n{titulo}:")
    print("="*50)
    
    # Create DataFrame for better visualization
    clases = sorted(np.unique(np.concatenate([y_real_disp, y_pred_disp])))
    cm_df = pd.DataFrame(cm, index=clases, columns=clases)
    
    print("Confusion matrix (rows: actual, columns: predicted):")
    print(cm_df)
    
    # Calculate metrics
    total = np.sum(cm)
    correctos = np.sum(np.diag(cm))
    precision_global = correctos / total if total > 0 else 0
    
    print(f"\nMetrics:")
    print(f"Global accuracy: {precision_global:.4f}")
    print(f"Total samples: {total}")
    print(f"Correct classifications: {correctos}")
    print(f"Incorrect classifications: {total - correctos}")
    
    # Calculate per-class metrics
    n_classes = len(clases)
    per_class_accuracy = []
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    
    for i, clase in enumerate(clases):
        # True Positives for this class
        tp = cm[i, i]
        
        # False Positives: sum of column i minus tp
        fp = np.sum(cm[:, i]) - tp
        
        # False Negatives: sum of row i minus tp
        fn = np.sum(cm[i, :]) - tp
        
        # True Negatives: total - (tp + fp + fn)
        tn = total - (tp + fp + fn)
        
        # Calculate metrics
        # Accuracy for this class = (TP + TN) / (TP + TN + FP + FN)
        acc = (tp + tn) / total if total > 0 else 0
        per_class_accuracy.append(acc)
        
        # Precision = TP / (TP + FP)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        per_class_precision.append(prec)
        
        # Recall = TP / (TP + FN)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        per_class_recall.append(rec)
        
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        per_class_f1.append(f1)
    
    # Print per-class metrics
    print("\nPer-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)
    for i, clase in enumerate(clases):
        print(f"{str(clase):<10} {per_class_accuracy[i]:<10.4f} {per_class_precision[i]:<10.4f} "
              f"{per_class_recall[i]:<10.4f} {per_class_f1[i]:<10.4f}")
    print("-" * 60)
    
    # =============================================================================
    # ARCHIVO 1: Matriz de Confusión
    # =============================================================================
    fig1, ax1 = plt.subplots(figsize=(9, 7))
    
    ax1.set_facecolor('white')
    im1 = ax1.imshow(cm, cmap='Blues', interpolation='nearest')
    ax1.set_xlabel('Predicted', fontsize=14, fontweight='normal')
    ax1.set_ylabel('Actual', fontsize=14, fontweight='normal')
    ax1.set_xticks(range(len(clases)))
    ax1.set_yticks(range(len(clases)))
    ax1.set_xticklabels(clases)
    ax1.set_yticklabels(clases)
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax1.spines[spine].set_color('black')
        ax1.spines[spine].set_linewidth(0.8)
    ax1.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    ax1.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    # Add text in each cell
    for i in range(len(clases)):
        for j in range(len(clases)):
            ax1.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=14, fontweight='normal')
    
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    plt.tight_layout()
    
    # Save confusion matrix separately
    filename_cm = f'confusion_matrix_{modelo_nombre.replace(" ", "_").lower()}_mar_menor.pdf'
    plt.savefig(f'{path_output}/{filename_cm}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved as '{filename_cm}'")
    
    # =============================================================================
    # ARCHIVO 2: Métricas por Clase (Grouped Bar Chart)
    # =============================================================================
    fig2, ax2 = plt.subplots(figsize=(11, 7))
    
    ax2.set_facecolor('white')
    
    # Set up bar positions
    x = np.arange(len(clases))
    width = 0.2  # Width of each bar
    
    # Create bars for each metric
    bars_acc = ax2.bar(x - 1.5*width, per_class_accuracy, width, 
                       label='Accuracy', color='#4DAF4A', edgecolor='black', linewidth=0.8)
    bars_prec = ax2.bar(x - 0.5*width, per_class_precision, width, 
                        label='Precision', color='#377EB8', edgecolor='black', linewidth=0.8)
    bars_rec = ax2.bar(x + 0.5*width, per_class_recall, width, 
                       label='Recall', color='#FF7F00', edgecolor='black', linewidth=0.8)
    bars_f1 = ax2.bar(x + 1.5*width, per_class_f1, width, 
                      label='F1-Score', color='#E41A1C', edgecolor='black', linewidth=0.8)
    
    ax2.set_title(f'Per-Class Metrics - {modelo_nombre}', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Class', fontsize=14, fontweight='normal')
    ax2.set_ylabel('Score', fontsize=14, fontweight='normal')
    ax2.set_xticks(x)
    ax2.set_xticklabels(clases)
    ax2.set_ylim([0, 1.15])
    ax2.grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    ax2.set_axisbelow(True)
    ax2.legend(loc='lower right', frameon=True, facecolor='#f0f0f0', edgecolor='black', fontsize=12)
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax2.spines[spine].set_color('black')
        ax2.spines[spine].set_linewidth(0.8)
    ax2.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    ax2.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    # Add value labels on bars
    y_max = max(max(per_class_accuracy), max(per_class_precision), 
                max(per_class_recall), max(per_class_f1))
    
    for bars in [bars_acc, bars_prec, bars_rec, bars_f1]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (y_max * 0.02),
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    plt.suptitle(titulo, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save metrics chart separately
    filename_metrics = f'perclass_metrics_{modelo_nombre.replace(" ", "_").lower()}_mar_menor.pdf'
    plt.savefig(f'{path_output}/{filename_metrics}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics saved as '{filename_metrics}'")
    
    return precision_global

# ====================================================================
# FUNCTIONS FOR SCIENTIFIC FIGURES (FIGURES 4, 5, 6 from original)
# ====================================================================

def crear_curvas_calibracion(modelos_clasificacion, X_test, y_test, y_classes, path_output, variable_name="Chlorophyll"):
    """
    Creates calibration curves (reliability diagrams) for classification models.
    Figure 4: Calibration curves showing predicted probability vs. observed frequency.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('white')
    
    colors = {'TabPFN': 'red', 'RandomForest_Optimized': 'blue', 'GradientBoosting_Optimized': 'green'}
    line_styles = {'TabPFN': '-', 'RandomForest_Optimized': '--', 'GradientBoosting_Optimized': ':'}
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration', alpha=0.7)
    
    for modelo_nombre, modelo_info in modelos_clasificacion.items():
        if modelo_nombre in ['TabPFN_Clas', 'RandomForest_Optimized', 'GradientBoosting_Optimized']:
            if hasattr(modelo_info['modelo'], 'predict_proba'):
                # Get predicted probabilities
                y_prob = modelo_info['modelo'].predict_proba(X_test)
                
                # For multi-class, we need to calibrate for each class or use macro-average
                # Here we'll use the probability of the true class for calibration
                n_classes = len(y_classes)
                
                # Flatten probabilities and true labels for binary calibration approach
                prob_flat = []
                true_flat = []
                
                for i in range(len(y_test)):
                    true_class = y_test[i]
                    prob_true_class = y_prob[i, true_class]
                    prob_flat.append(prob_true_class)
                    true_flat.append(1)  # True class is correct
                    
                    # Add negative examples for other classes
                    for j in range(n_classes):
                        if j != true_class:
                            prob_flat.append(y_prob[i, j])
                            true_flat.append(0)
                
                prob_flat = np.array(prob_flat)
                true_flat = np.array(true_flat)
                
                # Calculate calibration curve
                fraction_pos, mean_pred = calibration_curve(true_flat, prob_flat, n_bins=10, strategy='uniform')
                
                # Plot calibration curve
                display_name = modelo_nombre.replace('_Clas', '').replace('_Optimized', '')
                ax.plot(mean_pred, fraction_pos, marker='o', linewidth=2, markersize=8,
                       color=colors.get(display_name, 'gray'), linestyle=line_styles.get(display_name, '-'),
                       label=display_name, markeredgecolor='black', markeredgewidth=0.5)
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=14, fontweight='normal')
    ax.set_ylabel('Fraction of Positives', fontsize=14, fontweight='normal')
    ax.grid(True, color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=True, facecolor='#f0f0f0', edgecolor='black', fontsize=12)
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    ax.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    plt.tight_layout()
    
    # Save figure
    if variable_name.lower() == 'turbidity':
        filename = f'{path_output}/figure4_calibration_curves_turb.pdf'
    else:
        filename = f'{path_output}/figure4_calibration_curves_chl.pdf'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Figure 4 saved as '{filename}'")
    
    return fig

def crear_grafico_time_accuracy_tradeoff(metricas_todos_modelos, resultados_globales, path_output, variable_name="Chlorophyll"):
    """
    Creates time-accuracy tradeoff scatter plot.
    Figure 5: Computational efficiency vs. predictive accuracy tradeoff.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('white')
    
    # Prepare data
    tiempos = []
    accuracies = []
    memorias = []
    tipos = []
    nombres = []
    
    for m in metricas_todos_modelos:
        tiempos.append(m['tiempo_entrenamiento'])
        acc = m['accuracy']
        if acc is None:
            acc = 0
        accuracies.append(acc)
        memorias.append(m['memoria_usada_mb'])
        tipos.append(m.get('tipo_modelo', 'classification'))
        nombres.append(m['nombre'])
    
    # Convert to numpy arrays
    tiempos = np.array(tiempos)
    accuracies = np.array(accuracies)
    memorias = np.array(memorias)
    
    # Colors by model type (blue: classification, red: regression)
    colors = ['#377EB8' if t == 'classification' else '#E41A1C' for t in tipos]
    
    # Marker sizes based on memory footprint (scaled for better visibility)
    sizes = 50 + (memorias - memorias.min()) / (memorias.max() - memorias.min() + 1e-10) * 200
    
    # Create scatter plot
    scatter = ax.scatter(tiempos, accuracies, s=sizes, c=colors, alpha=0.7, 
                        edgecolors='black', linewidth=0.8, zorder=5)
    
    # Label key models (TabPFN, best performers)
    for i, nombre in enumerate(nombres):
        if 'TabPFN' in nombre or any(term in nombre for term in ['Optimized', 'RandomForest', 'GradientBoosting']):
            # Highlight only the most important models
            if 'TabPFN' in nombre or 'Optimized' in nombre:
                ax.annotate(nombre, (tiempos[i], accuracies[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='black', alpha=0.8))
    
    # Highlight Pareto frontier
    # Sort by time
    sorted_indices = np.argsort(tiempos)
    tiempos_sorted = tiempos[sorted_indices]
    accuracies_sorted = accuracies[sorted_indices]
    
    # Find Pareto optimal points (points that are not dominated)
    pareto_indices = []
    for i in range(len(tiempos_sorted)):
        dominated = False
        for j in range(len(tiempos_sorted)):
            if j != i:
                # Check if point j dominates point i (higher accuracy and lower or equal time)
                if accuracies_sorted[j] >= accuracies_sorted[i] and tiempos_sorted[j] <= tiempos_sorted[i]:
                    if accuracies_sorted[j] > accuracies_sorted[i] or tiempos_sorted[j] < tiempos_sorted[i]:
                        dominated = True
                        break
        if not dominated:
            pareto_indices.append(sorted_indices[i])
    
    # Plot Pareto frontier
    if len(pareto_indices) > 1:
        pareto_tiempos = [tiempos[i] for i in pareto_indices]
        pareto_acc = [accuracies[i] for i in pareto_indices]
        
        # Sort by time for line
        pareto_sorted = sorted(zip(pareto_tiempos, pareto_acc))
        pareto_tiempos_sorted = [p[0] for p in pareto_sorted]
        pareto_acc_sorted = [p[1] for p in pareto_sorted]
        
        ax.plot(pareto_tiempos_sorted, pareto_acc_sorted, 'k--', linewidth=1.5, 
                alpha=0.5, label='Pareto Frontier', zorder=1)
    
    ax.set_xlabel('Training Time (seconds, log scale)', fontsize=14, fontweight='normal')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='normal')
    ax.set_xscale('log')
    ax.set_xlim([0.01, max(tiempos) * 1.5])
    ax.set_ylim([0, 1.05])
    ax.grid(True, axis='both', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    ax.set_axisbelow(True)
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
    ax.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
    
    # Legend for model types
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#377EB8', markersize=10, 
               markeredgecolor='black', label='Classification'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E41A1C', markersize=10, 
               markeredgecolor='black', label='Regression'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, 
               markeredgecolor='black', label='Symbol size ∝ Memory', alpha=0.5)
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, 
              facecolor='#f0f0f0', edgecolor='black', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    if variable_name.lower() == 'turbidity':
        filename = f'{path_output}/figure5_time_accuracy_tradeoff_turb.pdf'
    else:
        filename = f'{path_output}/figure5_time_accuracy_tradeoff_chl.pdf'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Figure 5 saved as '{filename}'")
    
    return fig

def crear_radar_chart_top5(resultados_globales, path_output, variable_name="Chlorophyll"):
    """
    Creates radar chart comparing top 5 models across multiple metrics.
    Figure 6: Multi-dimensional performance comparison of top 5 models.
    """
    # Filter only classification models
    modelos_clasificacion = {k: v for k, v in resultados_globales.items() 
                            if v.get('tipo') == 'classification' and 'metricas' in v}
    
    if len(modelos_clasificacion) < 3:
        print("Not enough classification models for radar chart.")
        return None
    
    # Sort by accuracy and take top 5
    top_5_modelos = sorted(modelos_clasificacion.items(), 
                          key=lambda x: x[1]['accuracy'] if x[1]['accuracy'] is not None else 0, 
                          reverse=True)[:5]
    
    # Rename models for display
    renombres = {
        'TabPFN_Clas': 'TabPFN',
        'RandomForest_Optimized': 'Random Forest',
        'GradientBoosting_Optimized': 'Gradient Boosting',
        'RandomForestClassifier_Clas': 'RF Default',
        'KNeighborsClassifier_Clas': 'KNN',
        'GradientBoostingClassifier_Clas': 'GB Default',
        'SVM_Optimized': 'SVM Opt',
        'SVM_Clas': 'SVM Default'
    }
    
    # Metrics for radar chart
    metricas_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Calibration']
    n_metrics = len(metricas_radar)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.set_facecolor('white')
    
    # Color palette for top 5
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']
    
    # Store values for legend
    legend_handles = []
    
    # --- TRUNCAMIENTO DEL EJE RADIAL ---
    r_min = 0.85
    r_max = 1.0
    
    for idx, (modelo_nombre, modelo_info) in enumerate(top_5_modelos):
        # Get metrics
        m = modelo_info['metricas']
        
        # Calculate values for each metric
        acc_val = modelo_info['accuracy'] if modelo_info['accuracy'] is not None else 0
        prec_val = m.get('precision_weighted', 0) if m.get('precision_weighted') is not None else 0
        rec_val = m.get('recall_weighted', 0) if m.get('recall_weighted') is not None else 0
        f1_val = m.get('f1_weighted', 0) if m.get('f1_weighted') is not None else 0
        
        # Calibration score (1 - ECE)
        ece_val = m.get('ece', 0)
        if ece_val is None or np.isnan(ece_val):
            ece_val = 0
        cal_score = max(0, 1 - ece_val)
        
        valores = [acc_val, prec_val, rec_val, f1_val, cal_score]
        
        # Clamp values below r_min to r_min so they still appear on chart
        valores_clamped = [max(v, r_min) for v in valores]
        valores_clamped += valores_clamped[:1]  # Close the loop
        
        # Plot radar
        line = ax.plot(angles, valores_clamped, 'o-', linewidth=2, color=colors[idx], 
                      label=renombres.get(modelo_nombre, modelo_nombre))
        ax.fill(angles, valores_clamped, alpha=0.1, color=colors[idx])
        
        # Store for legend
        legend_handles.append(line[0])
        
        # Highlight the best model (first one) with a thicker line
        if idx == 0:
            ax.plot(angles, valores_clamped, 'o-', linewidth=3, color=colors[idx])
    
    # Set axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metricas_radar, fontsize=14, fontweight='normal')
    
    # --- AJUSTE DEL EJE RADIAL CON TRUNCAMIENTO ---
    ax.set_ylim([r_min, r_max])
    ax.set_yticks([0.85, 0.90, 0.95, 1.0])
    ax.set_yticklabels(['0.85', '0.90', '0.95', '1.0'], fontsize=10)
    ax.grid(True, color='black', linestyle=':', linewidth=0.3, alpha=1.0)
    
    # Customize spines
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(0.8)
    
    # Create custom legend with accuracies
    legend_labels = []
    for idx, (modelo_nombre, modelo_info) in enumerate(top_5_modelos):
        acc = modelo_info['accuracy']
        if acc is None:
            acc = 0
        legend_labels.append(f"{renombres.get(modelo_nombre, modelo_nombre)} (Acc:{acc:.3f})")
    
    ax.legend(legend_handles, legend_labels, loc='upper right', bbox_to_anchor=(1.3, 1.1), 
             frameon=True, facecolor='#f0f0f0', edgecolor='black', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    if variable_name.lower() == 'turbidity':
        filename = f'{path_output}/figure6_radar_chart_turb.pdf'
    else:
        filename = f'{path_output}/figure6_radar_chart_chl.pdf'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Figure 6 saved as '{filename}'")
    
    return fig

# ====================================================================
# MAIN CODE MODIFIED FOR MAR MENOR
# ====================================================================

# Additional exploratory analysis
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS - MAR MENOR")
print("="*50)
print("\nDescriptive statistics of key variables:")
variables_clave = ['Air_Temp_HS_Avg', 'RelHumidity_Avg', 'WS_ms_Avg', 
                   'SDI_Temp_3m', 'O2_sat2_Avg', 'SDI_TempCorrCond_3m', 
                   target_column]
variables_clave = [v for v in variables_clave if v in df.columns]
print(df[variables_clave].describe())

# Create EDA visualizations specific to Mar Menor
fig_eda = crear_visualizaciones_eda(df, target_column)

# 2. Prepare data for modeling
# Option 1: Classification (chlorophyll categories)
if 'Chl_category' in df.columns:
    y_class = df['Chl_category']
    y_class_encoded, y_classes = pd.factorize(y_class)
    print(f"\nClasses for classification: {y_classes}")
else:
    # Create categories if they don't exist
    y_class = pd.qcut(df[target_column], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    y_class_encoded, y_classes = pd.factorize(y_class)
    df['Chl_category'] = y_class

# Option 2: Regression (continuous chlorophyll values)
y_reg = df[target_column]

# Select features (exclude non-numeric or temporal columns)
exclude_cols = ['TIMESTAMP', target_column, 'Chl_category', 'hour', 'day', 'month', 'year', 'dayofyear', 'censored'] if 'censored' in df.columns else ['TIMESTAMP', target_column, 'Chl_category', 'hour', 'day', 'month', 'year', 'dayofyear']
feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

print(f"\nSelected features ({len(feature_cols)}):")
print(feature_cols)

X = df[feature_cols]

# Check class balance for classification
print(f"\nClass balance for classification:")
class_counts = pd.Series(y_class_encoded).value_counts().sort_index()
for i, count in enumerate(class_counts):
    print(f"  Class {y_classes[i] if i < len(y_classes) else i}: {count} samples ({count/len(y_class_encoded)*100:.1f}%)")

# If classes are highly imbalanced, consider oversampling/undersampling
if class_counts.min() / class_counts.max() < 0.1:
    print("\nWARNING: Highly imbalanced classes. Consider using balancing techniques.")

# 3. Split into training (80%), validation (10%) and test (10%)
# For classification
X_temp_class, X_test_class, y_temp_class, y_test_class = train_test_split(
    X, y_class_encoded, 
    test_size=0.10,
    random_state=42,
    stratify=y_class_encoded
)

X_train_class, X_val_class, y_train_class, y_val_class = train_test_split(
    X_temp_class, y_temp_class, 
    test_size=0.1111,
    random_state=42, 
    stratify=y_temp_class
)

# For regression
X_temp_reg, X_test_reg, y_temp_reg, y_test_reg = train_test_split(
    X, y_reg, 
    test_size=0.10,
    random_state=42
)

X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(
    X_temp_reg, y_temp_reg, 
    test_size=0.1111,
    random_state=42
)

# Size verification
print(f"\nDataset sizes (80/10/10):")
print(f"CLASSIFICATION:")
print(f"  Training: {X_train_class.shape[0]} samples")
print(f"  Validation: {X_val_class.shape[0]} samples")
print(f"  Test: {X_test_class.shape[0]} samples")
print(f"\nREGRESSION:")
print(f"  Training: {X_train_reg.shape[0]} samples")
print(f"  Validation: {X_val_reg.shape[0]} samples")
print(f"  Test: {X_test_reg.shape[0]} samples")

# 4. Scale features
scaler_class = StandardScaler()
X_train_scaled_class = scaler_class.fit_transform(X_train_class)
X_val_scaled_class = scaler_class.transform(X_val_class)
X_test_scaled_class = scaler_class.transform(X_test_class)

scaler_reg = StandardScaler()
X_train_scaled_reg = scaler_reg.fit_transform(X_train_reg)
X_val_scaled_reg = scaler_reg.transform(X_val_reg)
X_test_scaled_reg = scaler_reg.transform(X_test_reg)

# 5. Convert to numpy
X_train_scaled_class = np.array(X_train_scaled_class, dtype=np.float32)
X_val_scaled_class = np.array(X_val_scaled_class, dtype=np.float32)
X_test_scaled_class = np.array(X_test_scaled_class, dtype=np.float32)
y_train_class = np.array(y_train_class)
y_val_class = np.array(y_val_class)
y_test_class = np.array(y_test_class)

X_train_scaled_reg = np.array(X_train_scaled_reg, dtype=np.float32)
X_val_scaled_reg = np.array(X_val_scaled_reg, dtype=np.float32)
X_test_scaled_reg = np.array(X_test_scaled_reg, dtype=np.float32)
y_train_reg = np.array(y_train_reg)
y_val_reg = np.array(y_val_reg)
y_test_reg = np.array(y_test_reg)

# 6. Functions to save and load models
def guardar_modelo(modelo, scaler, nombre_archivo='modelo_mar_menor.pkl'):
    """Saves model and scaler"""
    modelo_data = {
        'modelo': modelo,
        'scaler': scaler,
        'columnas': feature_cols,
        'target_column': target_column,
        'y_classes': y_classes if 'y_classes' in locals() else None
    }
    
    with open(f'{path_output}/{nombre_archivo}', 'wb') as f:
        pickle.dump(modelo_data, f)
    print(f"\nModel saved to: {nombre_archivo}")

def cargar_modelo(nombre_archivo='modelo_mar_menor.pkl'):
    """Loads model and scaler"""
    with open(f'{path_output}/{nombre_archivo}', 'rb') as f:
        modelo_data = pickle.load(f)
    return modelo_data

# Dictionaries to store results
resultados_globales = {}
metricas_todos_modelos = []

# 7. REGRESSION FUNCTION WITH METRICS
def entrenar_modelo_regresion(X_train, y_train, X_val, y_val, X_test, y_test, metricas_globales, resultados_globales):
    """Trains and evaluates regression models with computational metrics"""
    print("\n" + "="*50)
    print("REGRESSION MODELS - CHLOROPHYLL (Continuous Values)")
    print("="*50)
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge, LinearRegression
        from sklearn.svm import SVR
        
        modelos_reg = {
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'RidgeRegression': Ridge(alpha=1.0, random_state=42),
            'LinearRegression': LinearRegression(),
            'SVR': SVR(kernel='rbf', C=1.0)
        }
        
        resultados_reg = {}
        
        for nombre, modelo in modelos_reg.items():
            print(f"\n--- {nombre} ---")
            
            # Get computational metrics - SPECIFY THAT IT'S REGRESSION
            metricas, y_test_pred = obtener_metricas_computacionales(
                modelo, X_train, y_train, X_test, y_test, f"{nombre}_Reg", tipo_modelo='regression'
            )
            
            # Regression metrics are already calculated in the function
            mse_test = metricas['mse']
            mae_test = metricas['mae']
            r2_test = metricas['r2_score']
            acc_test = metricas['accuracy']  # Accuracy of rounded value
            
            # Store results
            resultados_reg[nombre] = {
                'modelo': modelo,
                'mse_test': mse_test,
                'mae_test': mae_test,
                'r2_test': r2_test,
                'acc_test': acc_test,
                'y_test_pred': y_test_pred,
                'y_test_pred_class': metricas['y_pred_redondeado'],  # Rounded values
                'metricas': metricas
            }
            
            # Add to global metrics
            metricas_globales.append(metricas)
            
            # Add to global results
            resultados_globales[f"{nombre}_Reg"] = {
                'accuracy': acc_test,
                'y_pred': metricas['y_pred_redondeado'],  # Use rounded values
                'modelo': modelo,
                'metricas': metricas,
                'r2_score': r2_test,
                'tipo': 'regression'
            }
            
            print(f"  • R²: {r2_test:.4f}")
            print(f"  • MSE: {mse_test:.4f}")
            print(f"  • MAE: {mae_test:.4f}")
            print(f"  • Accuracy (rounded): {acc_test:.4f}")
            print(f"  • Training Time: {metricas['tiempo_entrenamiento']:.2f}s")
            
            # Add scatter plot for this regression model
            crear_scatter_regresion(
                y_test, 
                y_test_pred, 
                nombre,
                f"Chlorophyll (µg/L)"
            )
        
        # Select best regression model (by R²)
        mejor_nombre = max(resultados_reg.keys(), key=lambda x: resultados_reg[x]['r2_test'])
        
        print(f"\n{'='*50}")
        print(f"BEST REGRESSION MODEL: {mejor_nombre}")
        print(f"R²: {resultados_reg[mejor_nombre]['r2_test']:.4f}")
        print(f"Accuracy (rounded): {resultados_reg[mejor_nombre]['acc_test']:.4f}")
        
        # Show confusion matrix for best model (with rounded values)
        mejor_modelo = resultados_reg[mejor_nombre]
        mostrar_matriz_confusion(y_test, mejor_modelo['y_test_pred_class'], 
                               f"Confusion Matrix - {mejor_nombre} (Rounded Regression)",
                               mejor_nombre)
        
        return resultados_reg, metricas_globales
        
    except Exception as e:
        print(f"Error in regression: {e}")
        import traceback
        traceback.print_exc()
        return None, metricas_globales
    
# 8. CLASSIFICATION FUNCTION WITH METRICS
def entrenar_modelo_clasificacion(X_train, y_train, X_val, y_val, X_test, y_test, metricas_globales, resultados_globales):
    """Trains and evaluates classification models with computational metrics"""
    print("\n" + "="*50)
    print("CLASSIFICATION MODELS - CHLOROPHYLL CATEGORIES")
    print("="*50)
    
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        # Classification models with default parameters
        modelos_clas = {
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True),
            'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5, weights='distance', p=1)
        }
        
        resultados_clas = {}
        
        for nombre, modelo in modelos_clas.items():
            print(f"\n--- {nombre} ---")
            
            # Get computational metrics - SPECIFY THAT IT'S CLASSIFICATION
            metricas, y_test_pred = obtener_metricas_computacionales(
                modelo, X_train, y_train, X_test, y_test, f"{nombre}_Clas", tipo_modelo='classification'
            )
            
            # Store results
            resultados_clas[nombre] = {
                'modelo': modelo,
                'acc_test': metricas['accuracy'],
                'f1_test': metricas['f1_weighted'],
                'y_test_pred': y_test_pred,
                'metricas': metricas
            }
            
            # Add to global metrics
            metricas_globales.append(metricas)
            
            # Add to global results
            resultados_globales[f"{nombre}_Clas"] = {
                'accuracy': metricas['accuracy'],
                'y_pred': y_test_pred,
                'modelo': modelo,
                'metricas': metricas,
                'tipo': 'classification'
            }
            
            print(f"  • Accuracy: {metricas['accuracy']:.4f}")
            print(f"  • F1-Score (Weighted): {metricas['f1_weighted']:.4f}")
            print(f"  • NLL: {metricas.get('nll', 0):.4f}")
            print(f"  • ECE: {metricas.get('ece', 0):.4f}")
            print(f"  • Training Time: {metricas['tiempo_entrenamiento']:.2f}s")
            
            # Show confusion matrix
            mostrar_matriz_confusion(y_test, y_test_pred, 
                                   f"Confusion Matrix - {nombre}",
                                   nombre)
        
        # Select best classification model
        mejor_nombre = max(resultados_clas.keys(), key=lambda x: resultados_clas[x]['acc_test'])
        
        print(f"\n{'='*50}")
        print(f"BEST CLASSIFICATION MODEL: {mejor_nombre}")
        print(f"Accuracy: {resultados_clas[mejor_nombre]['acc_test']:.4f}")
        print(f"F1-Score: {resultados_clas[mejor_nombre]['f1_test']:.4f}")
        
        return resultados_clas, metricas_globales
        
    except Exception as e:
        print(f"Error in classification: {e}")
        import traceback
        traceback.print_exc()
        return None, metricas_globales

# 9. PRECISION IMPROVEMENT FUNCTION WITH METRICS
def aplicar_mejoras_precision(X_train, y_train, X_val, y_val, X_test, y_test, metricas_globales, resultados_globales, tipo='classification'):
    """Applies techniques to improve precision with metrics"""
    print("\n" + "="*50)
    print(f"APPLYING PRECISION IMPROVEMENTS ({tipo.upper()})")
    print("="*50)
    
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
        from sklearn.svm import SVC, SVR
        from imblearn.over_sampling import SMOTE
        
        if tipo == 'classification':
            # Class balancing with SMOTE for classification
            print("\n1. Applying SMOTE for class balancing...")
            try:
                smote = SMOTE(random_state=42)
                X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
                
                # Visualize balancing
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Before
                ax1.set_facecolor('white')
                counts_before = pd.Series(y_train).value_counts()
                bars_before = ax1.bar(range(len(counts_before)), counts_before.values, 
                                     color='skyblue', edgecolor='black', linewidth=0.8, width=0.7)
                ax1.set_title('Distribution Before SMOTE', fontsize=13)
                ax1.set_xlabel('Category', fontsize=11)
                ax1.set_ylabel('Frequency', fontsize=11)
                ax1.set_xticks(range(len(counts_before)))
                ax1.set_xticklabels([f'Cat {i}' for i in range(len(counts_before))])
                ax1.grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
                ax1.set_axisbelow(True)
                for spine in ['top', 'bottom', 'left', 'right']:
                    ax1.spines[spine].set_color('black')
                    ax1.spines[spine].set_linewidth(0.8)
                ax1.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
                ax1.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
                
                # After
                ax2.set_facecolor('white')
                counts_after = pd.Series(y_train_bal).value_counts()
                bars_after = ax2.bar(range(len(counts_after)), counts_after.values, 
                                    color='lightgreen', edgecolor='black', linewidth=0.8, width=0.7)
                ax2.set_title('Distribution After SMOTE', fontsize=13)
                ax2.set_xlabel('Category', fontsize=11)
                ax2.set_ylabel('Frequency', fontsize=11)
                ax2.set_xticks(range(len(counts_after)))
                ax2.set_xticklabels([f'Cat {i}' for i in range(len(counts_after))])
                ax2.grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
                ax2.set_axisbelow(True)
                for spine in ['top', 'bottom', 'left', 'right']:
                    ax2.spines[spine].set_color('black')
                    ax2.spines[spine].set_linewidth(0.8)
                ax2.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
                ax2.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
                
                plt.suptitle('Class Balancing with SMOTE', fontsize=15, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'{path_output}/smote_balancing_mar_menor.pdf', dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"   SMOTE not available: {e}")
                X_train_bal, y_train_bal = X_train, y_train
            
            # Train improved classification models
            modelos = {
                'RandomForest_Optimized': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
                'GradientBoosting_Optimized': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=10, random_state=42),
                'SVM_Optimized': SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42, probability=True)
            }
            
        else:  # Regression
            X_train_bal, y_train_bal = X_train, y_train
            
            # Train improved regression models
            modelos = {
                'RandomForestReg_Optimized': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
                'SVR_Optimized': SVR(kernel='rbf', C=2.0, gamma='scale')
            }
        
        mejores_resultados = {}
        
        for nombre, modelo in modelos.items():
            print(f"\n   Training {nombre}...")
            
            # Get computational metrics
            metricas, y_test_pred = obtener_metricas_computacionales(
                modelo, X_train_bal, y_train_bal, X_test, y_test, f"{nombre}",
                tipo_modelo=tipo
            )
            
            # Store in global list
            metricas_globales.append(metricas)
           
            # Store in global results
            resultados_globales[f"{nombre}"] = {
                'accuracy': metricas['accuracy'],
                'y_pred': y_test_pred,
                'modelo': modelo,
                'metricas': metricas,
                **({'r2_score': metricas['r2_score']} if tipo == 'regression' else {}),
                'tipo': tipo
            }
            
            print(f"   Test accuracy: {metricas['accuracy']:.4f}")
            if tipo == 'classification':
                print(f"   NLL: {metricas.get('nll', 0):.4f}")
                print(f"   ECE: {metricas.get('ece', 0):.4f}")
            print(f"   Training time: {metricas['tiempo_entrenamiento']:.2f}s")
            
            if tipo == 'classification':
                mostrar_matriz_confusion(y_test, y_test_pred, 
                                       f"Confusion Matrix - {nombre} Optimized",
                                       f"{nombre}")
            
            # Add scatter plot for improved regression models
            if tipo == 'regression':
                crear_scatter_regresion(
                    y_test,
                    y_test_pred,
                    f"{nombre}_Improved",
                    "Chlorophyll (µg/L)"
                )
        
        # Comparative plot of improvements
        model_names = [f"{m}" for m in modelos.keys()]
        accuracies = []
        for n in model_names:
            acc = resultados_globales[n]['accuracy']
            if acc is None:
                acc = 0
            accuracies.append(acc)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor('white')
        bars = ax.bar(model_names, accuracies, color=['#E41A1C', '#377EB8', '#4DAF4A'][:len(model_names)], 
                     edgecolor='black', linewidth=0.8, width=0.7)
        ax.set_title(f'Comparison of Improved Models ({tipo})', fontsize=16, fontweight='bold')
        ax.set_xlabel('Model', fontsize=14, fontweight='normal')
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='normal')
        ax.set_ylim([0, 1])
        ax.grid(True, axis='y', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
        ax.set_axisbelow(True)
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_color('black')
            ax.spines[spine].set_linewidth(0.8)
        ax.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
        ax.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)
        
        y_max = max(accuracies)
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (y_max * 0.02),
                   f'{acc:.4f}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'{path_output}/improvement_comparison_{tipo}.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Select best model
        mejor_nombre = max(model_names, key=lambda x: resultados_globales[x]['accuracy'] if resultados_globales[x]['accuracy'] is not None else 0)
        mejor_resultado = resultados_globales[mejor_nombre]
        
        print(f"\n{'='*50}")
        print(f"BEST IMPROVED MODEL ({tipo}): {mejor_nombre}")
        print(f"Test accuracy: {mejor_resultado['accuracy']:.4f}")
        
        return mejores_resultados, metricas_globales
        
    except Exception as e:
        print(f"Error in precision improvements: {e}")
        import traceback
        traceback.print_exc()
        return None, metricas_globales

# 10. MAIN TRAINING WITH TABPFN (CLASSIFICATION AND REGRESSION)
print("\n" + "="*80)
print("MAIN TRAINING - EVALUATION WITH COMPUTATIONAL METRICS")
print("="*80)

# First, classification models
print("\n" + "="*50)
print("CLASSIFICATION MODELS")
print("="*50)

resultados_clas, metricas_todos_modelos = entrenar_modelo_clasificacion(
    X_train_scaled_class, y_train_class,
    X_val_scaled_class, y_val_class,
    X_test_scaled_class, y_test_class,
    metricas_todos_modelos,
    resultados_globales
)

# Second, regression models
print("\n" + "="*50)
print("REGRESSION MODELS")
print("="*50)

resultados_reg, metricas_todos_modelos = entrenar_modelo_regresion(
    X_train_scaled_reg, y_train_reg,
    X_val_scaled_reg, y_val_reg,
    X_test_scaled_reg, y_test_reg,
    metricas_todos_modelos,
    resultados_globales
)

# Third, try TabPFN for classification
try:
    from tabpfn import TabPFNClassifier
    
    print("\n" + "="*50)
    print("ADVANCED MODEL - TabPFNClassifier")
    print("="*50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device used: {device}")
    
    # Configure TabPFN
    model_tabpfn = TabPFNClassifier(
        device=device,
        n_estimators=4,
        fit_mode='batched'
    )
    
    # Get computational metrics for TabPFN
    print("\nTraining TabPFN and calculating metrics...")
    metricas_tabpfn, y_test_pred_tabpfn = obtener_metricas_computacionales(
        model_tabpfn, 
        X_train_scaled_class, y_train_class, 
        X_test_scaled_class, y_test_class,
        'TabPFN_Clas',
        tipo_modelo='classification'
    )
    
    # Store metrics
    metricas_todos_modelos.append(metricas_tabpfn)
    resultados_globales['TabPFN_Clas'] = {
        'accuracy': metricas_tabpfn['accuracy'],
        'y_pred': y_test_pred_tabpfn,
        'modelo': model_tabpfn,
        'metricas': metricas_tabpfn,
        'tipo': 'classification'
    }
    
    print(f"\n✓ TabPFN metrics obtained:")
    print(f"  • Accuracy: {metricas_tabpfn['accuracy']:.4f}")
    print(f"  • F1 Macro: {metricas_tabpfn['f1_macro']:.4f}")
    print(f"  • F1 Weighted: {metricas_tabpfn['f1_weighted']:.4f}")
    print(f"  • NLL: {metricas_tabpfn.get('nll', 0):.4f}")
    print(f"  • ECE: {metricas_tabpfn.get('ece', 0):.4f}")
    print(f"  • Training Time: {metricas_tabpfn['tiempo_entrenamiento']:.2f}s")
    print(f"  • Prediction Time: {metricas_tabpfn['tiempo_prediccion']:.4f}s")
    print(f"  • Memory Used: {metricas_tabpfn['memoria_usada_mb']:.0f}MB")
    
    # Confusion matrix for TabPFN
    mostrar_matriz_confusion(y_test_class, y_test_pred_tabpfn, 
                           "TabPFN - Confusion Matrix",
                           "TabPFN")
    
    # Classification report
    print("\n" + "="*50)
    print("DETAILED CLASSIFICATION REPORT - TabPFN")
    print("="*50)
    reporte = classification_report(y_test_class, y_test_pred_tabpfn, zero_division=0,
                                   target_names=[str(c) for c in y_classes])
    print(reporte)
    
    # Save TabPFN model
    guardar_modelo(model_tabpfn, scaler_class, 'tabpfn_model_mar_menor.pkl')
    
except Exception as e:
    print(f"\n✗ TabPFNClassifier not available or error: {e}")
    print("Continuing with other models...")

# Fourth, try TabPFN for regression
try:
    from tabpfn import TabPFNRegressor
    
    print("\n" + "="*50)
    print("ADVANCED MODEL - TabPFNRegressor")
    print("="*50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device used: {device}")
    
    # Configure TabPFN for regression
    model_tabpfn_reg = TabPFNRegressor(
        device=device,
        n_estimators=4,
        fit_mode='batched'
    )
    
    # Get computational metrics for TabPFN Regressor
    print("\nTraining TabPFN Regressor and calculating metrics...")
    metricas_tabpfn_reg, y_test_pred_tabpfn_reg = obtener_metricas_computacionales(
        model_tabpfn_reg, 
        X_train_scaled_reg, y_train_reg, 
        X_test_scaled_reg, y_test_reg,
        'TabPFN_Reg',
        tipo_modelo='regression'
    )
    
    # Store metrics
    metricas_todos_modelos.append(metricas_tabpfn_reg)
    resultados_globales['TabPFN_Reg'] = {
        'accuracy': metricas_tabpfn_reg['accuracy'],
        'y_pred': metricas_tabpfn_reg['y_pred_redondeado'],
        'modelo': model_tabpfn_reg,
        'metricas': metricas_tabpfn_reg,
        'r2_score': metricas_tabpfn_reg['r2_score'],
        'tipo': 'regression'
    }
    
    print(f"\n✓ TabPFN Regressor metrics obtained:")
    print(f"  • R²: {metricas_tabpfn_reg['r2_score']:.4f}")
    print(f"  • MSE: {metricas_tabpfn_reg['mse']:.4f}")
    print(f"  • MAE: {metricas_tabpfn_reg['mae']:.4f}")
    print(f"  • Accuracy (rounded): {metricas_tabpfn_reg['accuracy']:.4f}")
    print(f"  • Training Time: {metricas_tabpfn_reg['tiempo_entrenamiento']:.2f}s")
    print(f"  • Prediction Time: {metricas_tabpfn_reg['tiempo_prediccion']:.4f}s")
    print(f"  • Memory Used: {metricas_tabpfn_reg['memoria_usada_mb']:.0f}MB")
    
    # Confusion matrix for TabPFN Regressor (rounded values)
    mostrar_matriz_confusion(y_test_reg, metricas_tabpfn_reg['y_pred_redondeado'], 
                           "TabPFN Regressor - Confusion Matrix (Rounded Values)",
                           "TabPFN_Reg")
    
    # Add scatter plot for TabPFN Regressor
    crear_scatter_regresion(
        y_test_reg, 
        y_test_pred_tabpfn_reg, 
        "TabPFN_Reg",
        "Chlorophyll (µg/L)"
    )
    
    # Save TabPFN Regressor model
    guardar_modelo(model_tabpfn_reg, scaler_reg, 'tabpfn_reg_model_mar_menor.pkl')
    
except Exception as e:
    print(f"\n✗ TabPFNRegressor not available or error: {e}")
    print("Continuing with other models...")

# 11. APPLY PRECISION IMPROVEMENTS FOR BOTH TYPES
print("\n" + "="*50)
print("PRECISION IMPROVEMENTS FOR CLASSIFICATION")
print("="*50)

mejores_resultados_clas, metricas_todos_modelos = aplicar_mejoras_precision(
    X_train_scaled_class, y_train_class,
    X_val_scaled_class, y_val_class,
    X_test_scaled_class, y_test_class,
    metricas_todos_modelos,
    resultados_globales,
    tipo='classification'
)

print("\n" + "="*50)
print("PRECISION IMPROVEMENTS FOR REGRESSION")
print("="*50)

mejores_resultados_reg, metricas_todos_modelos = aplicar_mejoras_precision(
    X_train_scaled_reg, y_train_reg,
    X_val_scaled_reg, y_val_reg,
    X_test_scaled_reg, y_test_reg,
    metricas_todos_modelos,
    resultados_globales,
    tipo='regression'
)

# 12. VISUALIZE COMPUTATIONAL METRICS
print("\n" + "="*80)
print("COMPUTATIONAL METRICS VISUALIZATION")
print("="*80)

# Create visualizations
visualizar_metricas_computacionales(metricas_todos_modelos)

# Create summary table
df_resumen = crear_tabla_resumen_metricas(metricas_todos_modelos)

# ====================================================================
# NEW FIGURES: Learning Curves, Temporal Decomposition, Feature Importance
# ====================================================================
print("\n" + "="*80)
print("GENERATING ADDITIONAL SCIENTIFIC FIGURES")
print("="*80)

# Figure 3b: Individual Learning Curves for each classification model
print("\n" + "="*80)
print("GENERATING INDIVIDUAL LEARNING CURVES FOR EACH MODEL")
print("="*80)

# Collect classification models for learning curves
learning_curves_models = {}
for model_name, model_info in resultados_globales.items():
    if model_info.get('tipo') == 'classification' and 'modelo' in model_info:
        # Include all classification models for individual curves
        learning_curves_models[model_name] = model_info['modelo']

if len(learning_curves_models) > 0:
    print(f"  Generating individual learning curves for {len(learning_curves_models)} models:")
    for name in learning_curves_models.keys():
        print(f"    - {name}")
    
    # Generate individual learning curves for each model
    learning_curves_results = crear_learning_curves_all_models(
        learning_curves_models,
        X_train_scaled_class,
        y_train_class,
        X_val_scaled_class,
        y_val_class,
        path_output,
        variable_name="Chlorophyll"
    )
    
    print(f"\n  Individual learning curves generated for {len(learning_curves_results)} models")
else:
    print("  No classification models found for learning curves.")

# Figure 3b (additional): Individual Learning Curves for regression models
print("\n" + "="*80)
print("GENERATING INDIVIDUAL LEARNING CURVES FOR REGRESSION MODELS")
print("="*80)

# Collect regression models for learning curves
regression_curves_models = {}
for model_name, model_info in resultados_globales.items():
    if model_info.get('tipo') == 'regression' and 'modelo' in model_info:
        # Include all regression models for individual curves
        regression_curves_models[model_name] = model_info['modelo']

if len(regression_curves_models) > 0:
    print(f"  Generating individual learning curves for {len(regression_curves_models)} regression models:")
    for name in regression_curves_models.keys():
        print(f"    - {name}")
    
    # Generate individual learning curves for each regression model
    regression_curves_results = crear_learning_curves_all_models_regression(
        regression_curves_models,
        X_train_scaled_reg,
        y_train_reg,
        X_val_scaled_reg,
        y_val_reg,
        path_output,
        variable_name="Chlorophyll"
    )
    
    print(f"\n  Individual regression learning curves generated for {len(regression_curves_results)} models")
else:
    print("  No regression models found for learning curves.")

# Figure 4c: Temporal Decomposition
print("\nGenerating Figure 4c: Temporal decomposition...")
fig_td, daily_avg, monthly_avg, yearly_avg = crear_temporal_decomposition(
    df, target_column, path_output, variable_name="Chlorophyll"
)

# Figure 5: Feature Importance (with TabPFN priority)
print("\n" + "="*80)
print("GENERATING FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Try to get feature importance for TabPFN first (since it's the best model)
tabpfn_importance_generated = False

if 'TabPFN_Clas' in resultados_globales:
    print("\nGenerating Figure 5: Feature importance using TabPFN (Permutation Importance)...")
    tabpfn_model = resultados_globales['TabPFN_Clas']['modelo']
    
    try:
        fig_fi_tabpfn, feature_importance_df_tabpfn = crear_feature_importance_tabpfn(
            tabpfn_model,
            X_train_scaled_class,
            y_train_class,
            feature_cols,
            path_output,
            "TabPFN",
            variable_name="Chlorophyll",
            n_repeats=5,
            n_samples=1000
        )
        tabpfn_importance_generated = True
        print("  ✓ TabPFN feature importance generated successfully")
    except Exception as e:
        print(f"  ✗ Error generating TabPFN feature importance: {e}")
        print("  Falling back to tree-based model...")

# If TabPFN failed, use tree-based model
if not tabpfn_importance_generated:
    print("\nGenerating Figure 5: Feature importance using tree-based model...")
    
    # Find the best tree-based model for feature importance
    best_tree_model_name = None
    best_tree_model = None
    best_accuracy = 0
    
    # First, look for optimized tree-based models
    for model_name in ['RandomForest_Optimized', 'GradientBoosting_Optimized', 
                       'RandomForestClassifier_Clas', 'GradientBoostingClassifier_Clas']:
        if model_name in resultados_globales:
            model_info = resultados_globales[model_name]
            model = model_info['modelo']
            if hasattr(model, 'feature_importances_'):
                best_tree_model_name = model_name
                best_tree_model = model
                best_accuracy = model_info.get('accuracy', 0)
                break
    
    # If no optimized model found, search through all models
    if best_tree_model_name is None:
        for model_name, model_info in resultados_globales.items():
            if model_info.get('tipo') == 'classification':
                model = model_info['modelo']
                if hasattr(model, 'feature_importances_'):
                    accuracy = model_info.get('accuracy', 0)
                    if accuracy is None:
                        accuracy = 0
                    if best_tree_model_name is None or accuracy > best_accuracy:
                        best_tree_model_name = model_name
                        best_tree_model = model
                        best_accuracy = accuracy
    
    if best_tree_model_name and best_tree_model:
        print(f"  Using {best_tree_model_name} for feature importance (accuracy: {best_accuracy:.4f})")
        fig_fi, feature_importance_df = crear_feature_importance(
            best_tree_model,
            X_train_scaled_class,
            y_train_class,
            feature_cols,
            path_output,
            best_tree_model_name,
            variable_name="Chlorophyll"
        )
    else:
        print("  No model with feature importance found. Using RandomForestClassifier as fallback...")
        from sklearn.ensemble import RandomForestClassifier
        fallback_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        fallback_rf.fit(X_train_scaled_class, y_train_class)
        fig_fi, feature_importance_df = crear_feature_importance(
            fallback_rf,
            X_train_scaled_class,
            y_train_class,
            feature_cols,
            path_output,
            "RandomForestClassifier (Fallback)",
            variable_name="Chlorophyll"
        )

# 13. TIME-ACCURACY TRADEOFF ANALYSIS
print("\n" + "="*80)
print("TIME-ACCURACY TRADEOFF ANALYSIS")
print("="*80)

fig_compromiso = plt.figure(figsize=(12, 8))
ax = fig_compromiso.add_subplot(111)

# Scatter plot: time vs accuracy
tiempos = [m['tiempo_entrenamiento'] for m in metricas_todos_modelos]
accuracies = []
for m in metricas_todos_modelos:
    acc = m['accuracy']
    if acc is None:
        acc = 0
    accuracies.append(acc)
nombres = [m['nombre'] for m in metricas_todos_modelos]
tipos = [m.get('tipo_modelo', 'classification') for m in metricas_todos_modelos]

# Colors by model type
colors = ['#377EB8' if t == 'classification' else '#E41A1C' for t in tipos]
sizes = [100 if 'Optimized' in n or 'TabPFN' in n else 60 for n in nombres]

ax.set_facecolor('white')
scatter = ax.scatter(tiempos, accuracies, s=sizes, alpha=0.7, 
                     c=colors, edgecolors='black', linewidth=0.8)

# Label each point
for i, (tiempo, acc, nombre) in enumerate(zip(tiempos, accuracies, nombres)):
    ax.annotate(nombre, (tiempo, acc), xytext=(5, 5), 
                textcoords='offset points', fontsize=8)

ax.set_xlabel('Training Time (seconds)', fontsize=14, fontweight='normal')
ax.set_ylabel('Accuracy', fontsize=14, fontweight='normal')
ax.set_title('Time-Accuracy Tradeoff by Model - Mar Menor', fontsize=16, fontweight='bold')
ax.grid(True, axis='both', color='black', linestyle=':', linewidth=0.3, alpha=1.0)
ax.set_axisbelow(True)
for spine in ['top', 'bottom', 'left', 'right']:
    ax.spines[spine].set_color('black')
    ax.spines[spine].set_linewidth(0.8)
ax.tick_params(axis='x', length=4, width=0.8, color='black', labelsize=10)
ax.tick_params(axis='y', length=4, width=0.8, color='black', labelsize=10)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#377EB8', markersize=10, markeredgecolor='black', label='Classification'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E41A1C', markersize=10, markeredgecolor='black', label='Regression')
]
ax.legend(handles=legend_elements, loc='upper left', frameon=True, edgecolor='black')

plt.tight_layout()
plt.savefig(f'{path_output}/time_accuracy_tradeoff_mar_menor.pdf', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ Tradeoff analysis saved to 'time_accuracy_tradeoff_mar_menor.pdf'")

# 14. FINAL MODEL COMPARISON
print("\n" + "="*80)
print("FINAL COMPARISON OF ALL MODELS")
print("="*80)

# Create comparative plot
crear_grafico_comparativo_modelos(resultados_globales)

# Create separate comparative plots for classification and regression
print("\nGenerating classification models comparison...")
crear_grafico_comparativo_modelos_clasificacion(resultados_globales)

print("\nGenerating regression models comparison...")
crear_grafico_comparativo_modelos_regresion(resultados_globales)

# ====================================================================
# GENERATE SCIENTIFIC FIGURES (FIGURES 4, 5, 6 from original)
# ====================================================================
print("\n" + "="*80)
print("GENERATING SCIENTIFIC FIGURES FOR PUBLICATION")
print("="*80)

# Figure 4: Calibration curves
print("\nGenerating Figure 4: Calibration curves...")
crear_curvas_calibracion(
    resultados_globales, 
    X_test_scaled_class, 
    y_test_class, 
    y_classes, 
    path_output, 
    variable_name="Chlorophyll"
)

# Figure 5: Time-accuracy tradeoff (improved version)
print("\nGenerating Figure 5: Time-accuracy tradeoff...")
crear_grafico_time_accuracy_tradeoff(
    metricas_todos_modelos, 
    resultados_globales, 
    path_output, 
    variable_name="Chlorophyll"
)

# Figure 6: Radar chart for top 5 models
print("\nGenerating Figure 6: Radar chart...")
crear_radar_chart_top5(
    resultados_globales, 
    path_output, 
    variable_name="Chlorophyll"
)

# Final console summary
print("\n" + "="*80)
print("FINAL EXECUTIVE SUMMARY - MAR MENOR")
print("="*80)

print("\nResults by model (ordered by accuracy):")
print("-" * 100)
print(f"{'MODEL':<30} {'TYPE':<12} {'ACCURACY':<10} {'F1 (W)':<10} {'NLL':<10} {'ECE':<10} {'TIME (s)':<12} {'MEM (MB)':<10}")
print("-" * 100)

for nombre, datos in sorted(resultados_globales.items(), 
                            key=lambda x: x[1]['accuracy'] if x[1]['accuracy'] is not None else 0, 
                            reverse=True):
    if 'metricas' in datos:
        tiempo = datos['metricas']['tiempo_entrenamiento']
        memoria = datos['metricas']['memoria_usada_mb']
        f1_w = datos['metricas']['f1_weighted']
        if f1_w is None:
            f1_w = 0
        nll = datos['metricas'].get('nll', 0)
        if nll is None:
            nll = np.nan
        ece = datos['metricas'].get('ece', 0)
        if ece is None:
            ece = np.nan
    else:
        tiempo = memoria = f1_w = nll = ece = 0
    
    tipo = datos.get('tipo', 'classification')[:10]
    
    acc_val = datos['accuracy']
    if acc_val is None:
        acc_val = 0
    
    nll_str = f"{nll:.4f}" if not np.isnan(nll) else 'N/A'
    ece_str = f"{ece:.4f}" if not np.isnan(ece) else 'N/A'
    
    print(f"{nombre:<30} {tipo:<12} {acc_val:<10.4f} {f1_w:<10.4f} {nll_str:<10} {ece_str:<10} {tiempo:<12.2f} {memoria:<10.0f}")

# Identify best model by type
mejor_clasificacion = max([(k, v) for k, v in resultados_globales.items() 
                          if v.get('tipo') == 'classification'], 
                         key=lambda x: x[1]['accuracy'] if x[1]['accuracy'] is not None else 0, default=(None, None))

mejor_regresion = max([(k, v) for k, v in resultados_globales.items() 
                      if v.get('tipo') == 'regression'], 
                     key=lambda x: x[1]['accuracy'] if x[1]['accuracy'] is not None else 0, default=(None, None))

mejor_modelo_nombre = max(resultados_globales.keys(), 
                         key=lambda x: resultados_globales[x]['accuracy'] if resultados_globales[x]['accuracy'] is not None else 0)
mejor_resultado = resultados_globales[mejor_modelo_nombre]

print("\n" + "="*80)
print("🏆 BEST MODELS BY CATEGORY")
print("="*80)
if mejor_clasificacion[0]:
    acc_val = mejor_clasificacion[1]['accuracy']
    if acc_val is None:
        acc_val = 0
    print(f"• BEST CLASSIFICATION: {mejor_clasificacion[0]}")
    print(f"  Accuracy: {acc_val:.4f}")
    if 'metricas' in mejor_clasificacion[1]:
        nll_val = mejor_clasificacion[1]['metricas'].get('nll', 0)
        if nll_val is None:
            nll_val = np.nan
        ece_val = mejor_clasificacion[1]['metricas'].get('ece', 0)
        if ece_val is None:
            ece_val = np.nan
        print(f"  NLL: {nll_val:.4f}" if not np.isnan(nll_val) else "  NLL: N/A")
        print(f"  ECE: {ece_val:.4f}" if not np.isnan(ece_val) else "  ECE: N/A")

if mejor_regresion[0]:
    acc_val = mejor_regresion[1]['accuracy']
    if acc_val is None:
        acc_val = 0
    print(f"• BEST REGRESSION: {mejor_regresion[0]}")
    print(f"  Accuracy: {acc_val:.4f}")
    if 'r2_score' in mejor_regresion[1]:
        r2_val = mejor_regresion[1]['r2_score']
        if r2_val is None:
            r2_val = 0
        print(f"  R² Score: {r2_val:.4f}")

acc_val = mejor_resultado['accuracy']
if acc_val is None:
    acc_val = 0
print(f"\n🏆 BEST OVERALL MODEL: {mejor_modelo_nombre}")
print(f"• Accuracy: {acc_val:.4f} ({acc_val*100:.1f}%)")
print(f"• Type: {mejor_resultado.get('tipo', 'classification')}")

if 'metricas' in mejor_resultado:
    f1_val = mejor_resultado['metricas']['f1_weighted']
    if f1_val is None:
        f1_val = 0
    print(f"• F1-Score (Weighted): {f1_val:.4f}")
    
    nll_val = mejor_resultado['metricas'].get('nll', 0)
    if nll_val is None:
        nll_val = np.nan
    if not np.isnan(nll_val):
        print(f"• NLL: {nll_val:.4f}")
    
    ece_val = mejor_resultado['metricas'].get('ece', 0)
    if ece_val is None:
        ece_val = np.nan
    if not np.isnan(ece_val):
        print(f"• ECE: {ece_val:.4f}")
    
    print(f"• Training Time: {mejor_resultado['metricas']['tiempo_entrenamiento']:.2f}s")
    print(f"• Prediction Time: {mejor_resultado['metricas']['tiempo_prediccion']:.3f}s")
    print(f"• Memory Used: {mejor_resultado['metricas']['memoria_usada_mb']:.0f}MB")

print("="*80)

# 15. SAVE COMPLETE RESULTS
resultados_completos = {
    'y_test_real_class': y_test_class,
    'y_test_real_reg': y_test_reg,
    'resultados_modelos': resultados_globales,
    'metricas_computacionales': metricas_todos_modelos,
    'best_model_classification': mejor_clasificacion[0],
    'best_model_regression': mejor_regresion[0],
    'best_model_overall': mejor_modelo_nombre,
    'best_accuracy': mejor_resultado['accuracy'],
    'feature_columns': feature_cols,
    'target_column': target_column,
    'y_classes': y_classes.tolist() if 'y_classes' in locals() else None,
    'execution_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'configuration': {
        'data_split': '80/10/10',
        'scaling': 'StandardScaler',
        'random_state': 42
    }
}

with open(f'{path_output}/complete_results_mar_menor.pkl', 'wb') as f:
    pickle.dump(resultados_completos, f)

print("\n" + "="*80)
print("✓ EXECUTION COMPLETED SUCCESSFULLY")
print("="*80)
print("\n📊 FILES GENERATED:")
print("  1. eda_mar_menor.pdf                      - Exploratory data analysis")
print("  2. computational_metrics_mar_menor.pdf    - Metrics comparison (includes NLL and ECE)")
print("  3. model_comparison_mar_menor.pdf         - Model comparison")
print("  4. time_accuracy_tradeoff_mar_menor.pdf   - Time-accuracy analysis")
print("  5. metrics_summary_mar_menor.csv          - Metrics table (includes NLL and ECE)")
print("  6. confusion_matrix_*_mar_menor.pdf       - Confusion matrices")
print("  7. smote_balancing_mar_menor.pdf          - SMOTE balancing")
print("  8. improvement_comparison_*.pdf           - Improved models")
print("  9. complete_results_mar_menor.pkl         - Serialized data")
print(" 10. tabpfn_model_mar_menor.pkl             - TabPFN model saved")
print(" 11. tabpfn_reg_model_mar_menor.pkl         - TabPFN Regressor model saved")
print(" 12. regression_scatter_*.pdf               - Predicted vs Actual scatter plots")
print("\n📈 SCIENTIFIC FIGURES FOR PUBLICATION:")
print(" 13. figure3b_learning_curves_chl.pdf       - Learning curves (Figure 3b) [NEW]")
print(" 14. figure4c_temporal_decomposition_chl.pdf - Temporal decomposition (Figure 4c) [NEW]")
print(" 15. figure5_feature_importance_chl.pdf     - Feature importance (Figure 5) [NEW]")
print(" 16. figure4_calibration_curves_chl.pdf     - Calibration curves (Figure 4)")
print(" 17. figure5_time_accuracy_tradeoff_chl.pdf - Time-accuracy tradeoff (Figure 5)")
print(" 18. figure6_radar_chart_chl.pdf            - Radar chart top 5 models (Figure 6)")
print("\n📈 METRICS CALCULATED PER MODEL:")
print("  • Accuracy, Precision, Recall, F1-Score")
print("  • R², MSE, MAE (for regression)")
print("  • Negative Log-Likelihood (NLL)")
print("  • Expected Calibration Error (ECE)")
print("  • Training and prediction time")
print("  • Memory usage (RAM)")
print("  • Detailed confusion matrices")
print("  • Predicted vs Actual scatter plots with residuals")
print("  • Learning curves (training vs validation performance)")
print("  • Temporal decomposition (seasonal and annual patterns)")
print("  • Feature importance (top features affecting chlorophyll)")

# Example prediction with new data
if len(X_test_class) > 0:
    print("\n" + "="*80)
    print("EXAMPLE PREDICTION WITH NEW DATA")
    print("="*80)
    
    # Use best classification model
    if mejor_clasificacion[0]:
        best_model_class = resultados_globales[mejor_clasificacion[0]]['modelo']
        
        # Select some test samples
        sample_indices = np.random.choice(len(X_test_class), min(3, len(X_test_class)), replace=False)
        
        print(f"\nNew data (3 samples):")
        for idx in sample_indices:
            sample_features = X_test_class[idx:idx+1]
            if isinstance(sample_features, np.ndarray):
                sample_features = pd.DataFrame(sample_features, columns=feature_cols)
            print(f"Sample {idx}:")
            print(sample_features)
            
            # Scale and predict
            sample_scaled = scaler_class.transform(sample_features)
            
            try:
                prediccion = best_model_class.predict(sample_scaled)
                prediccion_clase = y_classes[prediccion[0]] if 'y_classes' in locals() else prediccion[0]
                print(f"🎯 Prediction ({mejor_clasificacion[0]}): {prediccion_clase}")
                print(f"📊 Actual value: {y_classes[y_test_class[idx]] if 'y_classes' in locals() else y_test_class[idx]}")
                print("-" * 40)
            except Exception as e:
                print(f"  Error in prediction: {e}")

print("\n" + "="*80)
print("📋 DATASET SUMMARY")
print("="*80)
print(f"• Dataset: Mar Menor - Chlorophyll")
print(f"• Total samples: {len(df)}")
print(f"• Features: {len(feature_cols)}")
print(f"• Target variable: {target_column}")
print(f"• Chlorophyll range: {df[target_column].min():.2f} - {df[target_column].max():.2f} µg/L")
print(f"• Categories created: {len(y_classes) if 'y_classes' in locals() else 4}")
print(f"• Best accuracy achieved: {mejor_resultado['accuracy']:.4f} ({mejor_resultado['accuracy']*100:.1f}%)")
print("="*80)
