"""
Phase 4.1–4.3: Initial ML Benchmark on NASA Milling Dataset (V2 Flagship Project)

Strategic Directives:
1. Locked Primary Feature Set ONLY:
   - smcAC_rms
   - vib_spindle_kurtosis
   - vib_spindle_p2p
   - AE_table_rms
   - AE_spindle_p2p
2. Target: VB_mm (continuous regression)
3. Forbidden as features: case, run, cumulative_time_min, condition_id
4. Validation: Grouped Leave-One-Group-Out (LOGO) by case (16 independent tool folds).
5. Preprocessing: StandardScaler fitted ONLY on training fold for linear/kernel models.
6. Models:
   - Model 0: Mean Baseline
   - Model 1: Ridge Regression
   - Model 2: Support Vector Regression (SVR)
   - Model 3: Random Forest Regressor
   - Model 4: Gradient Boosting Regressor
7. Metrics: MAE (Primary, in mm), RMSE, R2, and Fold-level stability.
8. Deliverables:
   - data/phase4_oof_predictions.csv
   - reports/phase4_model_summary.csv
   - reports/phase4_fold_metrics.csv
   - 4 Figures in reports/figures/phase4_benchmark/
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# -------------------------------------------------------------
# Configuration and Directories
# -------------------------------------------------------------
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
REPORTS_DIR = WORKSPACE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "phase4_benchmark"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'figure.titlesize': 12,
    'figure.dpi': 200
})

print("="*80)
print("STARTING PHASE 4.1–4.3: INITIAL ML BENCHMARK")
print("="*80)

# -------------------------------------------------------------
# 1. Phase 4.1: Dataset Loading & Rigorous Quality Checks
# -------------------------------------------------------------
dataset_path = DATA_DIR / "feature_dataset_v32.csv"
df = pd.read_csv(dataset_path)

PRIMARY_FEATURES = [
    'smcAC_rms',
    'vib_spindle_kurtosis',
    'vib_spindle_p2p',
    'AE_table_rms',
    'AE_spindle_p2p'
]
TARGET_COL = 'VB_mm'
METADATA_COLS = ['case', 'run', 'material_name', 'material_code', 'DOC_mm', 'feed_mm_rev', 'condition_id', 'cumulative_time_min']

print("\n--- PHASE 4.1: ML DATASET PREPARATION CHECKS ---")
print(f"Row count: {len(df)} (Expected: 145)")
assert len(df) == 145, f"Unexpected row count: {len(df)}"

# Check missing values
missing_features = df[PRIMARY_FEATURES].isna().sum().to_dict()
missing_target = df[TARGET_COL].isna().sum()
print(f"Primary features missing values: {missing_features}")
print(f"Target missing values: {missing_target}")
assert all(v == 0 for v in missing_features.values()), "Missing values detected in primary features!"
assert missing_target == 0, "Missing values detected in target!"

# Check duplicates
duplicates = df.duplicated(subset=['case', 'run']).sum()
print(f"Duplicate (case, run) rows: {duplicates}")
assert duplicates == 0, "Duplicate rows detected!"

# Check data types and variance
for col in PRIMARY_FEATURES:
    dtype = df[col].dtype
    var = df[col].var()
    print(f"  Feature '{col}': dtype={dtype}, var={var:.6f}, range=[{df[col].min():.4f}, {df[col].max():.4f}]")
    assert var > 1e-5, f"Near-zero variance in feature {col}"

# Target distribution
print(f"Target '{TARGET_COL}': range=[{df[TARGET_COL].min():.4f}, {df[TARGET_COL].max():.4f}] mm, mean={df[TARGET_COL].mean():.4f}, std={df[TARGET_COL].std():.4f}")

# Grouping verification
cases = sorted(df['case'].unique())
print(f"Validation Groups (Cases/Tools): {len(cases)} groups: {cases}")
runs_per_case = df['case'].value_counts().sort_index().to_dict()
print(f"Runs per tool: {runs_per_case}")

print("Dataset preparation checks: 100% PASSED!")

# -------------------------------------------------------------
# 2. Phase 4.2 & 4.3: Validation Setup & Model Benchmark
# -------------------------------------------------------------
logo = LeaveOneGroupOut()
groups = df['case'].values
X = df[PRIMARY_FEATURES].values
y = df[TARGET_COL].values

MODELS = {
    'Mean Baseline': None,
    'Ridge': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0, random_state=42))
    ]),
    'SVR': Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR(kernel='rbf', C=1.0, epsilon=0.1))
    ]),
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
}

# Containers for predictions
# Each model will produce 145 out-of-fold predictions
oof_predictions = {m_name: np.zeros(len(df)) for m_name in MODELS}
fold_metrics_list = []

# Iterate across 16 folds (Leave-One-Tool-Out)
print("\n--- RUNNING LEAVE-ONE-TOOL-OUT GROUPED CROSS-VALIDATION (16 FOLDS) ---")

for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
    held_out_case = groups[test_idx[0]]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    n_test = len(test_idx)
    fold_result = {
        'fold': fold_idx + 1,
        'held_out_case': held_out_case,
        'num_test_runs': n_test,
        'condition_id': df.iloc[test_idx[0]]['condition_id'],
        'material_name': df.iloc[test_idx[0]]['material_name'],
        'DOC_mm': df.iloc[test_idx[0]]['DOC_mm'],
        'feed_mm_rev': df.iloc[test_idx[0]]['feed_mm_rev']
    }
    
    for m_name, model in MODELS.items():
        if m_name == 'Mean Baseline':
            # Predict the training set mean
            y_pred = np.full(n_test, y_train.mean())
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
        oof_predictions[m_name][test_idx] = y_pred
        
        # Calculate fold-level MAE
        f_mae = mean_absolute_error(y_test, y_pred)
        fold_result[f'MAE_{m_name}'] = round(f_mae, 4)
        
    fold_metrics_list.append(fold_result)
    
df_fold_metrics = pd.DataFrame(fold_metrics_list)
fold_metrics_csv = REPORTS_DIR / "phase4_fold_metrics.csv"
df_fold_metrics.to_csv(fold_metrics_csv, index=False)
print(f"Saved Fold Metrics Table: {fold_metrics_csv}")

# -------------------------------------------------------------
# 3. Overall Out-of-Fold Evaluation
# -------------------------------------------------------------
summary_rows = []
oof_df_rows = []

for m_name in MODELS:
    y_pred = oof_predictions[m_name]
    mae = mean_absolute_error(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Fold MAE statistics across the 16 tools
    f_maes = df_fold_metrics[f'MAE_{m_name}']
    f_mean = f_maes.mean()
    f_std = f_maes.std()
    f_min = f_maes.min()
    f_max = f_maes.max()
    
    summary_rows.append({
        'Model': m_name,
        'MAE_mm': round(mae, 4),
        'RMSE_mm': round(rmse, 4),
        'R2': round(r2, 4),
        'Fold_MAE_Mean': round(f_mean, 4),
        'Fold_MAE_Std': round(f_std, 4),
        'Fold_MAE_Min': round(f_min, 4),
        'Fold_MAE_Max': round(f_max, 4)
    })
    
    # Collect tidy rows for phase4_oof_predictions.csv
    for i in range(len(df)):
        oof_df_rows.append({
            'case': int(df.iloc[i]['case']),
            'run': int(df.iloc[i]['run']),
            'condition_id': df.iloc[i]['condition_id'],
            'material_name': df.iloc[i]['material_name'],
            'DOC_mm': float(df.iloc[i]['DOC_mm']),
            'feed_mm_rev': float(df.iloc[i]['feed_mm_rev']),
            'VB_mm': float(y[i]),
            'model_name': m_name,
            'prediction': float(y_pred[i]),
            'residual': float(y[i] - y_pred[i])
        })

df_model_summary = pd.DataFrame(summary_rows)
summary_csv = REPORTS_DIR / "phase4_model_summary.csv"
df_model_summary.to_csv(summary_csv, index=False)
print(f"Saved Model Summary Table: {summary_csv}")

df_oof_predictions = pd.DataFrame(oof_df_rows)
oof_csv = DATA_DIR / "phase4_oof_predictions.csv"
df_oof_predictions.to_csv(oof_csv, index=False)
print(f"Saved Out-of-Fold Predictions: {oof_csv} ({len(df_oof_predictions)} rows)")

print("\n" + "="*80)
print("BENCHMARK RESULTS SUMMARY (OUT-OF-FOLD EVALUATION)")
print("="*80)
print(df_model_summary.to_string(index=False))

# -------------------------------------------------------------
# 4. Generate Diagnostic Figures
# -------------------------------------------------------------
# Figure 1: Actual vs Predicted VB for each model
fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)

model_colors = {
    'Mean Baseline': '#7f8c8d',
    'Ridge': '#2980b9',
    'SVR': '#8e44ad',
    'Random Forest': '#27ae60',
    'Gradient Boosting': '#d35400'
}

for idx, m_name in enumerate(MODELS):
    ax = axes[idx]
    y_pred = oof_predictions[m_name]
    r2_val = df_model_summary.loc[df_model_summary['Model'] == m_name, 'R2'].values[0]
    mae_val = df_model_summary.loc[df_model_summary['Model'] == m_name, 'MAE_mm'].values[0]
    
    # Scatter colored by Material
    for mat_name, col, marker in [('Cast Iron', '#1f77b4', 'o'), ('Stainless Steel J45', '#d62728', 's')]:
        mask = df['material_name'] == mat_name
        ax.scatter(y[mask], y_pred[mask], color=col, marker=marker, s=35, alpha=0.7, edgecolors='black', lw=0.5, label=mat_name if idx==0 else "")
        
    # 1:1 Identity Line
    ax.plot([0, 1.6], [0, 1.6], 'k--', lw=1.2, alpha=0.7, label='Ideal 1:1 Line' if idx==0 else "")
    
    ax.set_title(f"{m_name}\nMAE={mae_val:.4f}mm | R²={r2_val:.3f}", fontsize=10, fontweight='bold')
    ax.set_xlabel("Actual Flank Wear VB (mm)", fontsize=9, fontweight='bold')
    if idx == 0:
        ax.set_ylabel("Predicted Flank Wear VB (mm)", fontsize=9, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(-0.05, 1.65)
    ax.set_ylim(-0.05, 1.65)
    ax.grid(True, linestyle=':', alpha=0.5)

plt.suptitle("Figure 1: Actual vs. Predicted Flank Wear (VB) Across Initial Benchmark Models\n"
             "(Grouped Leave-One-Tool-Out Validation Across 16 Independent Tool Inserts)", 
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
fig1_path = FIGURES_DIR / "fig1_actual_vs_predicted_by_model.png"
plt.savefig(fig1_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved Figure 1: {fig1_path.name}")

# Figure 2: Residuals vs Actual VB (Error Analysis)
fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)

for idx, m_name in enumerate(MODELS):
    ax = axes[idx]
    y_pred = oof_predictions[m_name]
    residuals = y - y_pred
    
    for mat_name, col, marker in [('Cast Iron', '#1f77b4', 'o'), ('Stainless Steel J45', '#d62728', 's')]:
        mask = df['material_name'] == mat_name
        ax.scatter(y[mask], residuals[mask], color=col, marker=marker, s=35, alpha=0.7, edgecolors='black', lw=0.5, label=mat_name if idx==0 else "")
        
    ax.axhline(0, color='black', linestyle='--', lw=1.2, alpha=0.8)
    ax.axhline(0.1, color='gray', linestyle=':', lw=0.8, alpha=0.6)
    ax.axhline(-0.1, color='gray', linestyle=':', lw=0.8, alpha=0.6)
    
    ax.set_title(f"{m_name}\nMax Error={np.abs(residuals).max():.3f}mm", fontsize=10, fontweight='bold')
    ax.set_xlabel("Actual Flank Wear VB (mm)", fontsize=9, fontweight='bold')
    if idx == 0:
        ax.set_ylabel("Residual: Actual - Predicted (mm)", fontsize=9, fontweight='bold')
        ax.legend(loc='lower left', fontsize=8)
    ax.set_xlim(-0.05, 1.65)
    ax.set_ylim(-0.85, 0.85)
    ax.grid(True, linestyle=':', alpha=0.5)

plt.suptitle("Figure 2: Residual Analysis Across Flank Wear Degradation Range\n"
             "(Assessing underprediction at high wear and heteroscedasticity across models)", 
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
fig2_path = FIGURES_DIR / "fig2_residuals_vs_actual_vb.png"
plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved Figure 2: {fig2_path.name}")

# Figure 3: Model MAE & RMSE Comparison Bar Chart
fig, ax1 = plt.subplots(figsize=(10, 5.5))

m_labels = list(MODELS.keys())
x_pos = np.arange(len(m_labels))
width = 0.35

maes = [df_model_summary.loc[df_model_summary['Model'] == m, 'MAE_mm'].values[0] for m in m_labels]
rmses = [df_model_summary.loc[df_model_summary['Model'] == m, 'RMSE_mm'].values[0] for m in m_labels]

rects1 = ax1.bar(x_pos - width/2, maes, width, label='MAE (Primary Metric) [mm]', color='#3498db', edgecolor='black', alpha=0.85)
rects2 = ax1.bar(x_pos + width/2, rmses, width, label='RMSE (Secondary Metric) [mm]', color='#e67e22', edgecolor='black', alpha=0.85)

ax1.set_ylabel("Error (mm of VB)", fontsize=10.5, fontweight='bold')
ax1.set_title("Figure 3: Initial ML Benchmark Model Performance Comparison\n"
              "(Out-of-Fold Evaluation Across 16 Leave-One-Tool-Out Groups)", fontsize=11.5, fontweight='bold', pad=12)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(m_labels, fontsize=9.5, fontweight='bold')
ax1.legend(fontsize=9.5, loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.5, axis='y')

# Add values above bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax1.annotate(f'{height:.4f}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
fig3_path = FIGURES_DIR / "fig3_model_mae_comparison.png"
plt.savefig(fig3_path, dpi=200)
plt.close()
print(f"Saved Figure 3: {fig3_path.name}")

# Figure 4: Fold-Level MAE Variation (Tool-to-Tool Stability)
fig, ax = plt.subplots(figsize=(12, 6))

fold_mae_data = [df_fold_metrics[f'MAE_{m}'] for m in m_labels]
bp = ax.boxplot(fold_mae_data, patch_artist=True, tick_labels=m_labels, widths=0.5)

b_colors = ['#7f8c8d', '#2980b9', '#8e44ad', '#27ae60', '#d35400']
for patch, color in zip(bp['boxes'], b_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)
    patch.set_edgecolor('black')

# Overlay individual fold points
for idx, m in enumerate(m_labels):
    y_vals = df_fold_metrics[f'MAE_{m}']
    x_jitter = np.random.normal(idx + 1, 0.04, size=len(y_vals))
    ax.scatter(x_jitter, y_vals, color='black', alpha=0.6, s=25, zorder=4)

ax.set_ylabel("Fold MAE (mm of VB per Tool)", fontsize=10.5, fontweight='bold')
ax.set_title("Figure 4: Tool-to-Tool Fold MAE Stability Across 16 Independent Tool Folds\n"
             "(Points represent individual held-out tool cases; box represents quartile dispersion)", 
             fontsize=11.5, fontweight='bold', pad=12)
ax.grid(True, linestyle=':', alpha=0.5, axis='y')

plt.tight_layout()
fig4_path = FIGURES_DIR / "fig4_fold_level_mae_variation.png"
plt.savefig(fig4_path, dpi=200)
plt.close()
print(f"Saved Figure 4: {fig4_path.name}")

print("\n" + "="*80)
print("PHASE 4.1–4.3 INITIAL ML BENCHMARK COMPLETED SUCCESSFULLY!")
print("="*80)
