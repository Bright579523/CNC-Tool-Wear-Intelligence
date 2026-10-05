"""
Phase 4.4: Feature Ablation Study on NASA Milling Dataset (V2 Flagship Project)

Research Question:
Does smcAC_spindle_band_pwr provide meaningful incremental predictive value
beyond the locked 5-feature Primary Set?

Experimental Setup:
- Model A: Locked Primary 5 Features (smcAC_rms, vib_spindle_kurtosis, vib_spindle_p2p, AE_table_rms, AE_spindle_p2p)
- Model B: Primary 5 Features + smcAC_spindle_band_pwr
- Validation: Grouped Leave-One-Tool-Out (LOGO) across 16 tools (case). Identical folds for A and B.
- Models & Hyperparameters (Identical across A and B):
  1. Ridge (alpha=1.0, StandardScaler in pipeline)
  2. SVR (kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', StandardScaler in pipeline)
  3. Random Forest (n_estimators=100, max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=42)
  4. Gradient Boosting (n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)
- Outputs:
  - reports/phase4_4_feature_ablation/feature_ablation_summary.csv
  - reports/phase4_4_feature_ablation/fold_level_comparison.csv
  - reports/phase4_4_feature_ablation/oof_predictions_ablation.csv
  - reports/phase4_4_feature_ablation/figures/fig1_fold_level_delta_mae.png
  - reports/phase4_4_feature_ablation/figures/fig2_overall_mae_comparison.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
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
ABLATION_DIR = WORKSPACE_DIR / "reports" / "phase4_4_feature_ablation"
FIGURES_DIR = ABLATION_DIR / "figures"
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
print("STARTING PHASE 4.4: FEATURE ABLATION STUDY (smcAC_spindle_band_pwr)")
print("="*80)

# -------------------------------------------------------------
# 1. Dataset Integrity Checks
# -------------------------------------------------------------
dataset_path = DATA_DIR / "feature_dataset_v32.csv"
df = pd.read_csv(dataset_path)

FEATURES_A = [
    'smcAC_rms',
    'vib_spindle_kurtosis',
    'vib_spindle_p2p',
    'AE_table_rms',
    'AE_spindle_p2p'
]

FEATURES_B = FEATURES_A + ['smcAC_spindle_band_pwr']
TARGET_COL = 'VB_mm'

print(f"Loaded dataset: {len(df)} rows.")
assert len(df) == 145, f"Unexpected row count: {len(df)}"
assert df[FEATURES_B].isna().sum().sum() == 0, "Missing values in features!"
assert df[TARGET_COL].isna().sum() == 0, "Missing values in target!"

cases = sorted(df['case'].unique())
print(f"Validation Groups (Cases/Tools): {len(cases)} groups: {cases}")

# -------------------------------------------------------------
# 2. LOGO Validation Setup & Model Definitions
# -------------------------------------------------------------
logo = LeaveOneGroupOut()
groups = df['case'].values
y = df[TARGET_COL].values

X_A = df[FEATURES_A].values
X_B = df[FEATURES_B].values

MODEL_CONFIGS = {
    'Ridge': {
        'model_A': Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0, random_state=42))]),
        'model_B': Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0, random_state=42))])
    },
    'SVR': {
        'model_A': Pipeline([('scaler', StandardScaler()), ('model', SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'))]),
        'model_B': Pipeline([('scaler', StandardScaler()), ('model', SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'))])
    },
    'Random Forest': {
        'model_A': RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=42),
        'model_B': RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=42)
    },
    'Gradient Boosting': {
        'model_A': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42),
        'model_B': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)
    }
}

oof_preds_A = {m: np.zeros(len(df)) for m in MODEL_CONFIGS}
oof_preds_B = {m: np.zeros(len(df)) for m in MODEL_CONFIGS}

fold_records = []

print("\n--- RUNNING PAIRED LEAVE-ONE-TOOL-OUT CROSS-VALIDATION (16 FOLDS) ---")

for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_A, y, groups=groups)):
    held_out_case = groups[test_idx[0]]
    n_test = len(test_idx)
    
    y_train, y_test = y[train_idx], y[test_idx]
    
    fold_entry = {
        'fold': fold_idx + 1,
        'case': held_out_case,
        'num_runs': n_test,
        'condition_id': df.iloc[test_idx[0]]['condition_id'],
        'material_name': df.iloc[test_idx[0]]['material_name'],
        'DOC_mm': df.iloc[test_idx[0]]['DOC_mm'],
        'feed_mm_rev': df.iloc[test_idx[0]]['feed_mm_rev']
    }
    
    for m_name, cfg in MODEL_CONFIGS.items():
        # Fit Model A (5 Primary Features)
        mod_A = cfg['model_A']
        mod_A.fit(X_A[train_idx], y_train)
        pred_A = mod_A.predict(X_A[test_idx])
        oof_preds_A[m_name][test_idx] = pred_A
        mae_A = mean_absolute_error(y_test, pred_A)
        
        # Fit Model B (5 Primary + Spindle Band Power)
        mod_B = cfg['model_B']
        mod_B.fit(X_B[train_idx], y_train)
        pred_B = mod_B.predict(X_B[test_idx])
        oof_preds_B[m_name][test_idx] = pred_B
        mae_B = mean_absolute_error(y_test, pred_B)
        
        delta_mae = mae_B - mae_A
        
        fold_entry[f'{m_name}_MAE_A'] = round(mae_A, 4)
        fold_entry[f'{m_name}_MAE_B'] = round(mae_B, 4)
        fold_entry[f'{m_name}_delta_MAE'] = round(delta_mae, 4)
        
    fold_records.append(fold_entry)

df_fold_comparison = pd.DataFrame(fold_records)
fold_comp_csv = ABLATION_DIR / "fold_level_comparison.csv"
df_fold_comparison.to_csv(fold_comp_csv, index=False)
print(f"Saved Fold-Level Comparison: {fold_comp_csv}")

# -------------------------------------------------------------
# 3. Overall OOF Metrics & Ablation Summary
# -------------------------------------------------------------
summary_records = []

for m_name in MODEL_CONFIGS:
    # Model A metrics
    pred_A = oof_preds_A[m_name]
    mae_A = mean_absolute_error(y, pred_A)
    rmse_A = root_mean_squared_error(y, pred_A)
    r2_A = r2_score(y, pred_A)
    
    # Model B metrics
    pred_B = oof_preds_B[m_name]
    mae_B = mean_absolute_error(y, pred_B)
    rmse_B = root_mean_squared_error(y, pred_B)
    r2_B = r2_score(y, pred_B)
    
    # Differences
    delta_mae = mae_B - mae_A
    rel_mae_pct = (delta_mae / mae_A) * 100
    delta_rmse = rmse_B - rmse_A
    rel_rmse_pct = (delta_rmse / rmse_A) * 100
    delta_r2 = r2_B - r2_A
    
    # Fold-level statistics
    deltas = df_fold_comparison[f'{m_name}_delta_MAE']
    mean_delta = deltas.mean()
    median_delta = deltas.median()
    n_improved = (deltas < 0).sum()
    n_worse = (deltas > 0).sum()
    n_tied = (deltas == 0).sum()
    min_delta = deltas.min()
    max_delta = deltas.max()
    
    # Row for Model A
    summary_records.append({
        'Model': m_name,
        'Feature_Set': 'Primary 5 (Model A)',
        'MAE_mm': round(mae_A, 4),
        'RMSE_mm': round(rmse_A, 4),
        'R2': round(r2_A, 4),
        'Delta_MAE_mm': 0.0,
        'Rel_MAE_Change_Pct': '0.0%',
        'Delta_RMSE_mm': 0.0,
        'Delta_R2': 0.0,
        'Folds_Improved': 'Baseline',
        'Mean_Fold_Delta_MAE': 0.0,
        'Verdict': 'Baseline Reference'
    })
    
    # Row for Model B
    summary_records.append({
        'Model': m_name,
        'Feature_Set': 'Primary 5 + Band Power (Model B)',
        'MAE_mm': round(mae_B, 4),
        'RMSE_mm': round(rmse_B, 4),
        'R2': round(r2_B, 4),
        'Delta_MAE_mm': round(delta_mae, 4),
        'Rel_MAE_Change_Pct': f"{rel_mae_pct:+.2f}%",
        'Delta_RMSE_mm': round(delta_rmse, 4),
        'Delta_R2': round(delta_r2, 4),
        'Folds_Improved': f"{n_improved}/16 (Worse: {n_worse}/16)",
        'Mean_Fold_Delta_MAE': round(mean_delta, 4),
        'Verdict': 'Improved' if delta_mae < -0.002 else ('Degraded' if delta_mae > 0.002 else 'Negligible Change')
    })

df_ablation_summary = pd.DataFrame(summary_records)
summary_csv = ABLATION_DIR / "feature_ablation_summary.csv"
df_ablation_summary.to_csv(summary_csv, index=False)
print(f"Saved Feature Ablation Summary: {summary_csv}")

print("\n" + "="*80)
print("FEATURE ABLATION SUMMARY (MODEL A vs MODEL B)")
print("="*80)
cols_to_print = ['Model', 'Feature_Set', 'MAE_mm', 'RMSE_mm', 'R2', 'Delta_MAE_mm', 'Rel_MAE_Change_Pct', 'Folds_Improved', 'Verdict']
print(df_ablation_summary[cols_to_print].to_string(index=False))

# -------------------------------------------------------------
# 4. Save Paired OOF Predictions
# -------------------------------------------------------------
oof_ablation_df = df[['case', 'run', 'condition_id', 'material_name', 'DOC_mm', 'feed_mm_rev', 'VB_mm']].copy()

for m_name in MODEL_CONFIGS:
    oof_ablation_df[f'pred_A_{m_name}'] = oof_preds_A[m_name]
    oof_ablation_df[f'pred_B_{m_name}'] = oof_preds_B[m_name]
    oof_ablation_df[f'residual_A_{m_name}'] = df['VB_mm'] - oof_preds_A[m_name]
    oof_ablation_df[f'residual_B_{m_name}'] = df['VB_mm'] - oof_preds_B[m_name]

oof_ablation_csv = ABLATION_DIR / "oof_predictions_ablation.csv"
oof_ablation_df.to_csv(oof_ablation_csv, index=False)
print(f"Saved Paired OOF Predictions: {oof_ablation_csv}")

# -------------------------------------------------------------
# 5. Diagnostic Figures
# -------------------------------------------------------------
# Figure 1: Fold-level Paired Delta MAE across 16 Folds
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
axes = axes.flatten()

model_list = list(MODEL_CONFIGS.keys())
tool_cases = df_fold_comparison['case'].values
x_ticks = np.arange(len(tool_cases))

for idx, m_name in enumerate(model_list):
    ax = axes[idx]
    deltas = df_fold_comparison[f'{m_name}_delta_MAE'].values
    
    # Colors: Green if improved (<0), Red if degraded (>0)
    bar_colors = ['#27ae60' if d < 0 else ('#e74c3c' if d > 0 else '#7f8c8d') for d in deltas]
    
    ax.bar(x_ticks, deltas, color=bar_colors, edgecolor='black', alpha=0.85, width=0.6)
    ax.axhline(0, color='black', linestyle='--', lw=1.2, alpha=0.8)
    
    mean_d = deltas.mean()
    median_d = np.median(deltas)
    n_imp = (deltas < 0).sum()
    
    ax.set_title(f"{m_name}\nMean ΔMAE={mean_d:+.4f}mm | Median={median_d:+.4f}mm | Improved in {n_imp}/16 Folds", 
                 fontsize=10, fontweight='bold')
    ax.set_xlabel("Held-Out Tool Case (Fold)", fontsize=9, fontweight='bold')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"C{c}" for c in tool_cases], fontsize=8)
    if idx % 2 == 0:
        ax.set_ylabel("ΔMAE: Model B - Model A (mm)", fontsize=9, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')
    ax.set_ylim(-0.06, 0.06)

plt.suptitle("Figure 1: Paired Fold-Level ΔMAE (Model B − Model A) Across 16 Leave-One-Tool-Out Folds\n"
             "(Green = Band Power Improved Error | Red = Band Power Degraded Error | Values Near Zero = No Meaningful Change)", 
             fontsize=11.5, fontweight='bold', y=0.99)
plt.tight_layout()
fig1_path = FIGURES_DIR / "fig1_fold_level_delta_mae.png"
plt.savefig(fig1_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved Figure 1: {fig1_path.name}")

# Figure 2: Overall MAE Comparison (Model A vs Model B)
fig, ax = plt.subplots(figsize=(9, 5.5))

m_labels = list(MODEL_CONFIGS.keys())
x_pos = np.arange(len(m_labels))
width = 0.35

maes_A = [mean_absolute_error(y, oof_preds_A[m]) for m in m_labels]
maes_B = [mean_absolute_error(y, oof_preds_B[m]) for m in m_labels]

rects1 = ax.bar(x_pos - width/2, maes_A, width, label='Model A (5 Primary Features)', color='#3498db', edgecolor='black', alpha=0.85)
rects2 = ax.bar(x_pos + width/2, maes_B, width, label='Model B (5 Primary + Spindle Band Power)', color='#9b59b6', edgecolor='black', alpha=0.85)

ax.set_ylabel("Out-of-Fold MAE (mm of VB)", fontsize=10.5, fontweight='bold')
ax.set_title("Figure 2: Overall Out-of-Fold MAE Comparison: Model A vs. Model B\n"
             "(Testing whether adding smcAC_spindle_band_pwr provides incremental value across 4 model families)", 
             fontsize=11.5, fontweight='bold', pad=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(m_labels, fontsize=9.5, fontweight='bold')
ax.legend(fontsize=9.5, loc='upper right')
ax.grid(True, linestyle=':', alpha=0.5, axis='y')
ax.set_ylim(0, 0.18)

# Value annotations & delta percentages
for idx in range(len(m_labels)):
    mA = maes_A[idx]
    mB = maes_B[idx]
    diff = mB - mA
    pct = (diff / mA) * 100
    
    # Annotate A
    ax.annotate(f"{mA:.4f}", xy=(x_pos[idx] - width/2, mA), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    # Annotate B
    ax.annotate(f"{mB:.4f}", xy=(x_pos[idx] + width/2, mB), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Text for Delta
    delta_color = '#27ae60' if diff < 0 else ('#c0392b' if diff > 0 else 'black')
    ax.annotate(f"Δ: {diff:+.4f}mm\n({pct:+.1f}%)", xy=(x_pos[idx], max(mA, mB) + 0.012),
                ha='center', va='bottom', fontsize=8, color=delta_color, fontweight='bold')

plt.tight_layout()
fig2_path = FIGURES_DIR / "fig2_overall_mae_comparison.png"
plt.savefig(fig2_path, dpi=200)
plt.close()
print(f"Saved Figure 2: {fig2_path.name}")

print("\n" + "="*80)
print("PHASE 4.4 FEATURE ABLATION STUDY COMPLETED SUCCESSFULLY!")
print("="*80)
