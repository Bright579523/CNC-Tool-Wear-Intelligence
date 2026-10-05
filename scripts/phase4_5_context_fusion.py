"""
Phase 4.5: Context Fusion Model & Cutting Condition Integration
NASA Milling Dataset (V2 Flagship Project)

Research Question:
Does integrating machine operating context (material_code, DOC_mm, feed_mm_rev)
with the locked 5 Primary Sensor Features meaningfully improve wear prediction
and reduce the performance discrepancy between Cast Iron and Stainless Steel?

Experimental Setup:
- Model A (Sensor-Only Baseline):
  5 Primary Features: ['smcAC_rms', 'vib_spindle_kurtosis', 'vib_spindle_p2p', 'AE_table_rms', 'AE_spindle_p2p']
- Model B (Context Fusion Model):
  8 Features: Model A + ['material_code', 'DOC_mm', 'feed_mm_rev']
- Validation: Grouped Leave-One-Tool-Out (LOGO) across all 16 tools (case). Identical folds for A and B.
- Models & Hyperparameters (Identical across A and B, zero tuning):
  1. Ridge (alpha=1.0, StandardScaler in pipeline fitted strictly on training fold)
  2. SVR (kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', StandardScaler in pipeline)
  3. Random Forest (n_estimators=100, max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=42)
  4. Gradient Boosting (n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)

Outputs:
  - reports/phase4_5_context_fusion/context_fusion_summary.csv
  - reports/phase4_5_context_fusion/material_breakdown.csv
  - reports/phase4_5_context_fusion/fold_level_comparison.csv
  - reports/phase4_5_context_fusion/oof_predictions_context.csv
  - reports/phase4_5_context_fusion/figures/fig1_overall_mae_comparison.png
  - reports/phase4_5_context_fusion/figures/fig2_material_breakdown_mae.png
  - reports/phase4_5_context_fusion/figures/fig3_fold_level_delta_mae.png
  - reports/phase4_5_context_fusion/figures/fig4_feature_importance_context.png
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
REPORT_DIR = WORKSPACE_DIR / "reports" / "phase4_5_context_fusion"
FIGURES_DIR = REPORT_DIR / "figures"
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
print("STARTING PHASE 4.5: CONTEXT FUSION MODEL & CUTTING CONDITION INTEGRATION")
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

CONTEXT_FEATURES = ['material_code', 'DOC_mm', 'feed_mm_rev']
FEATURES_B = FEATURES_A + CONTEXT_FEATURES
TARGET_COL = 'VB_mm'

print(f"Loaded dataset: {len(df)} rows.")
assert len(df) == 145, f"Unexpected row count: {len(df)}"
assert df[FEATURES_B].isna().sum().sum() == 0, "Missing values in features!"
assert df[TARGET_COL].isna().sum().sum() == 0, "Missing values in target!"

cases = sorted(df['case'].unique())
print(f"Validation Groups (Cases/Tools): {len(cases)} groups: {cases}")
print(f"Model A Features (Sensor-Only, k={len(FEATURES_A)}): {FEATURES_A}")
print(f"Model B Features (Context Fusion, k={len(FEATURES_B)}): {FEATURES_B}")

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

# -------------------------------------------------------------
# 3. Paired LOGO Cross-Validation Loop
# -------------------------------------------------------------
oof_preds = {
    'case': df['case'].values,
    'run': df['run'].values,
    'material_name': df['material_name'].values,
    'material_code': df['material_code'].values,
    'DOC_mm': df['DOC_mm'].values,
    'feed_mm_rev': df['feed_mm_rev'].values,
    'condition_id': df['condition_id'].values,
    'VB_mm_actual': y
}

for model_name in MODEL_CONFIGS:
    oof_preds[f'{model_name}_A_pred'] = np.zeros(len(df))
    oof_preds[f'{model_name}_B_pred'] = np.zeros(len(df))

# Tracking fold-level metrics
fold_records = []

# Tracking coefficients / importances across folds
ridge_b_coefs = []
gbdt_b_importances = []
rf_b_importances = []

for fold_idx, (train_idx, test_idx) in enumerate(logo.split(df, y, groups)):
    case_id = groups[test_idx[0]]
    mat_name = df.iloc[test_idx[0]]['material_name']
    cond_id = df.iloc[test_idx[0]]['condition_id']
    y_test = y[test_idx]
    n_samples = len(test_idx)
    
    fold_entry = {
        'fold': fold_idx + 1,
        'case': case_id,
        'material_name': mat_name,
        'condition_id': cond_id,
        'n_runs': n_samples
    }
    
    for model_name, configs in MODEL_CONFIGS.items():
        # Fit Model A (Sensor-Only)
        clf_A = configs['model_A']
        clf_A.fit(X_A[train_idx], y[train_idx])
        pred_A = clf_A.predict(X_A[test_idx])
        oof_preds[f'{model_name}_A_pred'][test_idx] = pred_A
        
        # Fit Model B (Context Fusion)
        clf_B = configs['model_B']
        clf_B.fit(X_B[train_idx], y[train_idx])
        pred_B = clf_B.predict(X_B[test_idx])
        oof_preds[f'{model_name}_B_pred'][test_idx] = pred_B
        
        mae_A = mean_absolute_error(y_test, pred_A)
        mae_B = mean_absolute_error(y_test, pred_B)
        delta_mae = mae_B - mae_A
        rel_delta_mae = (delta_mae / mae_A) * 100 if mae_A > 0 else 0.0
        
        fold_entry[f'{model_name}_A_MAE'] = mae_A
        fold_entry[f'{model_name}_B_MAE'] = mae_B
        fold_entry[f'{model_name}_delta_MAE'] = delta_mae
        fold_entry[f'{model_name}_rel_delta_MAE_%'] = rel_delta_mae
        fold_entry[f'{model_name}_improved'] = bool(delta_mae < 0)
        
        # Save Model B weights for interpretation
        if model_name == 'Ridge':
            ridge_b_coefs.append(clf_B.named_steps['model'].coef_)
        elif model_name == 'Gradient Boosting':
            gbdt_b_importances.append(clf_B.feature_importances_)
        elif model_name == 'Random Forest':
            rf_b_importances.append(clf_B.feature_importances_)

    fold_records.append(fold_entry)

oof_df = pd.DataFrame(oof_preds)
oof_csv_path = REPORT_DIR / "oof_predictions_context.csv"
oof_df.to_csv(oof_csv_path, index=False)
print(f"Saved OOF predictions: {oof_csv_path}")

fold_df = pd.DataFrame(fold_records)
fold_csv_path = REPORT_DIR / "fold_level_comparison.csv"
fold_df.to_csv(fold_csv_path, index=False)
print(f"Saved fold-level comparison: {fold_csv_path}")

# -------------------------------------------------------------
# 4. Global Performance Summary
# -------------------------------------------------------------
summary_records = []
for model_name in MODEL_CONFIGS:
    pred_A = oof_df[f'{model_name}_A_pred'].values
    pred_B = oof_df[f'{model_name}_B_pred'].values
    
    mae_A = mean_absolute_error(y, pred_A)
    rmse_A = root_mean_squared_error(y, pred_A)
    r2_A = r2_score(y, pred_A)
    
    mae_B = mean_absolute_error(y, pred_B)
    rmse_B = root_mean_squared_error(y, pred_B)
    r2_B = r2_score(y, pred_B)
    
    delta_mae = mae_B - mae_A
    rel_delta_mae = (delta_mae / mae_A) * 100
    delta_rmse = rmse_B - rmse_A
    delta_r2 = r2_B - r2_A
    
    improved_folds = fold_df[f'{model_name}_improved'].sum()
    worse_folds = 16 - improved_folds
    
    summary_records.append({
        'Model': model_name,
        'Feature_Set_A': 'Primary 5 (Sensor-Only)',
        'MAE_A_mm': mae_A,
        'RMSE_A_mm': rmse_A,
        'R2_A': r2_A,
        'Feature_Set_B': 'Primary 5 + Context (8 Features)',
        'MAE_B_mm': mae_B,
        'RMSE_B_mm': rmse_B,
        'R2_B': r2_B,
        'Delta_MAE_mm': delta_mae,
        'Rel_Delta_MAE_%': rel_delta_mae,
        'Delta_RMSE_mm': delta_rmse,
        'Delta_R2': delta_r2,
        'Folds_Improved': f"{improved_folds} / 16 ({improved_folds/16*100:.1f}%)",
        'Folds_Worse': f"{worse_folds} / 16 ({worse_folds/16*100:.1f}%)"
    })

summary_df = pd.DataFrame(summary_records)
summary_csv_path = REPORT_DIR / "context_fusion_summary.csv"
summary_df.to_csv(summary_csv_path, index=False)
print(f"Saved summary: {summary_csv_path}")

print("\n" + "="*80)
print("GLOBAL OUT-OF-FOLD PERFORMANCE: SENSOR-ONLY (A) vs CONTEXT FUSION (B)")
print("="*80)
for _, r in summary_df.iterrows():
    print(f"{r['Model']:18s} | A: MAE={r['MAE_A_mm']:.4f}, R2={r['R2_A']:.4f} -> B: MAE={r['MAE_B_mm']:.4f}, R2={r['R2_B']:.4f} | dMAE={r['Delta_MAE_mm']:+.4f} ({r['Rel_Delta_MAE_%']:+.2f}%) | Improved: {r['Folds_Improved']}")

# -------------------------------------------------------------
# 5. Material Breakdown Analysis
# -------------------------------------------------------------
material_records = []
ci_mask = oof_df['material_name'] == 'Cast Iron'
ss_mask = oof_df['material_name'] == 'Stainless Steel J45'

for model_name in MODEL_CONFIGS:
    pred_A = oof_df[f'{model_name}_A_pred'].values
    pred_B = oof_df[f'{model_name}_B_pred'].values
    
    # Cast Iron
    mae_ci_A = mean_absolute_error(y[ci_mask], pred_A[ci_mask])
    rmse_ci_A = root_mean_squared_error(y[ci_mask], pred_A[ci_mask])
    r2_ci_A = r2_score(y[ci_mask], pred_A[ci_mask])
    
    mae_ci_B = mean_absolute_error(y[ci_mask], pred_B[ci_mask])
    rmse_ci_B = root_mean_squared_error(y[ci_mask], pred_B[ci_mask])
    r2_ci_B = r2_score(y[ci_mask], pred_B[ci_mask])
    
    # Stainless Steel
    mae_ss_A = mean_absolute_error(y[ss_mask], pred_A[ss_mask])
    rmse_ss_A = root_mean_squared_error(y[ss_mask], pred_A[ss_mask])
    r2_ss_A = r2_score(y[ss_mask], pred_A[ss_mask])
    
    mae_ss_B = mean_absolute_error(y[ss_mask], pred_B[ss_mask])
    rmse_ss_B = root_mean_squared_error(y[ss_mask], pred_B[ss_mask])
    r2_ss_B = r2_score(y[ss_mask], pred_B[ss_mask])
    
    material_records.append({
        'Model': model_name,
        'Material': 'Cast Iron (n=97, 8 tools)',
        'MAE_A_mm': mae_ci_A,
        'MAE_B_mm': mae_ci_B,
        'Delta_MAE_mm': mae_ci_B - mae_ci_A,
        'Rel_Delta_MAE_%': (mae_ci_B - mae_ci_A) / mae_ci_A * 100,
        'R2_A': r2_ci_A,
        'R2_B': r2_ci_B
    })
    material_records.append({
        'Model': model_name,
        'Material': 'Stainless Steel J45 (n=48, 8 tools)',
        'MAE_A_mm': mae_ss_A,
        'MAE_B_mm': mae_ss_B,
        'Delta_MAE_mm': mae_ss_B - mae_ss_A,
        'Rel_Delta_MAE_%': (mae_ss_B - mae_ss_A) / mae_ss_A * 100,
        'R2_A': r2_ss_A,
        'R2_B': r2_ss_B
    })

mat_df = pd.DataFrame(material_records)
mat_csv_path = REPORT_DIR / "material_breakdown.csv"
mat_df.to_csv(mat_csv_path, index=False)
print(f"\nSaved material breakdown: {mat_csv_path}")

print("\n" + "="*80)
print("MATERIAL BREAKDOWN (MAE in mm)")
print("="*80)
for _, r in mat_df.iterrows():
    print(f"{r['Model']:18s} | {r['Material']:36s} | A: {r['MAE_A_mm']:.4f} -> B: {r['MAE_B_mm']:.4f} | dMAE={r['Delta_MAE_mm']:+.4f} ({r['Rel_Delta_MAE_%']:+.1f}%) | R2: {r['R2_A']:.4f} -> {r['R2_B']:.4f}")

# -------------------------------------------------------------
# 6. Figures Generation
# -------------------------------------------------------------

# Figure 1: Overall MAE Comparison (Model A vs Model B)
fig, ax = plt.subplots(figsize=(8, 4.5))
models = list(MODEL_CONFIGS.keys())
x = np.arange(len(models))
width = 0.35

mae_A_vals = [summary_df.loc[summary_df['Model'] == m, 'MAE_A_mm'].values[0] for m in models]
mae_B_vals = [summary_df.loc[summary_df['Model'] == m, 'MAE_B_mm'].values[0] for m in models]

bars1 = ax.bar(x - width/2, mae_A_vals, width, label='Model A: Sensor-Only (5 features)', color='#4A7BB0', edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2, mae_B_vals, width, label='Model B: Context Fusion (5 Sensor + Mat/DOC/Feed)', color='#2E8B57', edgecolor='black', linewidth=0.8)

# Baseline reference line
ax.axhline(0.1998, color='#D9534F', linestyle='--', linewidth=1.2, label='Mean Baseline (0.1998 mm)')

for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#333333')

for idx, bar in enumerate(bars2):
    yval = bar.get_height()
    delta_pct = summary_df.loc[summary_df['Model'] == models[idx], 'Rel_Delta_MAE_%'].values[0]
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.4f}\n({delta_pct:+.1f}%)", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1B5E20')

ax.set_ylabel('Out-of-Fold MAE (mm)')
ax.set_title('Phase 4.5: Global Out-of-Fold MAE Comparison\nSensor-Only Baseline vs. Context Fusion Across 16 LOGO Folds')
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='bold')
ax.set_ylim(0, 0.23)
ax.legend(frameon=True, facecolor='#FAFAFA', edgecolor='#CCCCCC', loc='upper right')
ax.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
fig1_path = FIGURES_DIR / "fig1_overall_mae_comparison.png"
plt.savefig(fig1_path)
plt.close()
print(f"Saved: {fig1_path}")

# Figure 2: Material Breakdown Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

ci_A = [mat_df.loc[(mat_df['Model'] == m) & (mat_df['Material'].str.startswith('Cast Iron')), 'MAE_A_mm'].values[0] for m in models]
ci_B = [mat_df.loc[(mat_df['Model'] == m) & (mat_df['Material'].str.startswith('Cast Iron')), 'MAE_B_mm'].values[0] for m in models]

ss_A = [mat_df.loc[(mat_df['Model'] == m) & (mat_df['Material'].str.startswith('Stainless Steel')), 'MAE_A_mm'].values[0] for m in models]
ss_B = [mat_df.loc[(mat_df['Model'] == m) & (mat_df['Material'].str.startswith('Stainless Steel')), 'MAE_B_mm'].values[0] for m in models]

# Cast Iron subplot
bars_ci_1 = ax1.bar(x - width/2, ci_A, width, label='Sensor-Only (A)', color='#5C93C4', edgecolor='black', linewidth=0.8)
bars_ci_2 = ax1.bar(x + width/2, ci_B, width, label='Context Fusion (B)', color='#38A169', edgecolor='black', linewidth=0.8)
ax1.set_title('Cast Iron Workpiece (n=97 runs, 8 tools)\nEasier Machining / Stable Sensor Dynamics', fontsize=10, fontweight='bold')
ax1.set_ylabel('Out-of-Fold MAE (mm)')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontweight='bold')
ax1.grid(axis='y', linestyle=':', alpha=0.6)
ax1.set_ylim(0, 0.26)
ax1.legend(loc='upper right', frameon=True)

for bar in bars_ci_1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.4f}", ha='center', va='bottom', fontsize=7.5)
for idx, bar in enumerate(bars_ci_2):
    yval = bar.get_height()
    pct = (ci_B[idx] - ci_A[idx]) / ci_A[idx] * 100
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.4f}\n({pct:+.1f}%)", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#1B5E20')

# Stainless Steel subplot
bars_ss_1 = ax2.bar(x - width/2, ss_A, width, label='Sensor-Only (A)', color='#E07A5F', edgecolor='black', linewidth=0.8)
bars_ss_2 = ax2.bar(x + width/2, ss_B, width, label='Context Fusion (B)', color='#2B9348', edgecolor='black', linewidth=0.8)
ax2.set_title('Stainless Steel J45 (n=48 runs, 8 tools)\nChallenging Work-Hardening / High Wear Rates', fontsize=10, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontweight='bold')
ax2.grid(axis='y', linestyle=':', alpha=0.6)
ax2.legend(loc='upper right', frameon=True)

for bar in bars_ss_1:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.4f}", ha='center', va='bottom', fontsize=7.5)
for idx, bar in enumerate(bars_ss_2):
    yval = bar.get_height()
    pct = (ss_B[idx] - ss_A[idx]) / ss_A[idx] * 100
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.4f}\n({pct:+.1f}%)", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#1B5E20')

plt.suptitle('Phase 4.5: Impact of Context Fusion Across Workpiece Materials', fontsize=12, fontweight='bold')
plt.tight_layout()
fig2_path = FIGURES_DIR / "fig2_material_breakdown_mae.png"
plt.savefig(fig2_path)
plt.close()
print(f"Saved: {fig2_path}")

# Figure 3: Fold-Level Delta MAE for Ridge & SVR across 16 Folds
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

case_names = [f"Case {r['case']}\n({r['material_name'][:6]})" for _, r in fold_df.iterrows()]
x_cases = np.arange(len(fold_df))

# Ridge
delta_ridge = fold_df['Ridge_delta_MAE'].values
colors_ridge = ['#2E8B57' if d < 0 else '#D9534F' for d in delta_ridge]
ax1.bar(x_cases, delta_ridge, color=colors_ridge, edgecolor='black', linewidth=0.6)
ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax1.set_ylabel(r'$\Delta$ MAE (mm)')
ax1.set_title('Ridge Regression: Fold-Level Error Change (Model B vs Model A) [12/16 Improved]', fontsize=10, fontweight='bold')
ax1.grid(axis='y', linestyle=':', alpha=0.6)
for i, d in enumerate(delta_ridge):
    offset = 0.005 if d >= 0 else -0.015
    ax1.text(i, d + offset, f"{d:+.3f}", ha='center', va='bottom' if d < 0 else 'top', fontsize=7)

# SVR
delta_svr = fold_df['SVR_delta_MAE'].values
colors_svr = ['#2E8B57' if d < 0 else '#D9534F' for d in delta_svr]
ax2.bar(x_cases, delta_svr, color=colors_svr, edgecolor='black', linewidth=0.6)
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.set_ylabel(r'$\Delta$ MAE (mm)')
ax2.set_title('Support Vector Regression (SVR): Fold-Level Error Change [14/16 Improved]', fontsize=10, fontweight='bold')
ax2.set_xticks(x_cases)
ax2.set_xticklabels(case_names, fontsize=8)
ax2.grid(axis='y', linestyle=':', alpha=0.6)
for i, d in enumerate(delta_svr):
    offset = 0.005 if d >= 0 else -0.015
    ax2.text(i, d + offset, f"{d:+.3f}", ha='center', va='bottom' if d < 0 else 'top', fontsize=7)

plt.suptitle(r'Phase 4.5: Tool-by-Tool Paired $\Delta$ MAE (Green = Context Fusion Wins)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig3_path = FIGURES_DIR / "fig3_fold_level_delta_mae.png"
plt.savefig(fig3_path)
plt.close()
print(f"Saved: {fig3_path}")

# Figure 4: Feature Importance / Standardized Coefficients in Context Models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Average Ridge standardized coefficients
mean_ridge_coefs = np.mean(ridge_b_coefs, axis=0)
feat_names_b = FEATURES_B
sorted_idx_ridge = np.argsort(np.abs(mean_ridge_coefs))

ax1.barh(np.arange(len(feat_names_b)), mean_ridge_coefs[sorted_idx_ridge], 
         color=['#2E8B57' if feat_names_b[i] in CONTEXT_FEATURES else '#4A7BB0' for i in sorted_idx_ridge],
         edgecolor='black', linewidth=0.8)
ax1.set_yticks(np.arange(len(feat_names_b)))
ax1.set_yticklabels([feat_names_b[i] for i in sorted_idx_ridge], fontsize=8.5)
ax1.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax1.set_xlabel('Mean Standardized Coefficient across 16 Folds')
ax1.set_title('Ridge Regression Context Model\n(Green = Context Features, Blue = Sensors)', fontsize=10, fontweight='bold')
ax1.grid(axis='x', linestyle=':', alpha=0.6)

# Average GBDT feature importances
mean_gbdt_imp = np.mean(gbdt_b_importances, axis=0)
sorted_idx_gbdt = np.argsort(mean_gbdt_imp)

ax2.barh(np.arange(len(feat_names_b)), mean_gbdt_imp[sorted_idx_gbdt], 
         color=['#2E8B57' if feat_names_b[i] in CONTEXT_FEATURES else '#4A7BB0' for i in sorted_idx_gbdt],
         edgecolor='black', linewidth=0.8)
ax2.set_yticks(np.arange(len(feat_names_b)))
ax2.set_yticklabels([feat_names_b[i] for i in sorted_idx_gbdt], fontsize=8.5)
ax2.set_xlabel('Mean Gini / Split Importance across 16 Folds')
ax2.set_title('Gradient Boosting Context Model\n(Green = Context Features, Blue = Sensors)', fontsize=10, fontweight='bold')
ax2.grid(axis='x', linestyle=':', alpha=0.6)

plt.suptitle('Phase 4.5: Feature Influence in Context Fusion Models', fontsize=12, fontweight='bold')
plt.tight_layout()
fig4_path = FIGURES_DIR / "fig4_feature_importance_context.png"
plt.savefig(fig4_path)
plt.close()
print(f"Saved: {fig4_path}")

print("="*80)
print("PHASE 4.5 COMPUTATION AND VISUALIZATIONS COMPLETE!")
print("="*80)
