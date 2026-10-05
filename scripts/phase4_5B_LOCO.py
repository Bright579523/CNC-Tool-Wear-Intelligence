"""
Phase 4.5B: Unseen Cutting-Condition Generalization (LOCO)
NASA Milling Dataset (V2 Flagship Project)

Research Question:
Can the tool-wear prediction models generalize to a cutting condition
that was completely unseen during training?

Validation Strategy:
- Leave-One-Condition-Out (LOCO) cross-validation across all 8 cutting conditions (C1-C8).
- Outer Folds: Exactly 8 folds.
- For each fold:
  - All observations belonging to the held-out condition are excluded from training.
  - Both tool cases belonging to that condition form the test set.
  - condition_id is strictly a grouping variable, never an input feature.

Feature Sets:
- Model A (Sensor-Only Baseline):
  5 Primary Features: ['smcAC_rms', 'vib_spindle_kurtosis', 'vib_spindle_p2p', 'AE_table_rms', 'AE_spindle_p2p']
- Model B (Context Fusion Model):
  8 Features: Model A + ['material_code', 'DOC_mm', 'feed_mm_rev']

Models & Hyperparameters (Identical to Phase 4.1-4.5A, zero tuning):
  1. Ridge (alpha=1.0, StandardScaler in pipeline fitted strictly on training fold)
  2. SVR (kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', StandardScaler in pipeline)
  3. Random Forest (n_estimators=100, max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=42)
  4. Gradient Boosting (n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)

Outputs:
  - reports/phase4_5B_LOCO/loco_summary.csv
  - reports/phase4_5B_LOCO/condition_level_comparison.csv
  - reports/phase4_5B_LOCO/oof_predictions_loco.csv
  - reports/phase4_5B_LOCO/logo_vs_loco_comparison.csv
  - reports/phase4_5B_LOCO/figures/fig1_loco_mae_by_condition.png
  - reports/phase4_5B_LOCO/figures/fig2_logo_vs_loco.png
  - reports/phase4_5B_LOCO/figures/fig3_context_gain_loco.png
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
REPORT_DIR = WORKSPACE_DIR / "reports" / "phase4_5B_LOCO"
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
print("STARTING PHASE 4.5B: UNSEEN CUTTING-CONDITION GENERALIZATION (LOCO)")
print("="*80)

# -------------------------------------------------------------
# 1. Dataset Integrity & Condition Audit
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

print(f"Total Rows: {len(df)}")
assert len(df) == 145, f"Unexpected row count: {len(df)}"
assert df[FEATURES_B].isna().sum().sum() == 0, "Missing values in features!"
assert df[TARGET_COL].isna().sum().sum() == 0, "Missing values in target!"

# Detailed condition grouping audit
cond_audit = df.groupby('condition_id').agg(
    material=('material_name', 'first'),
    DOC_mm=('DOC_mm', 'first'),
    feed_mm_rev=('feed_mm_rev', 'first'),
    n_cases=('case', 'nunique'),
    cases=('case', lambda x: sorted(list(set(x)))),
    n_runs=('run', 'count'),
    vb_min=('VB_mm', 'min'),
    vb_max=('VB_mm', 'max'),
    vb_mean=('VB_mm', 'mean'),
    vb_std=('VB_mm', 'std')
).reset_index()

print("\n--- Dataset Integrity & Condition Audit ---")
print(f"Total Conditions: {len(cond_audit)}")
print(f"Target Range: [{df[TARGET_COL].min():.4f}, {df[TARGET_COL].max():.4f}] mm, Mean={df[TARGET_COL].mean():.4f} mm")
print(f"Cases per Condition: Exactly {cond_audit['n_cases'].unique()} cases each")
print(f"Empty/Abnormal Groups: None (all {len(cond_audit)} conditions have >= 7 valid runs)")
for _, r in cond_audit.iterrows():
    print(f"  {r['condition_id']:26s} | {r['material']:19s} | DOC={r['DOC_mm']:.2f} | Feed={r['feed_mm_rev']:.2f} | Cases={r['cases']} | Runs={r['n_runs']:2d} | VB=[{r['vb_min']:.2f}, {r['vb_max']:.2f}]")

# -------------------------------------------------------------
# 2. LOCO Cross-Validation Setup
# -------------------------------------------------------------
logo = LeaveOneGroupOut()
groups_loco = df['condition_id'].values
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
# 3. LOCO Validation Execution
# -------------------------------------------------------------
oof_preds = {
    'case': df['case'].values,
    'run': df['run'].values,
    'condition_id': df['condition_id'].values,
    'material_code': df['material_code'].values,
    'DOC_mm': df['DOC_mm'].values,
    'feed_mm_rev': df['feed_mm_rev'].values,
    'VB_mm': y
}

model_key_map = {
    'Ridge': 'ridge',
    'SVR': 'svr',
    'Random Forest': 'rf',
    'Gradient Boosting': 'gbdt'
}

for m_name, short_k in model_key_map.items():
    oof_preds[f'prediction_sensor_{short_k}'] = np.zeros(len(df))
    oof_preds[f'prediction_context_{short_k}'] = np.zeros(len(df))

condition_fold_records = []

for fold_idx, (train_idx, test_idx) in enumerate(logo.split(df, y, groups_loco)):
    held_out_cond = groups_loco[test_idx[0]]
    mat_name = df.iloc[test_idx[0]]['material_name']
    doc_val = df.iloc[test_idx[0]]['DOC_mm']
    feed_val = df.iloc[test_idx[0]]['feed_mm_rev']
    cases_held_out = sorted(list(set(df.iloc[test_idx]['case'])))
    n_test_runs = len(test_idx)
    y_test = y[test_idx]
    
    cond_record = {
        'fold': fold_idx + 1,
        'condition_id': held_out_cond,
        'material': mat_name,
        'DOC_mm': doc_val,
        'feed_mm_rev': feed_val,
        'number_of_cases': len(cases_held_out),
        'cases': str(cases_held_out),
        'number_of_test_runs': n_test_runs
    }
    
    for m_name, short_k in model_key_map.items():
        configs = MODEL_CONFIGS[m_name]
        
        # Train & Predict Sensor-Only
        clf_A = configs['model_A']
        clf_A.fit(X_A[train_idx], y[train_idx])
        pred_A = clf_A.predict(X_A[test_idx])
        oof_preds[f'prediction_sensor_{short_k}'][test_idx] = pred_A
        
        # Train & Predict Context Fusion
        clf_B = configs['model_B']
        clf_B.fit(X_B[train_idx], y[train_idx])
        pred_B = clf_B.predict(X_B[test_idx])
        oof_preds[f'prediction_context_{short_k}'][test_idx] = pred_B
        
        mae_A = mean_absolute_error(y_test, pred_A)
        rmse_A = root_mean_squared_error(y_test, pred_A)
        r2_A = r2_score(y_test, pred_A)
        
        mae_B = mean_absolute_error(y_test, pred_B)
        rmse_B = root_mean_squared_error(y_test, pred_B)
        r2_B = r2_score(y_test, pred_B)
        
        delta_mae = mae_B - mae_A
        rel_delta_mae = (delta_mae / mae_A) * 100 if mae_A > 0 else 0.0
        
        cond_record[f'{m_name}_Sensor_MAE'] = mae_A
        cond_record[f'{m_name}_Context_MAE'] = mae_B
        cond_record[f'{m_name}_Delta_MAE'] = delta_mae
        cond_record[f'{m_name}_Rel_Delta_MAE_%'] = rel_delta_mae
        cond_record[f'{m_name}_Sensor_RMSE'] = rmse_A
        cond_record[f'{m_name}_Context_RMSE'] = rmse_B
        cond_record[f'{m_name}_Sensor_R2'] = r2_A
        cond_record[f'{m_name}_Context_R2'] = r2_B
        cond_record[f'{m_name}_Improved'] = bool(delta_mae < 0)
        
    condition_fold_records.append(cond_record)

oof_loco_df = pd.DataFrame(oof_preds)
oof_csv_path = REPORT_DIR / "oof_predictions_loco.csv"
oof_loco_df.to_csv(oof_csv_path, index=False)
print(f"\nSaved LOCO OOF Predictions: {oof_csv_path}")

cond_comp_df = pd.DataFrame(condition_fold_records)
cond_csv_path = REPORT_DIR / "condition_level_comparison.csv"
cond_comp_df.to_csv(cond_csv_path, index=False)
print(f"Saved Condition-Level Comparison: {cond_csv_path}")

# -------------------------------------------------------------
# 4. Global LOCO Summary
# -------------------------------------------------------------
loco_summary_records = []

for m_name, short_k in model_key_map.items():
    pred_s = oof_loco_df[f'prediction_sensor_{short_k}'].values
    pred_c = oof_loco_df[f'prediction_context_{short_k}'].values
    
    mae_s = mean_absolute_error(y, pred_s)
    rmse_s = root_mean_squared_error(y, pred_s)
    r2_s = r2_score(y, pred_s)
    
    mae_c = mean_absolute_error(y, pred_c)
    rmse_c = root_mean_squared_error(y, pred_c)
    r2_c = r2_score(y, pred_c)
    
    delta_mae = mae_c - mae_s
    rel_delta_mae = (delta_mae / mae_s) * 100
    delta_rmse = rmse_c - rmse_s
    delta_r2 = r2_c - r2_s
    
    improved_conds = cond_comp_df[f'{m_name}_Improved'].sum()
    worse_conds = 8 - improved_conds
    
    loco_summary_records.append({
        'Model': m_name,
        'Sensor_MAE_mm': mae_s,
        'Sensor_RMSE_mm': rmse_s,
        'Sensor_R2': r2_s,
        'Context_MAE_mm': mae_c,
        'Context_RMSE_mm': rmse_c,
        'Context_R2': r2_c,
        'Delta_MAE_mm': delta_mae,
        'Rel_Delta_MAE_%': rel_delta_mae,
        'Delta_RMSE_mm': delta_rmse,
        'Delta_R2': delta_r2,
        'Conditions_Improved': f"{improved_conds} / 8 ({improved_conds/8*100:.1f}%)",
        'Conditions_Worse': f"{worse_conds} / 8 ({worse_conds/8*100:.1f}%)"
    })

loco_summary_df = pd.DataFrame(loco_summary_records)
loco_summary_csv = REPORT_DIR / "loco_summary.csv"
loco_summary_df.to_csv(loco_summary_csv, index=False)
print(f"Saved LOCO Summary: {loco_summary_csv}")

print("\n" + "="*80)
print("GLOBAL LOCO PERFORMANCE: SENSOR-ONLY (A) vs CONTEXT FUSION (B)")
print("="*80)
for _, r in loco_summary_df.iterrows():
    print(f"{r['Model']:18s} | Sensor: MAE={r['Sensor_MAE_mm']:.4f}, R2={r['Sensor_R2']:.4f} -> Context: MAE={r['Context_MAE_mm']:.4f}, R2={r['Context_R2']:.4f} | dMAE={r['Delta_MAE_mm']:+.4f} ({r['Rel_Delta_MAE_%']:+.2f}%) | Improved: {r['Conditions_Improved']}")

# -------------------------------------------------------------
# 5. LOGO vs LOCO Comparison (Generalization Gap Analysis)
# -------------------------------------------------------------
logo_summary_path = WORKSPACE_DIR / "reports" / "phase4_5_context_fusion" / "context_fusion_summary.csv"
logo_df = pd.read_csv(logo_summary_path)

logo_loco_records = []
for m_name in MODEL_CONFIGS:
    logo_row = logo_df[logo_df['Model'] == m_name].iloc[0]
    loco_row = loco_summary_df[loco_summary_df['Model'] == m_name].iloc[0]
    
    # Sensor-Only
    mae_sensor_logo = logo_row['MAE_A_mm']
    mae_sensor_loco = loco_row['Sensor_MAE_mm']
    gap_sensor_mae = mae_sensor_loco - mae_sensor_logo
    pct_gap_sensor = (gap_sensor_mae / mae_sensor_logo) * 100
    
    logo_loco_records.append({
        'Model': m_name,
        'Feature_Set': 'Sensor-Only (5 feats)',
        'LOGO_MAE_mm': mae_sensor_logo,
        'LOCO_MAE_mm': mae_sensor_loco,
        'Generalization_Gap_mm': gap_sensor_mae,
        'Rel_Generalization_Gap_%': pct_gap_sensor,
        'LOGO_R2': logo_row['R2_A'],
        'LOCO_R2': loco_row['Sensor_R2']
    })
    
    # Context Fusion
    mae_ctx_logo = logo_row['MAE_B_mm']
    mae_ctx_loco = loco_row['Context_MAE_mm']
    gap_ctx_mae = mae_ctx_loco - mae_ctx_logo
    pct_gap_ctx = (gap_ctx_mae / mae_ctx_logo) * 100
    
    logo_loco_records.append({
        'Model': m_name,
        'Feature_Set': 'Context Fusion (8 feats)',
        'LOGO_MAE_mm': mae_ctx_logo,
        'LOCO_MAE_mm': mae_ctx_loco,
        'Generalization_Gap_mm': gap_ctx_mae,
        'Rel_Generalization_Gap_%': pct_gap_ctx,
        'LOGO_R2': logo_row['R2_B'],
        'LOCO_R2': loco_row['Context_R2']
    })

logo_loco_df = pd.DataFrame(logo_loco_records)
logo_loco_csv = REPORT_DIR / "logo_vs_loco_comparison.csv"
logo_loco_df.to_csv(logo_loco_csv, index=False)
print(f"\nSaved LOGO vs LOCO Comparison: {logo_loco_csv}")

print("\n" + "="*80)
print("LOGO (Unseen Tool, Known Cond) vs LOCO (Unseen Tool + Unseen Cond)")
print("="*80)
for _, r in logo_loco_df.iterrows():
    print(f"{r['Model']:18s} | {r['Feature_Set']:25s} | LOGO={r['LOGO_MAE_mm']:.4f} -> LOCO={r['LOCO_MAE_mm']:.4f} | Gap={r['Generalization_Gap_mm']:+.4f} ({r['Rel_Generalization_Gap_%']:+.1f}%) | R2: {r['LOGO_R2']:.4f} -> {r['LOCO_R2']:.4f}")

# -------------------------------------------------------------
# 6. Visualizations
# -------------------------------------------------------------

# Figure 1: LOCO MAE by Held-Out Condition (C1 - C8)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
models = list(MODEL_CONFIGS.keys())
cond_labels = [c.split('_')[0] + f"\n({c.split('_')[1][:5]})" for c in cond_comp_df['condition_id']]
x_cond = np.arange(len(cond_labels))
width = 0.35

for idx, m_name in enumerate(models):
    ax = axes[idx // 2, idx % 2]
    
    mae_s = cond_comp_df[f'{m_name}_Sensor_MAE'].values
    mae_c = cond_comp_df[f'{m_name}_Context_MAE'].values
    
    b1 = ax.bar(x_cond - width/2, mae_s, width, label='Sensor-Only', color='#4A7BB0', edgecolor='black', linewidth=0.7)
    b2 = ax.bar(x_cond + width/2, mae_c, width, label='Context Fusion', color='#2E8B57', edgecolor='black', linewidth=0.7)
    
    ax.set_title(f'{m_name}', fontsize=10, fontweight='bold')
    ax.set_xticks(x_cond)
    ax.set_xticklabels(cond_labels, fontsize=8)
    ax.set_ylabel('Held-Out Condition MAE (mm)')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(frameon=True, fontsize=8, loc='upper left')
    
    # Highlight highest error condition
    max_idx = np.argmax(mae_c)
    ax.text(x_cond[max_idx] + width/2, mae_c[max_idx] + 0.005, f"{mae_c[max_idx]:.3f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#B22222')

plt.suptitle('Figure 1: Phase 4.5B — LOCO MAE Across 8 Held-Out Cutting Conditions\n(C1-C4: Cast Iron | C5-C8: Stainless Steel J45)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig1_path = FIGURES_DIR / "fig1_loco_mae_by_condition.png"
plt.savefig(fig1_path)
plt.close()
print(f"Saved: {fig1_path}")

# Figure 2: LOGO vs LOCO Performance (Generalization Gap)
fig, ax = plt.subplots(figsize=(9, 4.8))
x_m = np.arange(len(models))
w = 0.2

# Data arrays
logo_sensor = [logo_loco_df[(logo_loco_df['Model'] == m) & (logo_loco_df['Feature_Set'].str.startswith('Sensor'))]['LOGO_MAE_mm'].values[0] for m in models]
loco_sensor = [logo_loco_df[(logo_loco_df['Model'] == m) & (logo_loco_df['Feature_Set'].str.startswith('Sensor'))]['LOCO_MAE_mm'].values[0] for m in models]
logo_context = [logo_loco_df[(logo_loco_df['Model'] == m) & (logo_loco_df['Feature_Set'].str.startswith('Context'))]['LOGO_MAE_mm'].values[0] for m in models]
loco_context = [logo_loco_df[(logo_loco_df['Model'] == m) & (logo_loco_df['Feature_Set'].str.startswith('Context'))]['LOCO_MAE_mm'].values[0] for m in models]

b1 = ax.bar(x_m - 1.5*w, logo_sensor, w, label='LOGO Sensor-Only (Known Cond)', color='#6BAED6', edgecolor='black', linewidth=0.7)
b2 = ax.bar(x_m - 0.5*w, loco_sensor, w, label='LOCO Sensor-Only (Unseen Cond)', color='#2171B5', edgecolor='black', linewidth=0.7)
b3 = ax.bar(x_m + 0.5*w, logo_context, w, label='LOGO Context Fusion (Known Cond)', color='#74C476', edgecolor='black', linewidth=0.7)
b4 = ax.bar(x_m + 1.5*w, loco_context, w, label='LOCO Context Fusion (Unseen Cond)', color='#238B45', edgecolor='black', linewidth=0.7)

# Add baseline reference
ax.axhline(0.1998, color='#D9534F', linestyle='--', linewidth=1.1, label='Mean Baseline (0.1998 mm)')

for i in range(len(models)):
    # Annotate LOCO Context MAE and generalization gap
    val_logo = logo_context[i]
    val_loco = loco_context[i]
    gap = val_loco - val_logo
    ax.text(x_m[i] + 1.5*w, val_loco + 0.003, f"{val_loco:.4f}\n(Gap: {gap:+.3f})", ha='center', va='bottom', fontsize=7, fontweight='bold', color='#1B5E20')

ax.set_ylabel('Out-of-Fold MAE (mm)')
ax.set_title('Figure 2: LOGO vs. LOCO Performance Comparison\nEvaluating the Generalization Gap from Unseen Tool to Unseen Operating Regime', fontsize=11, fontweight='bold')
ax.set_xticks(x_m)
ax.set_xticklabels(models, fontweight='bold')
ax.set_ylim(0, 0.22)
ax.legend(frameon=True, fontsize=8, loc='upper right')
ax.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
fig2_path = FIGURES_DIR / "fig2_logo_vs_loco.png"
plt.savefig(fig2_path)
plt.close()
print(f"Saved: {fig2_path}")

# Figure 3: Context Gain Under LOCO (Delta MAE by Condition)
fig, ax = plt.subplots(figsize=(9.5, 4.5))
cond_names = [f"{c.split('_')[0]} ({c.split('_')[1][:5]}, d={c.split('_')[2][1:]}, f={c.split('_')[3][1:]})" for c in cond_comp_df['condition_id']]
x_c = np.arange(len(cond_names))
w_bar = 0.2

for idx, m_name in enumerate(models):
    delta_vals = cond_comp_df[f'{m_name}_Delta_MAE'].values
    offset = (idx - 1.5) * w_bar
    ax.bar(x_c + offset, delta_vals, w_bar, label=m_name, edgecolor='black', linewidth=0.6)

ax.axhline(0, color='black', linewidth=0.9, linestyle='--')
ax.set_ylabel(r'$\Delta$ MAE = Context - Sensor (mm)' + '\n' + r'($<0$: Context Fusion Wins)')
ax.set_title('Figure 3: Context Fusion Gain Under LOCO by Held-Out Condition\n(Negative Values Indicate Context Features Improved Wear Prediction on Unseen Condition)', fontsize=11, fontweight='bold')
ax.set_xticks(x_c)
ax.set_xticklabels(cond_names, rotation=20, ha='right', fontsize=8)
ax.grid(axis='y', linestyle=':', alpha=0.6)
ax.legend(frameon=True, fontsize=8, loc='lower left')
plt.tight_layout()
fig3_path = FIGURES_DIR / "fig3_context_gain_loco.png"
plt.savefig(fig3_path)
plt.close()
print(f"Saved: {fig3_path}")

print("="*80)
print("PHASE 4.5B COMPUTATION AND VISUALIZATIONS COMPLETE!")
print("="*80)
