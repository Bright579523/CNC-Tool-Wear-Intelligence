"""
Phase 5.2A: Final Ridge Formulation & Coefficient Audit
NASA Milling Dataset (V2 Flagship Project)

Objectives:
1. Verify preprocessing & feature pipeline.
2. Reproduce locked Context-fused LOGO (16 folds) and Context-fused LOCO (8 folds) Ridge metrics.
3. Fit final Ridge model on all 145 valid samples and extract standardized coefficients for all 8 features.
4. Audit coefficient stability across cross-validation folds.
5. Generate publication-quality horizontal bar chart of final coefficients.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
REPORTS_DIR = WORKSPACE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plot styling
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
print("PHASE 5.2A: FINAL RIDGE FORMULATION & COEFFICIENT AUDIT")
print("="*80)

# 1. Load Dataset
df = pd.read_csv(DATA_DIR / "feature_dataset_v32.csv")
print(f"Loaded feature_dataset_v32.csv: {len(df)} rows, {df['case'].nunique()} tool cases, {df['condition_id'].nunique()} conditions.")

SENSOR_FEATURES = [
    'smcAC_rms',
    'vib_spindle_kurtosis',
    'vib_spindle_p2p',
    'AE_table_rms',
    'AE_spindle_p2p'
]

CONTEXT_FEATURES = [
    'material_code',
    'DOC_mm',
    'feed_mm_rev'
]

ALL_FEATURES = SENSOR_FEATURES + CONTEXT_FEATURES
TARGET_COL = 'VB_mm'

print("\nFeatures Verified:")
print(f"  Sensor Features ({len(SENSOR_FEATURES)}): {SENSOR_FEATURES}")
print(f"  Context Features ({len(CONTEXT_FEATURES)}): {CONTEXT_FEATURES}")
print(f"  Target: {TARGET_COL}")

# Audit material_code encoding
mat_audit = df.groupby(['material_code', 'material_name']).size().reset_index(name='count')
print("\nMaterial Encoding Audit:")
for _, row in mat_audit.iterrows():
    print(f"  material_code = {row['material_code']} -> {row['material_name']} ({row['count']} runs)")

# 2. Benchmark Reproduction: Context LOGO (16 Folds)
logo = LeaveOneGroupOut()
logo_groups = df['case'].values
X_all = df[ALL_FEATURES].values
y = df[TARGET_COL].values

logo_preds = np.zeros(len(df))
logo_coefs = []

for train_idx, test_idx in logo.split(X_all, y, groups=logo_groups):
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    logo_preds[test_idx] = pipe.predict(X_test)
    logo_coefs.append(pipe.named_steps['model'].coef_)

logo_mae = mean_absolute_error(y, logo_preds)
logo_rmse = root_mean_squared_error(y, logo_preds)
logo_r2 = r2_score(y, logo_preds)

print("\n" + "="*80)
print("BENCHMARK REPRODUCTION: CONTEXT LOGO (16 FOLDS)")
print("="*80)
print(f"  Reproduced LOGO MAE  : {logo_mae:.4f} mm (Expected locked: 0.1044 mm)")
print(f"  Reproduced LOGO RMSE : {logo_rmse:.4f} mm (Expected locked: 0.1425 mm)")
print(f"  Reproduced LOGO R2   : {logo_r2:.4f}    (Expected locked: 0.6989)")

# 3. Benchmark Reproduction: Context LOCO (8 Folds)
loco = LeaveOneGroupOut()
loco_groups = df['condition_id'].values

loco_preds = np.zeros(len(df))
loco_coefs = []

for train_idx, test_idx in loco.split(X_all, y, groups=loco_groups):
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    loco_preds[test_idx] = pipe.predict(X_test)
    loco_coefs.append(pipe.named_steps['model'].coef_)

loco_mae = mean_absolute_error(y, loco_preds)
loco_rmse = root_mean_squared_error(y, loco_preds)
loco_r2 = r2_score(y, loco_preds)

print("\n" + "="*80)
print("BENCHMARK REPRODUCTION: CONTEXT LOCO (8 FOLDS)")
print("="*80)
print(f"  Reproduced LOCO MAE  : {loco_mae:.4f} mm (Expected locked: 0.1221 mm)")
print(f"  Reproduced LOCO RMSE : {loco_rmse:.4f} mm (Expected locked: 0.1566 mm)")
print(f"  Reproduced LOCO R2   : {loco_r2:.4f}    (Expected locked: 0.6360)")

# 4. Final Full-Dataset Ridge Fit (Standardized Coefficients)
scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X_all)

ridge_full = Ridge(alpha=1.0, random_state=42)
ridge_full.fit(X_scaled, y)

coef_full = ridge_full.coef_
intercept_full = ridge_full.intercept_

# Check mean coefs across LOGO and LOCO folds
logo_coefs_mean = np.mean(logo_coefs, axis=0)
logo_coefs_std = np.std(logo_coefs, axis=0)
loco_coefs_mean = np.mean(loco_coefs, axis=0)
loco_coefs_std = np.std(loco_coefs, axis=0)

# Also fit sensor-only model on full dataset for explicit comparison
scaler_sensor = StandardScaler()
X_sensor_scaled = scaler_sensor.fit_transform(df[SENSOR_FEATURES].values)
ridge_sensor = Ridge(alpha=1.0, random_state=42)
ridge_sensor.fit(X_sensor_scaled, y)
sensor_only_coefs = ridge_sensor.coef_

coef_table = []
for i, f_name in enumerate(ALL_FEATURES):
    f_type = "Context" if f_name in CONTEXT_FEATURES else "Sensor"
    coef_val = coef_full[i]
    coef_table.append({
        'Feature': f_name,
        'Type': f_type,
        'Standardized_Coefficient': coef_val,
        'Sign': '+' if coef_val > 0 else '-',
        'Absolute_Coefficient': abs(coef_val),
        'LOGO_Fold_Mean': logo_coefs_mean[i],
        'LOGO_Fold_Std': logo_coefs_std[i],
        'LOCO_Fold_Mean': loco_coefs_mean[i],
        'LOCO_Fold_Std': loco_coefs_std[i]
    })

coef_df = pd.DataFrame(coef_table)
coef_df = coef_df.sort_values(by='Absolute_Coefficient', ascending=False).reset_index(drop=True)
coef_df['Rank'] = np.arange(1, len(coef_df) + 1)

print("\n" + "="*80)
print("FINAL RIDGE COEFFICIENTS (FULL DATASET, N=145, STANDARDIZED)")
print("="*80)
print(f"Intercept (beta_0): {intercept_full:+.4f} mm")
print(f"{'Rank':<5} {'Feature':<22} {'Type':<8} {'Coefficient':<14} {'Abs Coef':<10} {'LOGO Mean +/- Std':<20} {'LOCO Mean +/- Std'}")
print("-"*85)
for _, r in coef_df.iterrows():
    print(f"{r['Rank']:<5} {r['Feature']:<22} {r['Type']:<8} {r['Standardized_Coefficient']:>+10.4f}     {r['Absolute_Coefficient']:>8.4f}   {r['LOGO_Fold_Mean']:>+7.4f} +/- {r['LOGO_Fold_Std']:.4f}    {r['LOCO_Fold_Mean']:>+7.4f} +/- {r['LOCO_Fold_Std']:.4f}")

# Compare sensor features in sensor-only vs context-fused
print("\n" + "="*80)
print("COMPARISON: SENSOR-ONLY VS CONTEXT-FUSED RIDGE COEFFICIENTS")
print("="*80)
print(f"{'Feature':<22} {'Sensor-Only Coef':<18} {'Context-Fused Coef':<18} {'Delta Coef'}")
print("-"*65)
for i, f_name in enumerate(SENSOR_FEATURES):
    s_coef = sensor_only_coefs[i]
    c_idx = ALL_FEATURES.index(f_name)
    c_coef = coef_full[c_idx]
    delta = c_coef - s_coef
    print(f"{f_name:<22} {s_coef:>+14.4f}     {c_coef:>+14.4f}     {delta:>+10.4f}")

# Save CSV of coefficients
out_csv = REPORTS_DIR / "phase5_final_ridge_coefficients.csv"
coef_df.to_csv(out_csv, index=False)
print(f"\nSaved coefficient audit table to: {out_csv}")

# 5. Generate Publication-Quality Horizontal Bar Chart
fig, ax = plt.subplots(figsize=(9.2, 5.0))

# Sort by coefficient value (from negative to positive for clean visual layout)
plot_df = coef_df.sort_values(by='Standardized_Coefficient', ascending=True).reset_index(drop=True)

y_pos = np.arange(len(plot_df))
bars = ax.barh(y_pos, plot_df['Standardized_Coefficient'], height=0.55,
               color=['#D9534F' if c < 0 else '#4A7BB0' for c in plot_df['Standardized_Coefficient']],
               edgecolor='black', linewidth=0.8)

# Feature label customization (tagging Sensor vs Context)
y_labels = [f"{r['Feature']} ({r['Type']})" for _, r in plot_df.iterrows()]
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=9, fontweight='bold')

# Add zero line
ax.axvline(0, color='black', linewidth=1.0, linestyle='--')

# Value annotations
for i, b in enumerate(bars):
    val = plot_df.loc[i, 'Standardized_Coefficient']
    x_offset = 0.008 if val >= 0 else -0.008
    ha = 'left' if val >= 0 else 'right'
    ax.text(val + x_offset, b.get_y() + b.get_height()/2, f"{val:+.4f}",
            va='center', ha=ha, fontsize=8.5, fontweight='bold',
            color='#222222')

ax.set_xlabel('Standardized Ridge Coefficient (β)', fontsize=10, fontweight='bold')
ax.set_title('Final Context-Fused Ridge Regression Coefficients (alpha=1.0, N=145)\n[Sensor Features + Operating Context Features in Standardized Space]', fontsize=11, fontweight='bold', pad=12)
ax.set_xlim(-0.22, 0.36)
ax.grid(axis='x', linestyle=':', alpha=0.6)

# Legend elements
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4A7BB0', edgecolor='black', label='Positive Weight (Increases Estimated VB)'),
    Patch(facecolor='#D9534F', edgecolor='black', label='Negative Weight (Decreases Estimated VB)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8.5, frameon=True, framealpha=0.9)

plt.tight_layout()
fig_path = FIGURES_DIR / "fig_phase5_2a_ridge_coefficients.png"
plt.savefig(fig_path, bbox_inches='tight')
plt.close()
print(f"Saved coefficient figure to: {fig_path}")

print("="*80)
print("PHASE 5.2A SCRIPT EXECUTION COMPLETE")
print("="*80)
