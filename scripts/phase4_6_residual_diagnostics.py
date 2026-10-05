"""
Phase 4.6: Residual Diagnostics, Error Regime Decomposition & Failure Mode Analysis
NASA Milling Dataset (V2 Flagship Project)

Research Questions:
1. Where and why do the models make errors across the tool wear life cycle?
2. How does performance vary across Fresh (<0.20 mm), Moderate (0.20-0.40 mm),
   and Severe (>0.40 mm) wear regimes?
3. What is the extent of residual bias (overprediction on fresh tools vs. underprediction
   on severe wear)?
4. How much of the Stainless Steel error is driven by the extreme wear trajectory of Case 13?
5. What are the core failure modes and boundaries of the Phase 4 ML pipeline?

Inputs:
  - reports/phase4_5_context_fusion/oof_predictions_context.csv (LOGO Predictions)
  - reports/phase4_5B_LOCO/oof_predictions_loco.csv (LOCO Predictions)

Outputs:
  - reports/phase4_6_diagnostics/wear_regime_metrics.csv
  - reports/phase4_6_diagnostics/residual_bias_summary.csv
  - reports/phase4_6_diagnostics/case_level_residual_audit.csv
  - reports/phase4_6_diagnostics/figures/fig1_residual_vs_actual_vb.png
  - reports/phase4_6_diagnostics/figures/fig2_wear_regime_mae_bias.png
  - reports/phase4_6_diagnostics/figures/fig3_error_distribution_by_material.png
  - reports/phase4_6_diagnostics/figures/fig4_case13_deep_dive.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# -------------------------------------------------------------
# Configuration and Directories
# -------------------------------------------------------------
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = WORKSPACE_DIR / "reports"
DIAG_DIR = REPORTS_DIR / "phase4_6_diagnostics"
FIGURES_DIR = DIAG_DIR / "figures"
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
print("STARTING PHASE 4.6: RESIDUAL DIAGNOSTICS & FAILURE MODE ANALYSIS")
print("="*80)

# -------------------------------------------------------------
# 1. Load OOF Predictions Dataset
# -------------------------------------------------------------
logo_pred_path = REPORTS_DIR / "phase4_5_context_fusion" / "oof_predictions_context.csv"
df = pd.read_csv(logo_pred_path)
print(f"Loaded Phase 4.5A LOGO Predictions: {len(df)} rows across {df['case'].nunique()} tools.")

# Categorize wear into 3 lifecycle regimes:
# 1. Fresh / Initial wear: VB < 0.20 mm
# 2. Moderate / Steady-state wear: 0.20 <= VB <= 0.40 mm (ISO 0.30 mm tool life window)
# 3. Severe / Tertiary wear: VB > 0.40 mm
def categorize_regime(vb):
    if vb < 0.20:
        return '1_Fresh (<0.20mm)'
    elif vb <= 0.40:
        return '2_Moderate (0.20-0.40mm)'
    else:
        return '3_Severe (>0.40mm)'

df['regime'] = df['VB_mm_actual'].apply(categorize_regime)

MODELS = {
    'Ridge': 'Ridge_B_pred',
    'SVR': 'SVR_B_pred',
    'Random Forest': 'Random Forest_B_pred',
    'Gradient Boosting': 'Gradient Boosting_B_pred'
}

# Calculate residuals: e = y_pred - y_actual
# positive = overprediction, negative = underprediction
for m_name, col in MODELS.items():
    df[f'{m_name}_residual'] = df[col] - df['VB_mm_actual']
    df[f'{m_name}_abs_error'] = np.abs(df[f'{m_name}_residual'])

# -------------------------------------------------------------
# 2. Wear Regime Decomposition Analysis
# -------------------------------------------------------------
regimes = sorted(df['regime'].unique())
regime_records = []

for r in regimes:
    sub = df[df['regime'] == r]
    n_samples = len(sub)
    pct_samples = (n_samples / len(df)) * 100
    actual_mean = sub['VB_mm_actual'].mean()
    actual_min = sub['VB_mm_actual'].min()
    actual_max = sub['VB_mm_actual'].max()
    
    for m_name, col in MODELS.items():
        actual = sub['VB_mm_actual'].values
        pred = sub[col].values
        res = sub[f'{m_name}_residual'].values
        
        mae = mean_absolute_error(actual, pred)
        rmse = root_mean_squared_error(actual, pred)
        bias = np.mean(res)
        std_res = np.std(res)
        
        pct_over = np.mean(res > 0) * 100
        pct_under = np.mean(res < 0) * 100
        max_over = np.max(res)
        max_under = np.min(res)
        
        regime_records.append({
            'Regime': r,
            'Model': m_name,
            'Sample_Count': n_samples,
            'Sample_Pct': pct_samples,
            'Actual_Mean_mm': actual_mean,
            'Actual_Range_mm': f"[{actual_min:.2f}, {actual_max:.2f}]",
            'MAE_mm': mae,
            'RMSE_mm': rmse,
            'Mean_Bias_mm': bias,
            'Std_Residual_mm': std_res,
            'Pct_Overpredicted': pct_over,
            'Pct_Underpredicted': pct_under,
            'Max_Overpredict_mm': max_over,
            'Max_Underpredict_mm': max_under,
            'Pred_Range_mm': f"[{pred.min():.2f}, {pred.max():.2f}]"
        })

regime_df = pd.DataFrame(regime_records)
regime_csv_path = DIAG_DIR / "wear_regime_metrics.csv"
regime_df.to_csv(regime_csv_path, index=False)
print(f"Saved Wear Regime Metrics: {regime_csv_path}")

print("\n" + "="*80)
print("WEAR REGIME PERFORMANCE & BIAS BREAKDOWN")
print("="*80)
for r in regimes:
    sub_r = regime_df[regime_df['Regime'] == r]
    first = sub_r.iloc[0]
    print(f"\n--- {r}: n={first['Sample_Count']} ({first['Sample_Pct']:.1f}%), Actual Range={first['Actual_Range_mm']}, Mean={first['Actual_Mean_mm']:.3f} mm ---")
    for _, row in sub_r.iterrows():
        print(f"  {row['Model']:18s} | MAE={row['MAE_mm']:.4f} mm | Bias={row['Mean_Bias_mm']:+.4f} mm | Over={row['Pct_Overpredicted']:.1f}% | Pred Range={row['Pred_Range_mm']}")

# -------------------------------------------------------------
# 3. Residual Bias & Heteroscedasticity Analysis
# -------------------------------------------------------------
bias_records = []
for m_name, col in MODELS.items():
    actual = df['VB_mm_actual'].values
    pred = df[col].values
    res = df[f'{m_name}_residual'].values
    abs_err = df[f'{m_name}_abs_error'].values
    
    # Linear regression of residual vs actual: e = slope * actual + intercept
    # A negative slope confirms regression-to-the-mean (overpredicting low, underpredicting high)
    slope, intercept, r_val, p_val, std_err = stats.linregress(actual, res)
    
    # Correlation between absolute error and actual wear (Heteroscedasticity check)
    corr_hetero, p_hetero = stats.pearsonr(actual, abs_err)
    
    bias_records.append({
        'Model': m_name,
        'Overall_MAE_mm': mean_absolute_error(actual, pred),
        'Overall_Bias_mm': np.mean(res),
        'Residual_Slope_vs_Actual': slope,
        'Residual_Intercept': intercept,
        'Residual_R2': r_val**2,
        'Residual_P_Value': p_val,
        'Heteroscedasticity_Corr': corr_hetero,
        'Heteroscedasticity_P_Value': p_hetero,
        'Regime1_Fresh_Bias': regime_df[(regime_df['Model'] == m_name) & (regime_df['Regime'].str.startswith('1'))]['Mean_Bias_mm'].values[0],
        'Regime2_Mod_Bias': regime_df[(regime_df['Model'] == m_name) & (regime_df['Regime'].str.startswith('2'))]['Mean_Bias_mm'].values[0],
        'Regime3_Sev_Bias': regime_df[(regime_df['Model'] == m_name) & (regime_df['Regime'].str.startswith('3'))]['Mean_Bias_mm'].values[0]
    })

bias_df = pd.DataFrame(bias_records)
bias_csv_path = DIAG_DIR / "residual_bias_summary.csv"
bias_df.to_csv(bias_csv_path, index=False)
print(f"Saved Residual Bias Summary: {bias_csv_path}")

print("\n" + "="*80)
print("RESIDUAL BIAS & HETEROSCEDASTICITY DIAGNOSTICS")
print("="*80)
for _, r in bias_df.iterrows():
    print(f"{r['Model']:18s} | Residual Slope vs Actual: {r['Residual_Slope_vs_Actual']:+.4f} (p={r['Residual_P_Value']:.2e}) | Heteroscedasticity r={r['Heteroscedasticity_Corr']:+.4f} | Fresh Bias={r['Regime1_Fresh_Bias']:+.4f} | Severe Bias={r['Regime3_Sev_Bias']:+.4f}")

# -------------------------------------------------------------
# 4. Case-by-Case Diagnostic & Outlier Audit
# -------------------------------------------------------------
case_records = []
cases = sorted(df['case'].unique())

for case_id in cases:
    case_sub = df[df['case'] == case_id]
    mat = case_sub['material_name'].iloc[0]
    cond = case_sub['condition_id'].iloc[0]
    n_runs = len(case_sub)
    actual_vb = case_sub['VB_mm_actual'].values
    max_vb = np.max(actual_vb)
    
    row = {
        'case': case_id,
        'material': mat,
        'condition_id': cond,
        'n_runs': n_runs,
        'actual_max_VB_mm': max_vb,
        'actual_mean_VB_mm': np.mean(actual_vb)
    }
    
    for m_name, col in MODELS.items():
        pred_vb = case_sub[col].values
        mae = mean_absolute_error(actual_vb, pred_vb)
        bias = np.mean(case_sub[f'{m_name}_residual'])
        max_err = np.max(case_sub[f'{m_name}_abs_error'])
        
        row[f'{m_name}_MAE_mm'] = mae
        row[f'{m_name}_Bias_mm'] = bias
        row[f'{m_name}_Max_Abs_Error_mm'] = max_err
        
    case_records.append(row)

case_df = pd.DataFrame(case_records)
case_csv_path = DIAG_DIR / "case_level_residual_audit.csv"
case_df.to_csv(case_csv_path, index=False)
print(f"Saved Case-Level Residual Audit: {case_csv_path}")

# Forensic on Case 13 Impact
ss_df = df[df['material_name'] == 'Stainless Steel J45']
c13_df = df[df['case'] == 13]
non_c13_df = ss_df[ss_df['case'] != 13]

print("\n" + "="*80)
print("CASE 13 IMPACT ON STAINLESS STEEL PREDICTION ERROR")
print("="*80)
for m_name, col in MODELS.items():
    mae_all = mean_absolute_error(ss_df['VB_mm_actual'], ss_df[col])
    mae_c13 = mean_absolute_error(c13_df['VB_mm_actual'], c13_df[col])
    mae_ex_c13 = mean_absolute_error(non_c13_df['VB_mm_actual'], non_c13_df[col])
    reduction_pct = (mae_all - mae_ex_c13) / mae_all * 100
    print(f"{m_name:18s} | All Stainless (n=48): {mae_all:.4f} mm | Case 13 only (n=13): {mae_c13:.4f} mm | Remaining 7 tools (n=35): {mae_ex_c13:.4f} mm (Impact: -{reduction_pct:.1f}%)")

# -------------------------------------------------------------
# 5. Visualizations
# -------------------------------------------------------------

# Figure 1: Residual vs Actual VB Scatter Plots with Trendlines
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
models_list = list(MODELS.keys())

for idx, m_name in enumerate(models_list):
    ax = axes[idx // 2, idx % 2]
    
    ci_mask = df['material_name'] == 'Cast Iron'
    ss_mask = df['material_name'] == 'Stainless Steel J45'
    
    # Scatter points colored by material
    ax.scatter(df.loc[ci_mask, 'VB_mm_actual'], df.loc[ci_mask, f'{m_name}_residual'], 
               color='#4A7BB0', alpha=0.75, s=30, label='Cast Iron (n=97)', edgecolors='none')
    ax.scatter(df.loc[ss_mask, 'VB_mm_actual'], df.loc[ss_mask, f'{m_name}_residual'], 
               color='#D9534F', alpha=0.75, s=35, label='Stainless Steel J45 (n=48)', marker='^', edgecolors='black', linewidths=0.5)
    
    # Zero line
    ax.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    
    # Regression trendline showing bias slope
    actual = df['VB_mm_actual'].values
    res = df[f'{m_name}_residual'].values
    slope, intercept, r_val, p_val, _ = stats.linregress(actual, res)
    x_line = np.linspace(0, 1.55, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='#2E8B57', linewidth=1.8, label=f'Trend (Slope={slope:+.2f})')
    
    # Shaded lifecycle zones
    ax.axvspan(0, 0.20, color='#E8F5E9', alpha=0.35, zorder=-1)
    ax.axvspan(0.20, 0.40, color='#FFF9C4', alpha=0.35, zorder=-1)
    ax.axvspan(0.40, 1.55, color='#FFEBEE', alpha=0.35, zorder=-1)
    
    ax.set_title(f'{m_name} (OOF LOGO Context Model)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Residual: Pred - Actual (mm)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', fontsize=7.5, frameon=True)
    
    # Text annotation for zones
    if idx == 0:
        ax.text(0.10, 0.50, 'Fresh\n(<0.20)', ha='center', fontsize=7.5, color='#2E7D32', fontweight='bold')
        ax.text(0.30, 0.50, 'Moderate\n(0.20-0.40)', ha='center', fontsize=7.5, color='#F57F17', fontweight='bold')
        ax.text(0.95, 0.50, 'Severe (>0.40 mm)', ha='center', fontsize=7.5, color='#C62828', fontweight='bold')

for ax in axes[1, :]:
    ax.set_xlabel('Actual Flank Wear VB (mm)')

plt.suptitle('Figure 1: Residual Diagnostics vs. Actual Tool Wear\nDemonstrating Systematic Underprediction on Severe Wear and Overprediction on Fresh Tools', fontsize=12, fontweight='bold')
plt.tight_layout()
fig1_path = FIGURES_DIR / "fig1_residual_vs_actual_vb.png"
plt.savefig(fig1_path)
plt.close()
print(f"Saved: {fig1_path}")

# Figure 2: Wear Regime MAE and Bias Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
x = np.arange(len(models_list))
w = 0.25

regime_labels = ['Fresh (<0.20mm)', 'Moderate (0.20-0.40mm)', 'Severe (>0.40mm)']
regime_colors_mae = ['#6BAED6', '#74C476', '#FD8D3C']
regime_colors_bias = ['#4A7BB0', '#2E8B57', '#D9534F']

# MAE by regime
for i, r in enumerate(regimes):
    mae_vals = [regime_df[(regime_df['Model'] == m) & (regime_df['Regime'] == r)]['MAE_mm'].values[0] for m in models_list]
    bars = ax1.bar(x + (i - 1)*w, mae_vals, w, label=regime_labels[i], color=regime_colors_mae[i], edgecolor='black', linewidth=0.7)
    for b in bars:
        yval = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, yval + 0.003, f"{yval:.3f}", ha='center', va='bottom', fontsize=7)

ax1.set_title('MAE by Wear Lifecycle Regime', fontsize=10, fontweight='bold')
ax1.set_ylabel('MAE (mm)')
ax1.set_xticks(x)
ax1.set_xticklabels(models_list, fontweight='bold')
ax1.set_ylim(0, 0.23)
ax1.grid(axis='y', linestyle=':', alpha=0.6)
ax1.legend(fontsize=8, frameon=True, loc='upper left')

# Bias by regime
for i, r in enumerate(regimes):
    bias_vals = [regime_df[(regime_df['Model'] == m) & (regime_df['Regime'] == r)]['Mean_Bias_mm'].values[0] for m in models_list]
    bars = ax2.bar(x + (i - 1)*w, bias_vals, w, label=regime_labels[i], color=regime_colors_bias[i], edgecolor='black', linewidth=0.7)
    for b in bars:
        yval = b.get_height()
        va = 'bottom' if yval >= 0 else 'top'
        offset = 0.004 if yval >= 0 else -0.012
        ax2.text(b.get_x() + b.get_width()/2, yval + offset, f"{yval:+.3f}", ha='center', va=va, fontsize=7)

ax2.axhline(0, color='black', linewidth=0.9, linestyle='--')
ax2.set_title('Mean Prediction Bias by Wear Regime\n(+ = Overprediction, - = Underprediction)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Mean Bias: Pred - Actual (mm)')
ax2.set_xticks(x)
ax2.set_xticklabels(models_list, fontweight='bold')
ax2.set_ylim(-0.16, 0.09)
ax2.grid(axis='y', linestyle=':', alpha=0.6)
ax2.legend(fontsize=8, frameon=True, loc='lower left')

plt.suptitle('Figure 2: Performance and Prediction Bias Across Wear Regimes', fontsize=12, fontweight='bold')
plt.tight_layout()
fig2_path = FIGURES_DIR / "fig2_wear_regime_mae_bias.png"
plt.savefig(fig2_path)
plt.close()
print(f"Saved: {fig2_path}")

# Figure 3: Error Distributions by Workpiece Material
fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)

for idx, m_name in enumerate(models_list):
    ax = axes[idx]
    
    ci_res = df.loc[df['material_name'] == 'Cast Iron', f'{m_name}_residual'].values
    ss_res = df.loc[df['material_name'] == 'Stainless Steel J45', f'{m_name}_residual'].values
    
    ax.hist(ci_res, bins=12, density=True, alpha=0.55, color='#4A7BB0', edgecolor='black', label=f'Cast Iron (std={np.std(ci_res):.3f})')
    ax.hist(ss_res, bins=12, density=True, alpha=0.55, color='#D9534F', edgecolor='black', label=f'Stainless (std={np.std(ss_res):.3f})')
    
    ax.axvline(0, color='black', linestyle='--', linewidth=0.9)
    ax.set_title(f'{m_name}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Residual: Pred - Actual (mm)')
    if idx == 0:
        ax.set_ylabel('Density')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(fontsize=7.5, frameon=True, loc='upper left')

plt.suptitle('Figure 3: Residual Error Distributions by Workpiece Material', fontsize=11, fontweight='bold')
plt.tight_layout()
fig3_path = FIGURES_DIR / "fig3_error_distribution_by_material.png"
plt.savefig(fig3_path)
plt.close()
print(f"Saved: {fig3_path}")

# Figure 4: Deep Dive on Case 13 Trajectory
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

c13_runs = c13_df['run'].values
c13_actual = c13_df['VB_mm_actual'].values

ax1.plot(c13_runs, c13_actual, 'k-o', linewidth=2.0, markersize=6, label='Actual VB (Ground Truth)')
ax1.plot(c13_runs, c13_df['Ridge_B_pred'], 's--', color='#4A7BB0', linewidth=1.4, label='Ridge Context')
ax1.plot(c13_runs, c13_df['SVR_B_pred'], '^--', color='#2E8B57', linewidth=1.4, label='SVR Context')
ax1.plot(c13_runs, c13_df['Gradient Boosting_B_pred'], 'd--', color='#D9534F', linewidth=1.4, label='GBDT Context')
ax1.axhline(0.30, color='#F57F17', linestyle=':', linewidth=1.2, label='ISO End-of-Life (0.30 mm)')

ax1.set_xlabel('Milling Cut Sequence (Run)')
ax1.set_ylabel('Flank Wear VB (mm)')
ax1.set_title('Case 13 Tool Wear Trajectory & Model Predictions\n(Stainless Steel J45, Condition C5)', fontsize=10, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(fontsize=7.5, frameon=True, loc='upper left')

# Absolute error progression for Case 13
ax2.plot(c13_runs, np.abs(c13_df['Ridge_B_pred'] - c13_actual), 's-', color='#4A7BB0', label='Ridge Abs Error')
ax2.plot(c13_runs, np.abs(c13_df['SVR_B_pred'] - c13_actual), '^-', color='#2E8B57', label='SVR Abs Error')
ax2.plot(c13_runs, np.abs(c13_df['Gradient Boosting_B_pred'] - c13_actual), 'd-', color='#D9534F', label='GBDT Abs Error')
ax2.axvline(6, color='black', linestyle=':', alpha=0.7)
ax2.text(6.2, 0.65, 'Run 6: VB reaches 0.32 mm\nRapid Error Growth Beyond Run 10', fontsize=8, color='#333333')

ax2.set_xlabel('Milling Cut Sequence (Run)')
ax2.set_ylabel('Absolute Error (mm)')
ax2.set_title('Case 13 Prediction Error Escalation\nError Escalates in Severe Wear Phase (Runs 11-15)', fontsize=10, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(fontsize=7.5, frameon=True, loc='upper left')

plt.suptitle('Figure 4: Case 13 Failure Mode Analysis — Tracking Breakdown in Severe Wear', fontsize=12, fontweight='bold')
plt.tight_layout()
fig4_path = FIGURES_DIR / "fig4_case13_deep_dive.png"
plt.savefig(fig4_path)
plt.close()
print(f"Saved: {fig4_path}")

print("="*80)
print("PHASE 4.6 DIAGNOSTIC COMPUTATION AND FIGURES COMPLETE!")
print("="*80)
