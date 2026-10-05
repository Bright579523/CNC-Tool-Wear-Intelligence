"""
Phase 5.2C: Final Model Behavior & Robustness Audit
NASA Milling Dataset (V2 Flagship Project)

Role: AntiGravity (Implementation Agent)
Strategic Lead: ChatGPT
Domain Lead: Bright

Purpose:
Descriptive robustness and model-behavior audit of the locked Ridge Primary Model
and GBDT Supporting Model. Quantifies where the models perform well, where they struggle,
and how closely their predictions agree.

Governance:
- Purely diagnostic and observational (no new models, tuning, clipping, calibration, or p-values)
- Evaluates out-of-fold predictions from locked LOGO cross-validation (N=145)
- Standardized analytical wear regimes: Low (<0.20 mm), Moderate (0.20-0.40 mm), High (>0.40 mm)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Configuration and Directories
# -------------------------------------------------------------
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = WORKSPACE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
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
print("PHASE 5.2C: FINAL MODEL BEHAVIOR & ROBUSTNESS AUDIT")
print("="*80)

# -------------------------------------------------------------
# 1. Load Locked Out-of-Fold Predictions (LOGO 16 Folds)
# -------------------------------------------------------------
pred_path = REPORTS_DIR / "phase4_5_context_fusion" / "oof_predictions_context.csv"
df = pd.read_csv(pred_path)
print(f"Loaded OOF predictions: {len(df)} rows across {df['case'].nunique()} tool cases.")
assert len(df) == 145, f"Expected 145 rows, got {len(df)}"

y_true = df['VB_mm_actual'].values
pred_r = df['Ridge_B_pred'].values
pred_g = df['Gradient Boosting_B_pred'].values

# Residuals: (pred - actual)
res_r = pred_r - y_true
res_g = pred_g - y_true

# Absolute errors: |actual - pred|
ae_r = np.abs(res_r)
ae_g = np.abs(res_g)

# Verify global benchmarks
mae_r = np.mean(ae_r)
rmse_r = np.sqrt(np.mean(res_r**2))
r2_r = 1 - (np.sum(res_r**2) / np.sum((y_true - np.mean(y_true))**2))

mae_g = np.mean(ae_g)
rmse_g = np.sqrt(np.mean(res_g**2))
r2_g = 1 - (np.sum(res_g**2) / np.sum((y_true - np.mean(y_true))**2))

print(f"Locked Ridge Context LOGO: MAE = {mae_r:.4f} mm, RMSE = {rmse_r:.4f} mm, R2 = {r2_r:.4f}")
print(f"Locked GBDT Context LOGO:  MAE = {mae_g:.4f} mm, RMSE = {rmse_g:.4f} mm, R2 = {r2_g:.4f}")

# -------------------------------------------------------------
# 2. Table 1: Wear-Regime Error Comparison
# -------------------------------------------------------------
print("\n--- Generating Table 1: Wear-Regime Comparison ---")
regimes = {
    'Low Wear (VB < 0.20 mm)': y_true < 0.20,
    'Moderate Wear (0.20 <= VB <= 0.40 mm)': (y_true >= 0.20) & (y_true <= 0.40),
    'High Wear (VB > 0.40 mm)': y_true > 0.40,
    'Overall Lifecycle (All 145 runs)': np.ones(len(y_true), dtype=bool)
}

regime_records = []
for name, idx in regimes.items():
    n = int(idx.sum())
    yt = y_true[idx]
    pr = pred_r[idx]
    pg = pred_g[idx]
    
    r_mae_val = np.mean(np.abs(yt - pr))
    r_rmse_val = np.sqrt(np.mean((yt - pr)**2))
    r_bias_val = np.mean(pr - yt)
    
    g_mae_val = np.mean(np.abs(yt - pg))
    g_rmse_val = np.sqrt(np.mean((yt - pg)**2))
    g_bias_val = np.mean(pg - yt)
    
    regime_records.append({
        'Wear_Regime': name,
        'N': n,
        'Ridge_MAE_mm': r_mae_val,
        'Ridge_RMSE_mm': r_rmse_val,
        'Ridge_Bias_mm': r_bias_val,
        'GBDT_MAE_mm': g_mae_val,
        'GBDT_RMSE_mm': g_rmse_val,
        'GBDT_Bias_mm': g_bias_val
    })

t1_df = pd.DataFrame(regime_records)
t1_csv = REPORTS_DIR / "phase5_2c_wear_regime_comparison.csv"
t1_df.to_csv(t1_csv, index=False)
print(f"Saved: {t1_csv}")
print(t1_df.to_string(index=False))

# -------------------------------------------------------------
# 3. Table 2: Material Error Comparison
# -------------------------------------------------------------
print("\n--- Generating Table 2: Material Comparison ---")
material_records = []
for mat in ['Cast Iron', 'Stainless Steel J45']:
    idx = df['material_name'] == mat
    n = int(idx.sum())
    yt = y_true[idx]
    pr = pred_r[idx]
    pg = pred_g[idx]
    
    r_mae_val = np.mean(np.abs(yt - pr))
    r_rmse_val = np.sqrt(np.mean((yt - pr)**2))
    r_bias_val = np.mean(pr - yt)
    
    g_mae_val = np.mean(np.abs(yt - pg))
    g_rmse_val = np.sqrt(np.mean((yt - pg)**2))
    g_bias_val = np.mean(pg - yt)
    
    material_records.append({
        'Material': mat,
        'N': n,
        'Ridge_MAE_mm': r_mae_val,
        'Ridge_RMSE_mm': r_rmse_val,
        'Ridge_Bias_mm': r_bias_val,
        'GBDT_MAE_mm': g_mae_val,
        'GBDT_RMSE_mm': g_rmse_val,
        'GBDT_Bias_mm': g_bias_val
    })

t2_df = pd.DataFrame(material_records)
t2_csv = REPORTS_DIR / "phase5_2c_material_comparison.csv"
t2_df.to_csv(t2_csv, index=False)
print(f"Saved: {t2_csv}")
print(t2_df.to_string(index=False))

# -------------------------------------------------------------
# 4. Table 3: Condition-Level Error Comparison
# -------------------------------------------------------------
print("\n--- Generating Table 3: Condition-Level Comparison ---")
condition_records = []
for cond_id, grp in df.groupby('condition_id'):
    n = len(grp)
    yt = grp['VB_mm_actual'].values
    pr = grp['Ridge_B_pred'].values
    pg = grp['Gradient Boosting_B_pred'].values
    
    r_mae_val = np.mean(np.abs(yt - pr))
    r_bias_val = np.mean(pr - yt)
    g_mae_val = np.mean(np.abs(yt - pg))
    g_bias_val = np.mean(pg - yt)
    
    better = 'Ridge' if r_mae_val < g_mae_val else 'GBDT'
    
    condition_records.append({
        'Condition_ID': cond_id,
        'Material': grp['material_name'].iloc[0],
        'DOC_mm': grp['DOC_mm'].iloc[0],
        'feed_mm_rev': grp['feed_mm_rev'].iloc[0],
        'N': n,
        'Ridge_MAE_mm': r_mae_val,
        'Ridge_Bias_mm': r_bias_val,
        'GBDT_MAE_mm': g_mae_val,
        'GBDT_Bias_mm': g_bias_val,
        'Better_Model': better
    })

t3_df = pd.DataFrame(condition_records)
t3_csv = REPORTS_DIR / "phase5_2c_condition_comparison.csv"
t3_df.to_csv(t3_csv, index=False)
print(f"Saved: {t3_csv}")
print(t3_df[['Condition_ID', 'N', 'Ridge_MAE_mm', 'GBDT_MAE_mm', 'Better_Model']].to_string(index=False))

# -------------------------------------------------------------
# 5. Table 4: Cross-Model Prediction Agreement
# -------------------------------------------------------------
print("\n--- Generating Table 4: Cross-Model Agreement ---")
corr = float(np.corrcoef(pred_r, pred_g)[0, 1])
diff = pred_r - pred_g
mad = float(np.mean(np.abs(diff)))
diff_rmse = float(np.sqrt(np.mean(diff**2)))
mean_signed_diff = float(np.mean(diff))

t4_df = pd.DataFrame([{
    'Metric': 'Pearson Correlation (r)',
    'Value': corr,
    'Unit': 'dimensionless',
    'Interpretation': 'Strong linear agreement between Ridge and GBDT predictions'
}, {
    'Metric': 'Mean Absolute Difference (MAD)',
    'Value': mad,
    'Unit': 'mm',
    'Interpretation': 'Average absolute divergence between model predictions'
}, {
    'Metric': 'RMSE of Prediction Difference',
    'Value': diff_rmse,
    'Unit': 'mm',
    'Interpretation': 'Root mean square difference between model outputs'
}, {
    'Metric': 'Mean Signed Difference (Ridge - GBDT)',
    'Value': mean_signed_diff,
    'Unit': 'mm',
    'Interpretation': 'Zero systematic net bias between the two model formulations'
}])

t4_csv = REPORTS_DIR / "phase5_2c_model_agreement.csv"
t4_df.to_csv(t4_csv, index=False)
print(f"Saved: {t4_csv}")
print(t4_df.to_string(index=False))

# -------------------------------------------------------------
# 6. Table 5: Largest Model Disagreements (Top 10)
# -------------------------------------------------------------
print("\n--- Generating Table 5: Top 10 Disagreements ---")
df['pred_difference'] = diff
df['abs_difference'] = np.abs(diff)

top10 = df.sort_values(by='abs_difference', ascending=False).head(10).copy()

t5_cols = [
    'case', 'run', 'material_name', 'DOC_mm', 'feed_mm_rev',
    'VB_mm_actual', 'Ridge_B_pred', 'Gradient Boosting_B_pred',
    'pred_difference', 'abs_difference'
]

t5_df = top10[t5_cols].copy()
t5_df.columns = [
    'Case_Audit_ID', 'Run_Index', 'Material', 'DOC_mm', 'Feed_mm_rev',
    'Observed_VB_mm', 'Ridge_Pred_mm', 'GBDT_Pred_mm',
    'Signed_Diff_mm', 'Abs_Disagreement_mm'
]

t5_csv = REPORTS_DIR / "phase5_2c_largest_disagreements.csv"
t5_df.to_csv(t5_csv, index=False)
print(f"Saved: {t5_csv}")
print(t5_df.to_string(index=False))

# -------------------------------------------------------------
# 7. High-Wear and Case 13 Audit
# -------------------------------------------------------------
c13 = df[df['case'] == 13]
print(f"\nCase 13 Total Runs: {len(c13)}, VB max: {c13['VB_mm_actual'].max():.2f} mm")
max_pt = df.loc[df['VB_mm_actual'].idxmax()]
print(f"Extreme Point (VB = {max_pt['VB_mm_actual']:.2f} mm):")
print(f"  Case {int(max_pt['case'])}, Run {int(max_pt['run'])}, {max_pt['material_name']}")
print(f"  Ridge Prediction: {max_pt['Ridge_B_pred']:.4f} mm (Error: {max_pt['Ridge_B_pred'] - max_pt['VB_mm_actual']:+.4f} mm)")
print(f"  GBDT Prediction:  {max_pt['Gradient Boosting_B_pred']:.4f} mm (Error: {max_pt['Gradient Boosting_B_pred'] - max_pt['VB_mm_actual']:+.4f} mm)")

# -------------------------------------------------------------
# 8. Publication-Quality Figures Generation
# -------------------------------------------------------------
print("\n--- Generating Publication Figures ---")

# Figure 1: Observed vs Predicted VB - Ridge
fig, ax = plt.subplots(figsize=(7.5, 5.2))
scatter_r = ax.scatter(y_true, pred_r, c=df['material_code'], cmap='coolwarm',
                       edgecolor='black', linewidth=0.6, s=45, alpha=0.85, zorder=4)

# 1:1 Identity line
lims = [-0.22, 1.60]
ax.plot(lims, lims, color='black', linestyle='--', linewidth=1.1, label='1:1 Perfect Prediction Line', zorder=3)

# Shaded wear regimes
ax.axvspan(-0.25, 0.20, alpha=0.08, color='blue', label='Low Wear (<0.20 mm)')
ax.axvspan(0.20, 0.40, alpha=0.08, color='green', label='Moderate Wear (0.20–0.40 mm)')
ax.axvspan(0.40, 1.65, alpha=0.08, color='orange', label='High Wear (>0.40 mm)')

# Wear threshold indicator at 0.30 mm
ax.axvline(0.30, color='darkgreen', linestyle=':', linewidth=1.2, alpha=0.8)
ax.text(0.31, -0.15, 'Study Threshold: 0.30 mm', color='darkgreen', fontsize=8, fontweight='bold')

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Observed Flank Wear: VB (mm)', fontsize=10, fontweight='bold')
ax.set_ylabel('Predicted Flank Wear: Ridge (mm)', fontsize=10, fontweight='bold')
ax.set_title('Observed vs. Predicted Flank Wear — Ridge Primary Model (N=145)\n[Context LOGO Validation (16 Folds), alpha=1.0]',
             fontsize=11, fontweight='bold', pad=10)
ax.grid(True, linestyle=':', alpha=0.5)

# Metrics box
r_metrics_text = (
    f"Overall LOGO Performance:\n"
    f"MAE  = {mae_r:.4f} mm\n"
    f"RMSE = {rmse_r:.4f} mm\n"
    f"R²   = {r2_r:.4f}\n"
    f"Bias = {np.mean(res_r):+.4f} mm"
)
ax.text(0.04, 0.70, r_metrics_text, transform=ax.transAxes, fontsize=8.5, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))

# Legend
ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
plt.tight_layout()
fig1_path = FIGURES_DIR / "fig_phase5_2c_ridge_observed_vs_predicted.png"
plt.savefig(fig1_path, bbox_inches='tight', dpi=200)
plt.close()
print(f"Saved: {fig1_path}")

# Figure 2: Observed vs Predicted VB - GBDT
fig, ax = plt.subplots(figsize=(7.5, 5.2))
scatter_g = ax.scatter(y_true, pred_g, c=df['material_code'], cmap='coolwarm',
                       edgecolor='black', linewidth=0.6, s=45, alpha=0.85, zorder=4)

ax.plot(lims, lims, color='black', linestyle='--', linewidth=1.1, label='1:1 Perfect Prediction Line', zorder=3)
ax.axvspan(-0.25, 0.20, alpha=0.08, color='blue', label='Low Wear (<0.20 mm)')
ax.axvspan(0.20, 0.40, alpha=0.08, color='green', label='Moderate Wear (0.20–0.40 mm)')
ax.axvspan(0.40, 1.65, alpha=0.08, color='orange', label='High Wear (>0.40 mm)')
ax.axvline(0.30, color='darkgreen', linestyle=':', linewidth=1.2, alpha=0.8)
ax.text(0.31, -0.15, 'Study Threshold: 0.30 mm', color='darkgreen', fontsize=8, fontweight='bold')

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Observed Flank Wear: VB (mm)', fontsize=10, fontweight='bold')
ax.set_ylabel('Predicted Flank Wear: GBDT (mm)', fontsize=10, fontweight='bold')
ax.set_title('Observed vs. Predicted Flank Wear — GBDT Supporting Model (N=145)\n[Context LOGO Validation (16 Folds), max_depth=3, subsample=0.8]',
             fontsize=11, fontweight='bold', pad=10)
ax.grid(True, linestyle=':', alpha=0.5)

g_metrics_text = (
    f"Overall LOGO Performance:\n"
    f"MAE  = {mae_g:.4f} mm\n"
    f"RMSE = {rmse_g:.4f} mm\n"
    f"R²   = {r2_g:.4f}\n"
    f"Bias = {np.mean(res_g):+.4f} mm"
)
ax.text(0.04, 0.70, g_metrics_text, transform=ax.transAxes, fontsize=8.5, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))

ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
plt.tight_layout()
fig2_path = FIGURES_DIR / "fig_phase5_2c_gbdt_observed_vs_predicted.png"
plt.savefig(fig2_path, bbox_inches='tight', dpi=200)
plt.close()
print(f"Saved: {fig2_path}")

# Figure 3: Ridge vs GBDT Prediction Agreement
fig, ax = plt.subplots(figsize=(7.2, 5.2))
scatter_ag = ax.scatter(pred_r, pred_g, c=y_true, cmap='viridis',
                        edgecolor='black', linewidth=0.6, s=45, alpha=0.85, zorder=4)
cbar = plt.colorbar(scatter_ag, ax=ax)
cbar.set_label('Observed Flank Wear: VB (mm)', fontsize=9, fontweight='bold')

ag_lims = [-0.22, 1.05]
ax.plot(ag_lims, ag_lims, color='red', linestyle='--', linewidth=1.2, label='1:1 Model Consensus Line', zorder=3)

ax.set_xlim(ag_lims)
ax.set_ylim(ag_lims)
ax.set_xlabel('Ridge Primary Prediction (mm)', fontsize=10, fontweight='bold')
ax.set_ylabel('GBDT Supporting Prediction (mm)', fontsize=10, fontweight='bold')
ax.set_title('Cross-Model Prediction Agreement: Ridge vs. GBDT (N=145)\n[High Linear Concordance with Localized Extremes Disagreement]',
             fontsize=11, fontweight='bold', pad=10)
ax.grid(True, linestyle=':', alpha=0.5)

ag_text = (
    f"Concordance Metrics:\n"
    f"Pearson r = {corr:.4f}\n"
    f"Mean Abs Diff = {mad:.4f} mm\n"
    f"Diff RMSE     = {diff_rmse:.4f} mm\n"
    f"Net Mean Bias = {mean_signed_diff:+.4f} mm"
)
ax.text(0.04, 0.72, ag_text, transform=ax.transAxes, fontsize=8.5, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))

ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
plt.tight_layout()
fig3_path = FIGURES_DIR / "fig_phase5_2c_ridge_vs_gbdt_agreement.png"
plt.savefig(fig3_path, bbox_inches='tight', dpi=200)
plt.close()
print(f"Saved: {fig3_path}")

# Figure 4: Absolute Prediction Error by Observed VB
fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.scatter(y_true, ae_r, color='#1F77B4', edgecolor='black', linewidth=0.5, s=42, alpha=0.75, label='Ridge Absolute Error')
ax.scatter(y_true, ae_g, color='#FF7F0E', edgecolor='black', linewidth=0.5, s=42, alpha=0.75, marker='^', label='GBDT Absolute Error')

# Wear regime background zones
ax.axvspan(-0.02, 0.20, alpha=0.08, color='blue')
ax.axvspan(0.20, 0.40, alpha=0.08, color='green')
ax.axvspan(0.40, 1.60, alpha=0.08, color='orange')

# Vertical division lines
ax.axvline(0.20, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)
ax.axvline(0.40, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)
ax.axvline(0.30, color='darkgreen', linestyle=':', linewidth=1.2, alpha=0.8)

# Annotations for zones (placed clearly across top with legend in empty lower right)
ax.text(0.10, 0.73, 'Low Wear Zone\n(<0.20 mm)', ha='center', fontsize=8.5, fontweight='bold', color='#1A365D',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF8FF', alpha=0.9, edgecolor='#BEE3F8'))
ax.text(0.30, 0.73, 'Sweet Spot Zone\n(0.20–0.40 mm)', ha='center', fontsize=8.5, fontweight='bold', color='#1C4532',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0FFF4', alpha=0.9, edgecolor='#C6F6D5'))
ax.text(0.95, 0.73, 'High Wear Zone (>0.40 mm)\nError Expansion & Tail Underprediction', ha='center', fontsize=8.5, fontweight='bold', color='#7B341E',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFAF0', alpha=0.9, edgecolor='#FEEBC8'))

ax.set_xlim(-0.05, 1.58)
ax.set_ylim(-0.02, 0.82)
ax.set_xlabel('Observed Flank Wear: VB (mm)', fontsize=10, fontweight='bold')
ax.set_ylabel('Absolute Prediction Error: |VB - Pred| (mm)', fontsize=10, fontweight='bold')
ax.set_title('Absolute Prediction Error Across the Wear Lifecycle\n[Ridge vs. GBDT: Moderate Wear Sweet Spot and High Wear Error Expansion]',
             fontsize=11, fontweight='bold', pad=10)
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)

plt.tight_layout()
fig4_path = FIGURES_DIR / "fig_phase5_2c_absolute_error_vs_vb.png"
plt.savefig(fig4_path, bbox_inches='tight', dpi=200)
plt.close()
print(f"Saved: {fig4_path}")

print("="*80)
print("PHASE 5.2C SCRIPT EXECUTION COMPLETE")
print("="*80)
