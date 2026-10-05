"""
Phase 3.2: Feature Dataset Generation, Redundancy Audit & Compact Feature Selection
NASA Milling Tool Wear Dataset (V2 Flagship Project)

Strategic Directives:
1. Build clean ML-ready feature dataset (145 valid runs).
2. Complete Feature Quality Audit (missing, unique, variance, range, outliers, clipping).
3. Complete Redundancy & Collinearity Audit (Pearson, Spearman, correlation clusters).
4. Preserve complementary physical information across all 4 domains (Current, Vibration, AE, Kinematics).
5. Ground selection in Phase 3.1 within-tool evidence (decouple wear tracking from cutting-condition confounding).
6. Downgrade smcDC features to Diagnostic / Conditional (41 runs clipped flat at +10V).
7. Condition metadata audit: Material + DOC + Feed (8 conditions).
8. Classify features into: Primary / Final Candidate, Secondary, Diagnostic / Conditional, Removed.
9. Strictly NO ML training, NO test splits, NO GroupKFold (belongs to Phase 4).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Configuration and Directories
# -------------------------------------------------------------
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
REPORTS_DIR = WORKSPACE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "phase3_feature_selection"
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
print("STARTING PHASE 3.2: FEATURE DATASET GENERATION & REDUNDANCY AUDIT")
print("="*80)

# -------------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------------
foundation_path = DATA_DIR / "mill_runs_foundation.csv"
features_path = DATA_DIR / "phase3_sensor_exploratory_features.csv"
within_tool_path = DATA_DIR / "phase3_within_tool_correlation_audit.csv"

df_foundation = pd.read_csv(foundation_path)
df_features = pd.read_csv(features_path)
df_within_audit = pd.read_csv(within_tool_path)

print(f"Loaded Foundation CSV: {len(df_foundation)} runs.")
print(f"Loaded Exploratory Features CSV: {len(df_features)} valid runs.")
print(f"Loaded Within-Tool Audit CSV: {len(df_within_audit)} candidate rows.")

# Merge cumulative_time_min from foundation
if 'cumulative_time_min' not in df_features.columns:
    df_features = df_features.merge(
        df_foundation[['case', 'run', 'cumulative_time_min']], 
        on=['case', 'run'], 
        how='left'
    )

# -------------------------------------------------------------
# 2. Condition Metadata & Coverage Audit (8 Cutting Conditions)
# -------------------------------------------------------------
# condition_id: C1..C8 with clear physical parameters
condition_map = {
    ('Cast Iron', 0.75, 0.25): ('C1_CastIron_d0.75_f0.25', 'Cast Iron', 0.75, 0.25),
    ('Cast Iron', 0.75, 0.50): ('C2_CastIron_d0.75_f0.50', 'Cast Iron', 0.75, 0.50),
    ('Cast Iron', 1.50, 0.25): ('C3_CastIron_d1.50_f0.25', 'Cast Iron', 1.50, 0.25),
    ('Cast Iron', 1.50, 0.50): ('C4_CastIron_d1.50_f0.50', 'Cast Iron', 1.50, 0.50),
    ('Stainless Steel J45', 0.75, 0.25): ('C5_Stainless_d0.75_f0.25', 'Stainless Steel J45', 0.75, 0.25),
    ('Stainless Steel J45', 0.75, 0.50): ('C6_Stainless_d0.75_f0.50', 'Stainless Steel J45', 0.75, 0.50),
    ('Stainless Steel J45', 1.50, 0.25): ('C7_Stainless_d1.50_f0.25', 'Stainless Steel J45', 1.50, 0.25),
    ('Stainless Steel J45', 1.50, 0.50): ('C8_Stainless_d1.50_f0.50', 'Stainless Steel J45', 1.50, 0.50)
}

# Attach condition_id to df_features
def get_condition_id(row):
    key = (row['material_name'], row['DOC_mm'], row['feed_mm_rev'])
    return condition_map[key][0]

df_features['condition_id'] = df_features.apply(get_condition_id, axis=1)
df_foundation['condition_id'] = df_foundation.apply(get_condition_id, axis=1)

# Condition Coverage Table
coverage_rows = []
for key, (cond_id, mat, doc, feed) in condition_map.items():
    sub_found = df_foundation[df_foundation['condition_id'] == cond_id]
    sub_feat = df_features[df_features['condition_id'] == cond_id]
    
    cases_tested = sorted(sub_found['case'].unique().tolist())
    total_runs = len(sub_found)
    valid_runs = len(sub_feat)
    missing_vb_runs = len(sub_found[sub_found['status_flag'] == 'MISSING_VB'])
    corrupt_runs = len(sub_found[sub_found['status_flag'] == 'CORRUPTED_SIGNAL'])
    missing_vb_pct = round(missing_vb_runs / total_runs * 100, 1)
    
    coverage_rows.append({
        'condition_id': cond_id,
        'material_name': mat,
        'DOC_mm': doc,
        'feed_mm_rev': feed,
        'tool_cases': str(cases_tested),
        'num_tools': len(cases_tested),
        'total_recorded_runs': total_runs,
        'valid_sensor_runs': valid_runs,
        'corrupted_sensor_runs': corrupt_runs,
        'runs_with_measured_VB': valid_runs,
        'missing_VB_runs': missing_vb_runs,
        'missing_VB_pct': f"{missing_vb_pct}%"
    })

df_condition_coverage = pd.DataFrame(coverage_rows)
cond_csv_path = REPORTS_DIR / "condition_coverage_table.csv"
df_condition_coverage.to_csv(cond_csv_path, index=False)
print(f"Saved Condition Coverage Table: {cond_csv_path}")

# -------------------------------------------------------------
# 3. Candidate Features Definition & Quality Audit
# -------------------------------------------------------------
CANDIDATE_FEATURES = [
    # Current / Load
    ('smcAC_rms', 'smcAC', 'Current / Dynamic Torque', 'Time', 'AC current dynamic energy (cutting force proxy)'),
    ('smcAC_std', 'smcAC', 'Current / Dynamic Torque', 'Time', 'Torque fluctuation standard deviation (collinear with RMS)'),
    ('smcAC_p2p', 'smcAC', 'Current / Dynamic Torque', 'Time', 'Peak-to-peak AC current excursion'),
    ('smcAC_kurtosis', 'smcAC', 'Current / Dynamic Torque', 'Time', 'AC current distribution peakedness'),
    ('smcAC_crest', 'smcAC', 'Current / Dynamic Torque', 'Time', 'AC current crest factor (peak to RMS ratio)'),
    ('smcAC_spindle_band_pwr', 'smcAC', 'Kinematics / Spindle Load', 'Frequency', 'Kinematic spectral power at spindle freq (11-16.5 Hz)'),
    
    # Motor Current DC (Hardware Clipped)
    ('smcDC_mean', 'smcDC', 'DC Motor Baseline Power', 'Time', 'DC current average motor load (CLIPPED at +10V in 41 runs)'),
    ('smcDC_std', 'smcDC', 'DC Motor Baseline Power', 'Time', 'DC current ripple amplitude (variance compressed by clipping)'),
    
    # Spindle Vibration (Mechanical Impact / Degradation)
    ('vib_spindle_kurtosis', 'vib_spindle', 'Mechanical Impact / Shocks', 'Time', 'Spindle impact peakedness (wear-specific, low DOC/Feed bias)'),
    ('vib_spindle_p2p', 'vib_spindle', 'Mechanical Impact / Shocks', 'Time', 'Spindle vibration peak excursion (edge micro-fractures)'),
    ('vib_spindle_mean', 'vib_spindle', 'Mechanical Impact / Baseline', 'Time', 'Spindle vibration envelope baseline level'),
    
    # Table Acoustic Emission (Friction & Workpiece Dynamics)
    ('AE_table_rms', 'AE_table', 'Friction / Contact Energy', 'Time', 'Table acoustic emission effective energy (flank wear friction)'),
    ('AE_table_mean', 'AE_table', 'Friction / Contact Energy', 'Time', 'Table acoustic emission envelope mean (collinear with RMS)'),
    ('AE_table_p2p', 'AE_table', 'Friction / Acoustic Bursts', 'Time', 'Table acoustic burst peak-to-peak range'),
    ('AE_table_kurtosis', 'AE_table', 'Material Metallurgy / Bursts', 'Time', 'Table acoustic burst peakedness (material-dependent, zero wear)'),
    
    # Spindle Acoustic Emission (Tool Holder Micro-bursts)
    ('AE_spindle_p2p', 'AE_spindle', 'Friction / Tool Holder Bursts', 'Time', 'Spindle acoustic burst range (wear-sensitive, lower DOC bias)'),
    ('AE_spindle_rms', 'AE_spindle', 'Friction / Tool Holder Energy', 'Time', 'Spindle acoustic effective energy'),
    ('AE_spindle_mean', 'AE_spindle', 'Friction / Tool Holder Energy', 'Time', 'Spindle acoustic envelope mean (collinear with RMS)'),
    ('AE_spindle_kurtosis', 'AE_spindle', 'Friction / Micro-chipping', 'Time', 'Spindle acoustic burst peakedness'),
    
    # Table Vibration & Harmonics
    ('vib_table_spindle_band_pwr', 'vib_table', 'Kinematics / Machine Runout', 'Frequency', 'Table vibration power at spindle rotational freq (11-16.5 Hz)'),
    ('vib_table_tooth_band_pwr', 'vib_table', 'Kinematics / Tooth Impacts', 'Frequency', 'Table vibration power at tooth passing freq (75-90 Hz)'),
    ('vib_table_mean', 'vib_table', 'Table Structural Dynamics', 'Time', 'Table vibration envelope baseline (dampens with wear)'),
    ('vib_table_std', 'vib_table', 'Table Structural Dynamics', 'Time', 'Table vibration fluctuation amplitude'),
    ('vib_table_p2p', 'vib_table', 'Table Structural Dynamics', 'Time', 'Table vibration envelope range (reflects material, negative wear)'),
    ('vib_table_kurtosis', 'vib_table', 'Table Structural Dynamics', 'Time', 'Table impact shock peakedness (unstable across tools)')
]

audit_rows = []
for col_name, sensor, phys_domain, domain_type, interp in CANDIDATE_FEATURES:
    vals = df_features[col_name]
    
    n_missing = vals.isna().sum()
    pct_missing = round(n_missing / len(vals) * 100, 2)
    n_unique = vals.nunique()
    mean_val = vals.mean()
    std_val = vals.std()
    var_val = vals.var()
    min_val = vals.min()
    max_val = vals.max()
    median_val = vals.median()
    skew_val = vals.skew()
    kurt_val = vals.kurt()
    
    # IQR Outliers
    q25 = vals.quantile(0.25)
    q75 = vals.quantile(0.75)
    iqr = q75 - q25
    lower_fence = q25 - 1.5 * iqr
    upper_fence = q75 + 1.5 * iqr
    n_outliers = ((vals < lower_fence) | (vals > upper_fence)).sum()
    
    # Near Zero Variance Check
    nzv_flag = (var_val < 1e-4) or (vals.value_counts(normalize=True).iloc[0] > 0.95)
    
    # Hardware / Signal Quality Issues
    if sensor == 'smcDC':
        quality_note = "HARDWARE SATURATION: 41/145 runs (28.3%) clipped flat at +9.995 V"
    elif sensor in ['vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']:
        quality_note = "Processed 8ms analog RMS envelope voltage (non-negative, smoothed)"
    else:
        quality_note = "Direct AC amplified signal (zero-centered bipolar, clean)"
        
    audit_rows.append({
        'feature_name': col_name,
        'sensor_channel': sensor,
        'physical_domain': phys_domain,
        'domain_type': domain_type,
        'missing_count': n_missing,
        'missing_pct': f"{pct_missing}%",
        'num_unique_values': n_unique,
        'mean': round(mean_val, 4),
        'std': round(std_val, 4),
        'variance': round(var_val, 5),
        'min': round(min_val, 4),
        'median': round(median_val, 4),
        'max': round(max_val, 4),
        'skewness': round(skew_val, 2),
        'kurtosis': round(kurt_val, 2),
        'iqr_outliers_count': n_outliers,
        'near_zero_variance': nzv_flag,
        'data_quality_issues': quality_note
    })

df_feature_audit = pd.DataFrame(audit_rows)
audit_csv_path = REPORTS_DIR / "feature_audit_table.csv"
df_feature_audit.to_csv(audit_csv_path, index=False)
print(f"Saved Feature Quality Audit Table: {audit_csv_path}")

# -------------------------------------------------------------
# 4. Redundancy / Collinearity Audit & Selection Classification
# -------------------------------------------------------------
feature_cols = [f[0] for f in CANDIDATE_FEATURES]
corr_pearson = df_features[feature_cols].corr(method='pearson')
corr_spearman = df_features[feature_cols].corr(method='spearman')

within_tool_dict = df_within_audit.set_index('column_name')['within_tool_median_spearman'].to_dict()
within_tool_pos = df_within_audit.set_index('column_name')['tools_positive'].to_dict()

# Define Final Classification & Explicit Engineering Rationale
# Categories:
# 1. Primary / Final Candidate
# 2. Secondary
# 3. Diagnostic / Conditional
# 4. Removed
SELECTION_DECISIONS = {
    # Current / Load
    'smcAC_rms': (
        'Primary / Final Candidate',
        'Spindle motor current RMS / dynamic load proxy. Monotonic wear progression within EVERY tool (median within-tool rs = +0.993, 14/14 tools). Clean DAQ, highly robust.'
    ),
    'smcAC_std': (
        'Removed',
        'Mathematically redundant with smcAC_rms (Pearson r = 0.9999). For a zero-mean AC waveform, RMS and standard deviation provide identical information.'
    ),
    'smcAC_p2p': (
        'Secondary',
        'Dynamic excursion range. Useful alternative index for peak tooth impact forces, but highly collinear with smcAC_rms (r = 0.985). Retained as secondary candidate.'
    ),
    'smcAC_kurtosis': (
        'Removed',
        'Near-zero association with flank wear (pooled rs = -0.062, within-tool median rs = -0.298). Does not track wear progression reliably.'
    ),
    'smcAC_crest': (
        'Removed',
        'Weak association with flank wear (pooled rs = -0.122, within-tool median rs = -0.471). Normalization dampens the wear signal.'
    ),
    'smcAC_spindle_band_pwr': (
        'Secondary',
        'Kinematic spindle rotational harmonic power (13.8 Hz). Highly collinear with smcAC_rms (r = 0.985). Retained as Secondary / Candidate for Phase 4 ablation (RMS only vs RMS + spindle band).'
    ),
    
    # Motor Current DC
    'smcDC_mean': (
        'Diagnostic / Conditional',
        'Severe hardware saturation at +9.995V in 41/145 runs (28.3%). High correlation (rs = 0.783) is compromised by flat clipping. Kept for sensitivity checks only; NOT for primary ML.'
    ),
    'smcDC_std': (
        'Diagnostic / Conditional',
        'Dynamic ripple variance compressed by hardware clipping in 41 runs. Excluded from primary feature set; retained as conditional diagnostic.'
    ),
    
    # Spindle Vibration
    'vib_spindle_kurtosis': (
        'Primary / Final Candidate',
        'Captures changes in the peakedness of spindle-side vibration envelope associated with increasing wear (pooled rs = +0.591, within-tool median rs = +0.528) with near-zero DOC/Feed bias (rs = -0.21 and -0.15).'
    ),
    'vib_spindle_p2p': (
        'Primary / Final Candidate',
        'Spindle vibration peak envelope excursion. Complements kurtosis by capturing absolute peak vibration amplitudes (within-tool median rs = +0.609, 13/14 positive).'
    ),
    'vib_spindle_mean': (
        'Removed',
        'Correlates negatively with wear within tools (median rs = -0.821, 0/14 positive). Baseline level is dominated by machine operating setup rather than tool wear.'
    ),
    
    # Table Acoustic Emission
    'AE_table_rms': (
        'Primary / Final Candidate',
        'Acoustic-emission envelope associated with cutting/contact activity and showing strong within-tool association with measured VB (median within-tool rs = +0.957, 14/14 positive).'
    ),
    'AE_table_mean': (
        'Removed',
        'Mathematically redundant with AE_table_rms (Pearson r = 0.9991). Retain AE_table_rms as the more standard energetic metric.'
    ),
    'AE_table_p2p': (
        'Secondary',
        'Table acoustic burst range. Captures transient acoustic bursts from micro-fractures (within-tool median rs = +0.804), but partially collinear with AE_table_rms (r = 0.887).'
    ),
    'AE_table_kurtosis': (
        'Removed',
        'Zero wear tracking within tools (median rs = +0.036, positive in 7 tools, negative in 7 tools). Pooled correlation (rs = 0.135) was an illusion driven by Stainless Steel having higher burstiness than Cast Iron (rs = 0.750 with Material).'
    ),
    
    # Spindle Acoustic Emission
    'AE_spindle_p2p': (
        'Primary / Final Candidate',
        'Range of the processed acoustic-emission envelope measured at the spindle-side sensor. Strong wear tracking (median within-tool rs = +0.785, 14/14 positive) and lower DOC confounding (rs = 0.237) than table AE.'
    ),
    'AE_spindle_rms': (
        'Secondary',
        'Acoustic energy transmitted to tool holder (within-tool median rs = +0.901). High fidelity, but retained as secondary to avoid double-counting spindle AE with AE_spindle_p2p.'
    ),
    'AE_spindle_mean': (
        'Removed',
        'Mathematically redundant with AE_spindle_rms (Pearson r = 0.9994).'
    ),
    'AE_spindle_kurtosis': (
        'Removed',
        'Weak and inconsistent wear tracking within tools (median rs = +0.121, positive in only 8/14 tools).'
    ),

    
    # Table Vibration & Harmonics
    'vib_table_spindle_band_pwr': (
        'Secondary',
        'Table vibration power at spindle rotational frequency (13.8 Hz). Measures rotational unbalance and mechanical runout. Retained as secondary kinematic candidate.'
    ),
    'vib_table_tooth_band_pwr': (
        'Secondary',
        'Table vibration power at cutter tooth passing frequency (82.6 Hz). Measures tooth engagement dynamics. Retained as secondary kinematic candidate.'
    ),
    'vib_table_mean': (
        'Removed',
        'Table vibration RMS envelope baseline dampens as tool wears (within-tool median rs = -0.588). Reflects table mass dampening rather than progressive wear.'
    ),
    'vib_table_std': (
        'Removed',
        'Table envelope fluctuation amplitude decreases with wear within tools (median rs = -0.679).'
    ),
    'vib_table_p2p': (
        'Removed',
        'Table vibration envelope range correlates negatively with wear within tools (median rs = -0.653) and is dominated by Material differences (rs = 0.628).'
    ),
    'vib_table_kurtosis': (
        'Removed',
        'Highly erratic behavior across tools (within-tool correlations range from -1.0 to +0.97). Unreliable as a predictive feature.'
    )
}

redundancy_rows = []
for col_name, sensor, phys_domain, domain_type, interp in CANDIDATE_FEATURES:
    p_vb, _ = stats.pearsonr(df_features[col_name], df_features['VB_mm'])
    s_vb, _ = stats.spearmanr(df_features[col_name], df_features['VB_mm'])
    
    # Top Collinear Feature among other candidates
    other_corrs = corr_pearson[col_name].drop(col_name)
    top_collin_feat = other_corrs.abs().idxmax()
    top_collin_r = other_corrs[top_collin_feat]
    
    # Collinear Group assignment
    if col_name in ['smcAC_rms', 'smcAC_std', 'smcAC_p2p', 'smcAC_spindle_band_pwr']:
        collin_group = "Group 1: Spindle Motor Current Dynamic Load"
    elif col_name in ['smcDC_mean', 'smcDC_std']:
        collin_group = "Group 2: Spindle Motor DC Baseline Load (Clipped)"
    elif col_name in ['AE_table_rms', 'AE_table_mean', 'AE_table_p2p']:
        collin_group = "Group 3: Table Acoustic Friction Energy & Baseline"
    elif col_name in ['AE_spindle_rms', 'AE_spindle_mean', 'AE_spindle_p2p']:
        collin_group = "Group 4: Spindle Acoustic Transmission & Bursts"
    elif col_name in ['vib_spindle_kurtosis', 'vib_spindle_p2p']:
        collin_group = "Group 5: Spindle Mechanical Impact & Shock"
    elif col_name in ['vib_table_spindle_band_pwr', 'vib_table_tooth_band_pwr']:
        collin_group = "Group 6: Table Kinematic Harmonics"
    elif col_name in ['vib_table_mean', 'vib_table_std', 'vib_table_p2p']:
        collin_group = "Group 7: Table Vibration Envelope & Baseline"
    else:
        collin_group = "Group 8: Uninformative / Shape / Kurtosis Candidates"
        
    status, rationale = SELECTION_DECISIONS[col_name]
    
    redundancy_rows.append({
        'feature_name': col_name,
        'sensor_channel': sensor,
        'physical_domain': phys_domain,
        'collinear_cluster': collin_group,
        'pearson_VB': round(p_vb, 3),
        'spearman_VB': round(s_vb, 3),
        'within_tool_median_spearman': round(within_tool_dict.get(col_name, np.nan), 3),
        'within_tool_pos_fraction': within_tool_pos.get(col_name, 'N/A'),
        'top_collinear_feature': top_collin_feat,
        'top_collinear_pearson_r': round(top_collin_r, 3),
        'final_selection_status': status,
        'selection_rationale': rationale
    })

df_redundancy = pd.DataFrame(redundancy_rows)
red_csv_path = REPORTS_DIR / "feature_redundancy_table.csv"
df_redundancy.to_csv(red_csv_path, index=False)
print(f"Saved Feature Redundancy Table: {red_csv_path}")

# -------------------------------------------------------------
# 5. Build and Export the Clean Feature Dataset (feature_dataset_v32.csv)
# -------------------------------------------------------------
primary_features = [k for k, v in SELECTION_DECISIONS.items() if v[0] == 'Primary / Final Candidate']
secondary_features = [k for k, v in SELECTION_DECISIONS.items() if v[0] == 'Secondary']
diagnostic_features = [k for k, v in SELECTION_DECISIONS.items() if v[0] == 'Diagnostic / Conditional']

print(f"\nFinal Feature Partition:")
print(f"- Primary / Final Candidates ({len(primary_features)}): {primary_features}")
print(f"- Secondary Candidates ({len(secondary_features)}): {secondary_features}")
print(f"- Diagnostic / Conditional ({len(diagnostic_features)}): {diagnostic_features}")
print(f"- Removed ({len(SELECTION_DECISIONS) - len(primary_features) - len(secondary_features) - len(diagnostic_features)} features)")

metadata_cols = [
    'case', 'run', 'material_name', 'material_code', 
    'DOC_mm', 'feed_mm_rev', 'condition_id', 'cumulative_time_min'
]
target_col = ['VB_mm']

# Combine columns in logical order: Target -> Metadata -> Primary -> Secondary -> Diagnostic
ordered_cols = target_col + metadata_cols + primary_features + secondary_features + diagnostic_features
df_export = df_features[ordered_cols].copy()

# Rename target column to VB_mm (and provide alias column if desired, keeping VB_mm standard)
df_export_path = DATA_DIR / "feature_dataset_v32.csv"
df_export.to_csv(df_export_path, index=False)
print(f"\nExported ML-ready Feature Dataset: {df_export_path} ({df_export.shape[0]} rows, {df_export.shape[1]} columns)")

# -------------------------------------------------------------
# 6. Generate Figures
# -------------------------------------------------------------
# Figure 1: Full Candidate Feature Correlation Heatmap (Annotated with Selection Status)
fig, ax = plt.subplots(figsize=(13, 11))

# Sort features by selection status then domain
status_order = {'Primary / Final Candidate': 0, 'Secondary': 1, 'Diagnostic / Conditional': 2, 'Removed': 3}
sorted_feat_cols = sorted(feature_cols, key=lambda x: (status_order[SELECTION_DECISIONS[x][0]], x))

corr_matrix_sorted = corr_pearson.loc[sorted_feat_cols, sorted_feat_cols]

im = ax.imshow(corr_matrix_sorted.values, cmap='coolwarm', vmin=-1.0, vmax=1.0)
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label("Pearson Correlation Coefficient (r)", fontsize=10, fontweight='bold')

ax.set_xticks(np.arange(len(sorted_feat_cols)))
ax.set_yticks(np.arange(len(sorted_feat_cols)))

# Color tick labels by status
status_colors = {
    'Primary / Final Candidate': '#27ae60',
    'Secondary': '#2980b9',
    'Diagnostic / Conditional': '#d35400',
    'Removed': '#7f8c8d'
}
tick_labels = []
for f in sorted_feat_cols:
    st = SELECTION_DECISIONS[f][0]
    tick_labels.append(f"{f}")

ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
ax.set_yticklabels(tick_labels, fontsize=8)

for i, f in enumerate(sorted_feat_cols):
    st = SELECTION_DECISIONS[f][0]
    ax.get_xticklabels()[i].set_color(status_colors[st])
    ax.get_yticklabels()[i].set_color(status_colors[st])
    if st == 'Primary / Final Candidate':
        ax.get_xticklabels()[i].set_weight('bold')
        ax.get_yticklabels()[i].set_weight('bold')

ax.set_title("Figure 1: Candidate Feature Collinearity Matrix & Phase 3.2 Selection Tiers\n"
             "Green Bold = Primary Candidate | Blue = Secondary | Orange = Diagnostic/Clipped | Grey = Removed", 
             fontsize=11.5, fontweight='bold', pad=15)
plt.tight_layout()
fig1_path = FIGURES_DIR / "fig1_feature_correlation_heatmap.png"
plt.savefig(fig1_path, dpi=200)
plt.close()
print(f"Saved Figure 1: {fig1_path.name}")

# Figure 2: Primary Feature Distributions vs Wear Stages
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

wear_bins = [0.0, 0.20, 0.40, 1.0]
wear_labels = ['Fresh (VB < 0.20mm)', 'Moderate (0.20-0.40mm)', 'Severely Worn (VB > 0.40mm)']
df_features['wear_stage'] = pd.cut(df_features['VB_mm'], bins=wear_bins, labels=wear_labels, right=False)

box_colors = ['#2ecc71', '#f39c12', '#e74c3c']

plot_features = primary_features + ['smcAC_spindle_band_pwr']

for idx, col in enumerate(plot_features):
    ax = axes[idx]
    box_data = [df_features[df_features['wear_stage'] == lbl][col].dropna() for lbl in wear_labels]
    bp = ax.boxplot(box_data, patch_artist=True, tick_labels=['Fresh', 'Moderate', 'Worn'], widths=0.55)
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        
    tag = "Primary" if col in primary_features else "Secondary (Phase 4 Ablation)"
    ax.set_title(f"{col} [{tag}]", fontsize=9.5, fontweight='bold')
    ax.set_ylabel("Feature Value (V or V²)", fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)

plt.suptitle("Figure 2: Candidate Feature Distributions Across Flank Wear Degradation Stages\n"
             "(5 Primary Candidates + Spindle Band Power Ablation Candidate Showing Consistent Progression Across Wear States)", 
             fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout()
fig2_path = FIGURES_DIR / "fig2_candidate_feature_distributions.png"
plt.savefig(fig2_path, dpi=200)
plt.close()
print(f"Saved Figure 2: {fig2_path.name}")

# Figure 3: Retained vs Removed Features — Within-Tool Wear Fidelity vs Collinearity
fig, ax = plt.subplots(figsize=(14, 8))

# Scatter of Top Collinearity vs Within-Tool Median Spearman
status_markers = {
    'Primary / Final Candidate': ('#27ae60', 'o', 100, 'Primary / Final Candidate (Retained)'),
    'Secondary': ('#2980b9', 's', 80, 'Secondary Candidate (Retained)'),
    'Diagnostic / Conditional': ('#e67e22', '^', 90, 'Diagnostic / Conditional (smcDC Clipped)'),
    'Removed': ('#95a5a6', 'X', 70, 'Removed (Redundant / Low Fidelity)')
}

for status, (color, marker, size, label) in status_markers.items():
    sub = df_redundancy[df_redundancy['final_selection_status'] == status]
    ax.scatter(
        sub['within_tool_median_spearman'], 
        sub['top_collinear_pearson_r'].abs(),
        c=color, marker=marker, s=size, label=label,
        edgecolors='black', lw=0.7, alpha=0.85, zorder=5
    )
    
# Annotate key features
for idx, row in df_redundancy.iterrows():
    fn = row['feature_name']
    if row['final_selection_status'] in ['Primary / Final Candidate', 'Diagnostic / Conditional'] or fn in ['smcAC_std', 'AE_table_kurtosis', 'AE_table_mean']:
        ax.annotate(
            fn, 
            (row['within_tool_median_spearman'], abs(row['top_collinear_pearson_r'])),
            fontsize=8, fontweight='bold' if 'Primary' in row['final_selection_status'] else 'normal',
            xytext=(5, 3), textcoords='offset points', alpha=0.9
        )

ax.axvline(0.50, color='#27ae60', linestyle='--', lw=1.0, alpha=0.6, label='Wear Fidelity Threshold (within-tool rs > 0.50)')
ax.axhline(0.95, color='#c0392b', linestyle=':', lw=1.0, alpha=0.6, label='Extreme Collinearity Threshold (|r| > 0.95)')

ax.set_xlabel("Within-Tool Median Spearman Correlation with VB (Wear Tracking Fidelity)", fontsize=10.5, fontweight='bold')
ax.set_ylabel("Maximum Collinearity with Any Other Candidate (|Pearson r|)", fontsize=10.5, fontweight='bold')
ax.set_title("Figure 3: Phase 3.2 Compact Feature Selection Logic — Wear Tracking Fidelity vs. Collinearity\n"
             "(Illustrating why collinear twins like smcAC_std/AE_table_mean and uninformative features like AE_table_kurtosis were pruned)", 
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlim(-0.9, 1.05)
ax.set_ylim(0.2, 1.02)
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='lower left', fontsize=9.5, framealpha=0.95)

plt.tight_layout()
fig3_path = FIGURES_DIR / "fig3_retained_vs_removed_features.png"
plt.savefig(fig3_path, dpi=200)
plt.close()
print(f"Saved Figure 3: {fig3_path.name}")

print("\n" + "="*80)
print("PHASE 3.2 FEATURE SELECTION & REDUNDANCY AUDIT COMPLETED!")
print("="*80)
