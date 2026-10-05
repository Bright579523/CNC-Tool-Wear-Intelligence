"""
Phase 2 & Phase 2.5: Manufacturing Analytics & Productivity-Adjusted Tool-Life Metrics
NASA Milling Dataset (V2 Flagship Project)

Author: AntiGravity (Implementation Agent)
Domain Lead / Owner: Bright
Strategic Lead: ChatGPT

Analytical Rules:
- Manufacturing-first, data-science-supported (NO ML in Phase 2)
- Descriptive terminology only (NO causal claims)
- Threshold VB = 0.30 mm is an analytical project benchmark, not universal standard
- Interval wear rates (ΔVB / Δtime) computed between consecutive observations
- Condition-matched tool life is primary; pooled interval ratio is analytical detail only
- Case 6 documented as aborted / insufficient data
- Official Productivity Metric: Specific Material Removal Rate MRR' = ap * vf (mm²/min)
  (Parameter-free; does NOT assume radial engagement ae)
- Phase 2.5 Metric: Productivity-Adjusted Tool Life = MRR' * T_0.30 (mm² per unit width)
- Material name strictly standardized as 'Stainless Steel J45' across all outputs
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. Experimental Parameters from NASA Readme.pdf
# -------------------------------------------------------------
SPINDLE_RPM = 826.0           # 826 RPM (spindle speed)
CUTTING_SPEED_M_MIN = 200.0   # 200 m/min nominal cutting speed
WORKPIECE_LENGTH_MM = 483.0   # 483 mm length per pass
WORKPIECE_WIDTH_MM = 178.0    # 178 mm width

# Load Phase 1 Foundation Data
FOUNDATION_PATH = DATA_DIR / "mill_runs_foundation.csv"
assert FOUNDATION_PATH.exists(), f"Missing {FOUNDATION_PATH}"
df_runs = pd.read_csv(FOUNDATION_PATH)

# Standardize material naming to 'Stainless Steel J45'
df_runs['material_name'] = df_runs['material_name'].replace({
    'Steel J45': 'Stainless Steel J45',
    'stainless steel J45': 'Stainless Steel J45'
})

print(f"Loaded {len(df_runs)} runs from {FOUNDATION_PATH.name}")

# -------------------------------------------------------------
# 2. Tool-Life Calculation with Linear Interpolation
# -------------------------------------------------------------
def estimate_tool_life(valid_df, threshold):
    """
    Estimates cumulative cutting time and pass count to reach threshold VB.
    Returns: (time_min, pass_num, method_str, flag_str)
    """
    if len(valid_df) < 2:
        return np.nan, np.nan, "Not Reached / Insufficient Data", "Aborted / Insufficient Observations"
        
    exceeded = valid_df[valid_df['VB_mm'] >= threshold]
    if len(exceeded) == 0:
        max_t = valid_df['cumulative_time_min'].max()
        max_p = valid_df['run'].max()
        return np.nan, np.nan, "Right-Censored / Not Reached", f"Right-Censored at {max_t:.1f} min (Pass {max_p})"
        
    first_hit = exceeded.iloc[0]
    t_hit = first_hit['cumulative_time_min']
    p_hit = first_hit['run']
    vb_hit = first_hit['VB_mm']
    
    # Check if exactly on threshold or was first point
    hit_idx = valid_df.index.get_loc(first_hit.name)
    if hit_idx == 0 or vb_hit == threshold:
        return float(t_hit), float(p_hit), "Exact Observation", "Observed at or above threshold at initial measurement" if hit_idx == 0 else "Exact Match"
        
    # Linear interpolation between previous and current observation
    prev_hit = valid_df.iloc[hit_idx - 1]
    t_prev = prev_hit['cumulative_time_min']
    p_prev = prev_hit['run']
    vb_prev = prev_hit['VB_mm']
    
    if vb_hit != vb_prev:
        fraction = (threshold - vb_prev) / (vb_hit - vb_prev)
        t_interp = t_prev + fraction * (t_hit - t_prev)
        p_interp = p_prev + fraction * (p_hit - p_prev)
        return float(t_interp), float(p_interp), "Linear Interpolated", "Interpolated between bounding observations"
    else:
        return float(t_hit), float(p_hit), "Exact Observation", "Exact Match"

# -------------------------------------------------------------
# 3. Compute Granular Interval Wear Rates & Tool Life Summary
# -------------------------------------------------------------
case_stats = []
tool_life_records = []
interval_records = []

for c in range(1, 17):
    sub = df_runs[df_runs['case'] == c]
    valid = sub[sub['status_flag'] == 'VALID'].sort_values('cumulative_time_min')
    
    mat = sub['material_name'].iloc[0]
    doc = sub['DOC_mm'].iloc[0]
    feed = sub['feed_mm_rev'].iloc[0]
    rep = sub['replicate'].iloc[0]
    
    # Table feed rate (vf in mm/min)
    vf = feed * SPINDLE_RPM
    # Official Productivity Metric: specific material removal rate (MRR' = ap * vf in mm²/min)
    mrr_prime = doc * vf
    
    total_runs = len(sub)
    valid_runs = len(valid)
    
    if c == 6:
        quality_flag = "ABORTED_EXPERIMENT (Only 1 Pass, VB=0.0)"
    elif c == 2:
        quality_flag = "VALID_TRAJECTORY (Run 1 Corrupted Sensor)"
    elif c == 12:
        quality_flag = "VALID_TRAJECTORY (Run 1 Corrupted Sensor & Missing VB)"
    else:
        quality_flag = "VALID_TRAJECTORY"
        
    # Tool life estimates at 0.30 mm (primary), 0.50 mm, 0.80 mm
    t_030, p_030, m_030, f_030 = estimate_tool_life(valid, 0.30)
    t_050, p_050, m_050, f_050 = estimate_tool_life(valid, 0.50)
    t_080, p_080, m_080, f_080 = estimate_tool_life(valid, 0.80)
    
    # Phase 2.5 Productivity-adjusted tool life metric: MRR' * T_0.30 (mm² per unit width)
    mat_removed_proxy_030 = (mrr_prime * t_030) if not np.isnan(t_030) else np.nan
    mat_removed_proxy_050 = (mrr_prime * t_050) if not np.isnan(t_050) else np.nan
    mat_removed_proxy_080 = (mrr_prime * t_080) if not np.isnan(t_080) else np.nan
    
    for thresh, t_val, p_val, meth, note, m_rem in [
        (0.30, t_030, p_030, m_030, f_030, mat_removed_proxy_030),
        (0.50, t_050, p_050, m_050, f_050, mat_removed_proxy_050),
        (0.80, t_080, p_080, m_080, f_080, mat_removed_proxy_080)
    ]:
        tool_life_records.append({
            'case': c,
            'replicate': rep,
            'material': mat,
            'DOC_mm': doc,
            'feed_mm_rev': feed,
            'feed_speed_mm_min': vf,
            'MRR_prime_mm2_min': mrr_prime,
            'threshold_VB_mm': thresh,
            'estimated_time_to_threshold_min': round(t_val, 2) if not np.isnan(t_val) else np.nan,
            'estimated_passes_to_threshold': round(p_val, 2) if not np.isnan(p_val) else np.nan,
            'material_removed_proxy_mm2': round(m_rem, 1) if not np.isnan(m_rem) else np.nan,
            'estimation_method': meth,
            'data_quality_flag': quality_flag if c == 6 else note
        })
        
    # Interval wear progression (ΔVB / Δtime)
    if len(valid) >= 2:
        t_arr = valid['cumulative_time_min'].values
        vb_arr = valid['VB_mm'].values
        p_arr = valid['run'].values
        
        dt_arr = np.diff(t_arr)
        dvb_arr = np.diff(vb_arr)
        int_rates = dvb_arr / dt_arr
        
        for k in range(len(dt_arr)):
            interval_records.append({
                'case': c,
                'replicate': rep,
                'material': mat,
                'DOC_mm': doc,
                'feed_mm_rev': feed,
                'interval_idx': k + 1,
                't_start_min': t_arr[k],
                't_end_min': t_arr[k+1],
                'delta_t_min': dt_arr[k],
                'VB_start_mm': vb_arr[k],
                'VB_end_mm': vb_arr[k+1],
                'delta_VB_mm': dvb_arr[k],
                'interval_wear_rate_mm_min': int_rates[k]
            })
            
        case_stats.append({
            'case': c,
            'replicate': rep,
            'material': mat,
            'DOC_mm': doc,
            'feed_mm_rev': feed,
            'vf_mm_min': vf,
            'MRR_prime_mm2_min': mrr_prime,
            'total_passes': total_runs,
            'valid_passes': valid_runs,
            'init_time_min': t_arr[0],
            'init_VB_mm': vb_arr[0],
            'final_time_min': t_arr[-1],
            'final_VB_mm': vb_arr[-1],
            'time_span_min': t_arr[-1] - t_arr[0],
            'total_delta_VB_mm': vb_arr[-1] - vb_arr[0],
            'median_interval_rate_mm_min': np.median(int_rates),
            'mean_interval_rate_mm_min': np.mean(int_rates),
            'min_interval_rate_mm_min': np.min(int_rates),
            'max_interval_rate_mm_min': np.max(int_rates),
            'time_to_030_min': t_030,
            'passes_to_030': p_030,
            'material_removed_proxy_030_mm2': mat_removed_proxy_030,
            'time_to_050_min': t_050,
            'time_to_080_min': t_080,
            'quality_flag': quality_flag
        })
    else:
        case_stats.append({
            'case': c,
            'replicate': rep,
            'material': mat,
            'DOC_mm': doc,
            'feed_mm_rev': feed,
            'vf_mm_min': vf,
            'MRR_prime_mm2_min': mrr_prime,
            'total_passes': total_runs,
            'valid_passes': valid_runs,
            'init_time_min': 0.0,
            'init_VB_mm': 0.0,
            'final_time_min': 0.0,
            'final_VB_mm': 0.0,
            'time_span_min': 0.0,
            'total_delta_VB_mm': 0.0,
            'median_interval_rate_mm_min': np.nan,
            'mean_interval_rate_mm_min': np.nan,
            'min_interval_rate_mm_min': np.nan,
            'max_interval_rate_mm_min': np.nan,
            'time_to_030_min': np.nan,
            'passes_to_030': np.nan,
            'material_removed_proxy_030_mm2': np.nan,
            'time_to_050_min': np.nan,
            'time_to_080_min': np.nan,
            'quality_flag': quality_flag
        })

df_tool_life = pd.DataFrame(tool_life_records)
df_case_stats = pd.DataFrame(case_stats)
df_intervals = pd.DataFrame(interval_records)

# Save requested tool-life summary table
TOOL_LIFE_CSV = DATA_DIR / "phase2_tool_life_metrics.csv"
df_tool_life.to_csv(TOOL_LIFE_CSV, index=False)
print(f"Saved tool-life metrics table to: {TOOL_LIFE_CSV}")

# Save case-level manufacturing summary table
CASE_METRICS_CSV = DATA_DIR / "phase2_case_manufacturing_metrics.csv"
df_case_stats.to_csv(CASE_METRICS_CSV, index=False)
print(f"Saved case manufacturing metrics to: {CASE_METRICS_CSV}")

# Save intervals dataset for reference
df_intervals.to_csv(DATA_DIR / "phase2_wear_intervals.csv", index=False)

# -------------------------------------------------------------
# 4. Generate 8-Condition Comparison Matrix (with Phase 2.5 Metrics)
# -------------------------------------------------------------
CONDITIONS = [
    {'cond_id': 1, 'material': 'Cast Iron', 'DOC': 0.75, 'feed': 0.25, 'cases': [3, 11]},
    {'cond_id': 2, 'material': 'Cast Iron', 'DOC': 0.75, 'feed': 0.50, 'cases': [2, 12]},
    {'cond_id': 3, 'material': 'Cast Iron', 'DOC': 1.50, 'feed': 0.25, 'cases': [4, 10]},
    {'cond_id': 4, 'material': 'Cast Iron', 'DOC': 1.50, 'feed': 0.50, 'cases': [1, 9]},
    {'cond_id': 5, 'material': 'Stainless Steel J45', 'DOC': 0.75, 'feed': 0.25, 'cases': [7, 13]},
    {'cond_id': 6, 'material': 'Stainless Steel J45', 'DOC': 0.75, 'feed': 0.50, 'cases': [8, 14]},
    {'cond_id': 7, 'material': 'Stainless Steel J45', 'DOC': 1.50, 'feed': 0.25, 'cases': [6, 15]},
    {'cond_id': 8, 'material': 'Stainless Steel J45', 'DOC': 1.50, 'feed': 0.50, 'cases': [5, 16]},
]

matrix_rows = []
for cond in CONDITIONS:
    cid = cond['cond_id']
    mat = cond['material']
    doc = cond['DOC']
    feed = cond['feed']
    c_r1, c_r2 = cond['cases']
    
    r1_stat = df_case_stats[df_case_stats['case'] == c_r1].iloc[0]
    r2_stat = df_case_stats[df_case_stats['case'] == c_r2].iloc[0]
    
    t_r1 = r1_stat['time_to_030_min']
    t_r2 = r2_stat['time_to_030_min']
    p_r1 = r1_stat['passes_to_030']
    p_r2 = r2_stat['passes_to_030']
    
    m_rem_r1 = r1_stat['material_removed_proxy_030_mm2']
    m_rem_r2 = r2_stat['material_removed_proxy_030_mm2']
    
    if np.isnan(t_r1) and np.isnan(t_r2):
        mean_t = np.nan
        std_t = np.nan
        mean_p = np.nan
        mean_m_rem = np.nan
        status = "Both Aborted"
    elif np.isnan(t_r1):
        mean_t = t_r2
        std_t = np.nan
        mean_p = p_r2
        mean_m_rem = m_rem_r2
        status = f"Rep 1 Aborted (Case {c_r1})"
    elif np.isnan(t_r2):
        mean_t = t_r1
        std_t = np.nan
        mean_p = p_r1
        mean_m_rem = m_rem_r1
        status = f"Rep 2 Aborted (Case {c_r2})"
    else:
        mean_t = (t_r1 + t_r2) / 2.0
        std_t = np.std([t_r1, t_r2], ddof=1)
        mean_p = (p_r1 + p_r2) / 2.0
        mean_m_rem = (m_rem_r1 + m_rem_r2) / 2.0
        status = "Both Replicates Valid"
        
    vf = feed * SPINDLE_RPM
    mrr_prime = doc * vf
    
    matrix_rows.append({
        'condition_id': cid,
        'material': mat,
        'DOC_mm': doc,
        'feed_mm_rev': feed,
        'feed_speed_mm_min': vf,
        'MRR_prime_mm2_min': mrr_prime,
        'case_rep1': c_r1,
        'case_rep2': c_r2,
        'status_note': status,
        't030_rep1_min': round(t_r1, 2) if not np.isnan(t_r1) else np.nan,
        't030_rep2_min': round(t_r2, 2) if not np.isnan(t_r2) else np.nan,
        'mean_t030_min': round(mean_t, 2) if not np.isnan(mean_t) else np.nan,
        'std_t030_min': round(std_t, 2) if not np.isnan(std_t) else np.nan,
        'mean_pass030': round(mean_p, 2) if not np.isnan(mean_p) else np.nan,
        'material_removed_proxy_030_rep1_mm2': round(m_rem_r1, 1) if not np.isnan(m_rem_r1) else np.nan,
        'material_removed_proxy_030_rep2_mm2': round(m_rem_r2, 1) if not np.isnan(m_rem_r2) else np.nan,
        'mean_material_removed_proxy_030_mm2': round(mean_m_rem, 1) if not np.isnan(mean_m_rem) else np.nan,
    })

df_matrix = pd.DataFrame(matrix_rows)
MATRIX_CSV = DATA_DIR / "phase2_condition_comparison_matrix.csv"
df_matrix.to_csv(MATRIX_CSV, index=False)
print(f"Saved condition comparison matrix to: {MATRIX_CSV}")

# -------------------------------------------------------------
# 5. Generate the 6 Required Phase 2 & 2.5 Visualizations
# -------------------------------------------------------------
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# -------------------------------------------------------------
# FIGURE 1: wear_trajectories_by_condition.png
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=True)
axes = axes.flatten()

for idx, cond in enumerate(CONDITIONS):
    ax = axes[idx]
    c_ids = cond['cases']
    mat = cond['material']
    doc = cond['DOC']
    feed = cond['feed']
    
    color_base = '#1f77b4' if mat == 'Cast Iron' else '#d62728'
    
    for c_id in c_ids:
        sub = df_runs[df_runs['case'] == c_id]
        val = sub[sub['status_flag'] == 'VALID'].sort_values('cumulative_time_min')
        rep = sub['replicate'].iloc[0]
        
        if c_id == 6:
            ax.scatter([0.0], [0.0], marker='x', s=100, color='darkred', lw=2, label='Case 6 (Rep 1: Aborted)')
            ax.text(5, 0.08, 'Case 6 Aborted (1 pass)', fontsize=8, color='darkred', style='italic')
        elif len(val) > 0:
            marker_shape = 'o' if rep == 1 else 's'
            line_style = '-' if rep == 1 else '--'
            alpha_val = 0.9 if rep == 1 else 0.75
            lbl = f"Case {c_id} (Rep {rep})"
            ax.plot(val['cumulative_time_min'], val['VB_mm'], 
                    marker=marker_shape, linestyle=line_style, color=color_base, 
                    alpha=alpha_val, markersize=5, lw=1.6, label=lbl)
            
    # Reference line for analysis threshold
    ax.axhline(0.30, color='#2c3e50', linestyle=':', lw=1.2, label='Analysis Benchmark (0.30 mm)' if idx == 0 else "")
    
    ax.set_title(f"Cond {cond['cond_id']}: {mat}\nDOC={doc} mm, Feed={feed} mm/rev", fontsize=10, fontweight='bold')
    ax.set_xlabel("Cumulative Machining Time (min)", fontsize=9)
    if idx % 4 == 0:
        ax.set_ylabel("Flank Wear VB (mm)", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.set_ylim(-0.02, 1.6)

plt.suptitle("Figure 1: Observed Flank Wear (VB) Trajectories Across 8 Cutting Conditions\n(Comparing Replicate 1 [-] vs Replicate 2 [--] for Cast Iron [Blue] and Stainless Steel J45 [Red])", 
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
fig1_path = FIGURES_DIR / "wear_trajectories_by_condition.png"
plt.savefig(fig1_path, dpi=220)
plt.close()
print(f"Saved Figure 1 to: {fig1_path}")

# -------------------------------------------------------------
# FIGURE 2: material_wear_comparison.png
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: Trajectories by Material
for c in range(1, 17):
    if c == 6: continue
    sub = df_runs[df_runs['case'] == c]
    val = sub[sub['status_flag'] == 'VALID'].sort_values('cumulative_time_min')
    mat = sub['material_name'].iloc[0]
    col = '#1f77b4' if mat == 'Cast Iron' else '#d62728'
    alpha = 0.55 if mat == 'Cast Iron' else 0.7
    lbl = mat if (c in [1, 5]) else ""
    ax1.plot(val['cumulative_time_min'], val['VB_mm'], color=col, alpha=alpha, lw=1.5, label=lbl)

ax1.axhline(0.30, color='#2c3e50', linestyle='--', lw=1.2, label='Analysis Benchmark (0.30 mm)')
ax1.set_xlabel("Cumulative Machining Time (min)", fontsize=10.5, fontweight='bold')
ax1.set_ylabel("Flank Wear VB (mm)", fontsize=10.5, fontweight='bold')
ax1.set_title("A) Tool Wear Progression Envelopes by Material", fontsize=11.5, fontweight='bold')
ax1.legend(loc='upper right', frameon=True)
ax1.grid(True, linestyle=':', alpha=0.5)

# Panel B: Interval Wear Rates Distribution (Boxplot / Scatter)
ci_intervals = df_intervals[df_intervals['material'] == 'Cast Iron']['interval_wear_rate_mm_min'].values
ss_intervals = df_intervals[df_intervals['material'] == 'Stainless Steel J45']['interval_wear_rate_mm_min'].values

box_data = [ci_intervals, ss_intervals]
bp = ax2.boxplot(box_data, patch_artist=True, tick_labels=['Cast Iron\n(N=80 intervals)', 'Stainless Steel J45\n(N=47 intervals)'],
                 widths=0.45, medianprops=dict(color='black', lw=2))

colors = ['#1f77b4', '#d62728']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Overlay individual jittered points
np.random.seed(42)
for i, vals in enumerate(box_data):
    jitter = np.random.normal(0, 0.04, size=len(vals))
    ax2.scatter(np.ones(len(vals))*(i+1) + jitter, vals, color=colors[i], edgecolors='black', alpha=0.6, s=30)

median_ci = np.median(ci_intervals)
median_ss = np.median(ss_intervals)
ax2.text(1.28, median_ci, f"Median:\n{median_ci:.4f} mm/min", ha='left', va='center', fontsize=8.5, fontweight='bold', color='navy')
ax2.text(2.28, median_ss, f"Median:\n{median_ss:.4f} mm/min", ha='left', va='center', fontsize=8.5, fontweight='bold', color='darkred')
ax2.set_xlim(0.5, 2.7)

ax2.set_ylabel("Observed Interval Wear Rate ΔVB / Δt (mm/min)", fontsize=10.5, fontweight='bold')
ax2.set_title("B) Distribution of Interval Wear Rates (ΔVB / Δt)", fontsize=11.5, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.5)

plt.suptitle("Figure 2: Observed Tool Wear Progression and Interval Wear Rates by Workpiece Material\n(Descriptive comparison showing faster observed wear rate in Stainless Steel J45 under the tested conditions)", 
             fontsize=12.5, fontweight='bold', y=0.98)
plt.tight_layout()
fig2_path = FIGURES_DIR / "material_wear_comparison.png"
plt.savefig(fig2_path, dpi=220)
plt.close()
print(f"Saved Figure 2 to: {fig2_path}")

# -------------------------------------------------------------
# FIGURE 3: feed_wear_comparison.png
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Panel A: Cast Iron Feed comparison
for c in [3, 11, 2, 12, 4, 10, 1, 9]:
    sub = df_runs[df_runs['case'] == c]
    val = sub[sub['status_flag'] == 'VALID'].sort_values('cumulative_time_min')
    feed = sub['feed_mm_rev'].iloc[0]
    col = '#3498db' if feed == 0.25 else '#e67e22'
    style = '-' if sub['replicate'].iloc[0] == 1 else '--'
    lbl = f"f = {feed} mm/rev" if c in [3, 2] else ""
    ax1.plot(val['cumulative_time_min'], val['VB_mm'], color=col, linestyle=style, lw=1.5, alpha=0.85, label=lbl)

ax1.axhline(0.30, color='#2c3e50', linestyle=':', lw=1.2, label='Benchmark (0.30 mm)')
ax1.set_title("A) Cast Iron: Feed Rate Comparison (0.25 vs 0.50 mm/rev)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Cumulative Machining Time (min)", fontsize=10, fontweight='bold')
ax1.set_ylabel("Flank Wear VB (mm)", fontsize=10, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.5)

# Panel B: Stainless Steel J45 Feed comparison
for c in [7, 13, 8, 14, 15, 5, 16]:
    sub = df_runs[df_runs['case'] == c]
    val = sub[sub['status_flag'] == 'VALID'].sort_values('cumulative_time_min')
    feed = sub['feed_mm_rev'].iloc[0]
    col = '#3498db' if feed == 0.25 else '#e67e22'
    style = '-' if sub['replicate'].iloc[0] == 1 else '--'
    lbl = f"f = {feed} mm/rev" if c in [7, 8] else ""
    ax2.plot(val['cumulative_time_min'], val['VB_mm'], color=col, linestyle=style, lw=1.5, alpha=0.85, label=lbl)

ax2.axhline(0.30, color='#2c3e50', linestyle=':', lw=1.2)
ax2.set_title("B) Stainless Steel J45: Feed Rate Comparison (0.25 vs 0.50 mm/rev)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Cumulative Machining Time (min)", fontsize=10, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.5)

plt.suptitle("Figure 3: Observed Relationship Between Feed Rate (mm/rev) and Wear Progression\n(Solid lines = Replicate 1, Dashed lines = Replicate 2; Higher feed associates with steeper trajectories in both materials)", 
             fontsize=12.5, fontweight='bold', y=0.98)
plt.tight_layout()
fig3_path = FIGURES_DIR / "feed_wear_comparison.png"
plt.savefig(fig3_path, dpi=220)
plt.close()
print(f"Saved Figure 3 to: {fig3_path}")

# -------------------------------------------------------------
# FIGURE 4: doc_wear_comparison.png
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Panel A: Cast Iron DOC comparison
for c in [3, 11, 2, 12, 4, 10, 1, 9]:
    sub = df_runs[df_runs['case'] == c]
    val = sub[sub['status_flag'] == 'VALID'].sort_values('cumulative_time_min')
    doc = sub['DOC_mm'].iloc[0]
    col = '#2ecc71' if doc == 0.75 else '#9b59b6'
    style = '-' if sub['replicate'].iloc[0] == 1 else '--'
    lbl = f"DOC = {doc} mm" if c in [3, 4] else ""
    ax1.plot(val['cumulative_time_min'], val['VB_mm'], color=col, linestyle=style, lw=1.5, alpha=0.85, label=lbl)

ax1.axhline(0.30, color='#2c3e50', linestyle=':', lw=1.2, label='Benchmark (0.30 mm)')
ax1.set_title("A) Cast Iron: DOC Comparison (0.75 vs 1.50 mm)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Cumulative Machining Time (min)", fontsize=10, fontweight='bold')
ax1.set_ylabel("Flank Wear VB (mm)", fontsize=10, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.5)

# Panel B: Stainless Steel J45 DOC comparison
for c in [7, 13, 8, 14, 15, 5, 16]:
    sub = df_runs[df_runs['case'] == c]
    val = sub[sub['status_flag'] == 'VALID'].sort_values('cumulative_time_min')
    doc = sub['DOC_mm'].iloc[0]
    col = '#2ecc71' if doc == 0.75 else '#9b59b6'
    style = '-' if sub['replicate'].iloc[0] == 1 else '--'
    lbl = f"DOC = {doc} mm" if c in [7, 15] else ""
    ax2.plot(val['cumulative_time_min'], val['VB_mm'], color=col, linestyle=style, lw=1.5, alpha=0.85, label=lbl)

ax2.axhline(0.30, color='#2c3e50', linestyle=':', lw=1.2)
ax2.set_title("B) Stainless Steel J45: DOC Comparison (0.75 vs 1.50 mm)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Cumulative Machining Time (min)", fontsize=10, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.5)

plt.suptitle("Figure 4: Observed Relationship Between Axial Depth of Cut (DOC) and Wear Progression\n(Green = 0.75 mm, Purple = 1.50 mm; Increasing DOC associates with reduced machining time before reaching 0.30 mm)", 
             fontsize=12.5, fontweight='bold', y=0.98)
plt.tight_layout()
fig4_path = FIGURES_DIR / "doc_wear_comparison.png"
plt.savefig(fig4_path, dpi=220)
plt.close()
print(f"Saved Figure 4 to: {fig4_path}")

# -------------------------------------------------------------
# FIGURE 5: tool_life_comparison.png
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6.5))

cond_labels = [
    "C1: Cast Iron\nd0.75 f0.25", "C2: Cast Iron\nd0.75 f0.50", "C3: Cast Iron\nd1.50 f0.25", "C4: Cast Iron\nd1.50 f0.50",
    "C5: Stainless J45\nd0.75 f0.25", "C6: Stainless J45\nd0.75 f0.50", "C7: Stainless J45\nd1.50 f0.25", "C8: Stainless J45\nd1.50 f0.50"
]

x_pos = np.arange(len(cond_labels))
bar_w = 0.35

r1_lifes = []
r2_lifes = []

for cond in CONDITIONS:
    c_rep1 = cond['cases'][0]
    c_rep2 = cond['cases'][1]
    
    row1 = df_case_stats[df_case_stats['case'] == c_rep1].iloc[0]
    row2 = df_case_stats[df_case_stats['case'] == c_rep2].iloc[0]
    
    r1_lifes.append(row1['time_to_030_min'] if not np.isnan(row1['time_to_030_min']) else 0.0)
    r2_lifes.append(row2['time_to_030_min'] if not np.isnan(row2['time_to_030_min']) else 0.0)

b1 = ax.bar(x_pos - bar_w/2, r1_lifes, bar_w, label='Replicate 1', color='#2b5c8f', edgecolor='black')
b2 = ax.bar(x_pos + bar_w/2, r2_lifes, bar_w, label='Replicate 2', color='#6baed6', edgecolor='black', hatch='//')

# Annotate Case 6
ax.text(6 - bar_w/2, 2, "Aborted\n(Case 6)", ha='center', va='bottom', fontsize=8, color='crimson', fontweight='bold')

# Values above bars
for i in range(8):
    if r1_lifes[i] > 0:
        ax.text(x_pos[i] - bar_w/2, r1_lifes[i] + 1.0, f"{r1_lifes[i]:.1f}m", ha='center', va='bottom', fontsize=8)
    if r2_lifes[i] > 0:
        ax.text(x_pos[i] + bar_w/2, r2_lifes[i] + 1.0, f"{r2_lifes[i]:.1f}m", ha='center', va='bottom', fontsize=8)

ax.set_xticks(x_pos)
ax.set_xticklabels(cond_labels, fontsize=9.5)
ax.set_ylabel("Estimated Machining Time to VB = 0.30 mm (Minutes)", fontsize=11, fontweight='bold')
ax.set_title("Figure 5: Estimated Tool Life to Project Analysis Benchmark (VB = 0.30 mm)\n(Comparing Replicate 1 vs Replicate 2 Across 8 Cutting Conditions)", fontsize=12.5, fontweight='bold', pad=15)
ax.legend(loc='upper right')
ax.grid(True, linestyle=':', alpha=0.5)
ax.set_ylim(0, 68)

plt.tight_layout()
fig5_path = FIGURES_DIR / "tool_life_comparison.png"
plt.savefig(fig5_path, dpi=220)
plt.close()
print(f"Saved Figure 5 to: {fig5_path}")

# -------------------------------------------------------------
# FIGURE 6: productivity_vs_tool_life.png (Enhanced 2-Panel)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

valid_cases = df_case_stats[df_case_stats['quality_flag'] != 'ABORTED_EXPERIMENT (Only 1 Pass, VB=0.0)'].copy()

# Panel A: Productivity Proxy vs Tool Life (Minutes)
markers = {1: 'o', 2: 's'}
for mat, col in [('Cast Iron', '#1f77b4'), ('Stainless Steel J45', '#d62728')]:
    mat_sub = valid_cases[valid_cases['material'] == mat]
    for rep in [1, 2]:
        rep_sub = mat_sub[mat_sub['replicate'] == rep]
        ax1.scatter(rep_sub['MRR_prime_mm2_min'], rep_sub['time_to_030_min'], 
                    color=col, marker=markers[rep], s=110, edgecolors='black', lw=1.2,
                    label=f"{mat} (Rep {rep})", alpha=0.85)

for _, r in valid_cases.iterrows():
    ax1.annotate(f"C{int(r['case'])}", (r['MRR_prime_mm2_min'], r['time_to_030_min']),
                 textcoords="offset points", xytext=(5, 3), fontsize=8, alpha=0.85)

ax1.axvline(309.75, color='gray', linestyle='--', lw=1.2, alpha=0.7)
ax1.text(315, 52, "Equal Productivity (MRR' = 309.8 mm²/min)\n• High Feed (d0.75, f0.50): Life ~40-44 min\n• High DOC (d1.50, f0.25): Life ~20-24 min", 
         fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="gray", alpha=0.9))

ax1.set_xlabel("Productivity Proxy: MRR' = ap × vf (mm²/min)", fontsize=10.5, fontweight='bold')
ax1.set_ylabel("Tool Life to VB = 0.30 mm (Minutes)", fontsize=10.5, fontweight='bold')
ax1.set_title("A) Productivity Proxy vs. Tool Life Duration", fontsize=11.5, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.legend(loc='upper right', frameon=True, fontsize=8.5)

# Panel B: Productivity-Adjusted Tool Life: Cumulative Material Removed Proxy (mm² per unit width)
cond_names = [f"C{r['condition_id']}:\n{r['material'].split()[0]}\nd{r['DOC_mm']} f{r['feed_mm_rev']}" for _, r in df_matrix.iterrows()]
x_m = np.arange(len(cond_names))
m_rem_vals = df_matrix['mean_material_removed_proxy_030_mm2'].values

bar_cols = ['#1f77b4' if 'Cast' in m else '#d62728' for m in df_matrix['material']]
bars = ax2.bar(x_m, m_rem_vals, width=0.55, color=bar_cols, edgecolor='black', alpha=0.85)

for idx, bar in enumerate(bars):
    val = m_rem_vals[idx]
    if not np.isnan(val):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 300, f"{val:.0f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    else:
        ax2.text(bar.get_x() + bar.get_width()/2, 300, "Aborted", ha='center', va='bottom', fontsize=8, color='crimson')

# Highlight Equal-Productivity Trade-Off (C2 vs C3)
ax2.annotate("Equal MRR': High Feed yields\n~1.91× higher removal proxy",
             xy=(1, m_rem_vals[1]), xytext=(1.8, 14500),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color="darkblue", lw=1.5),
             fontsize=9, fontweight='bold', color='darkblue',
             bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec="darkblue"))

ax2.set_xticks(x_m)
ax2.set_xticklabels(cond_names, fontsize=8.5)
ax2.set_ylabel("Productivity-Adjusted Tool-Life Proxy:\nMRR' × T_0.30 (mm² per unit width)", fontsize=10.5, fontweight='bold')
ax2.set_title("B) Phase 2.5: Cumulative Material-Removal Proxy before VB = 0.30 mm", fontsize=11.5, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.set_ylim(0, 19500)

plt.suptitle("Figure 6: Manufacturing Trade-Off & Productivity-Adjusted Tool-Life Proxy\n(Under identical MRR' = 309.8 mm²/min, High Feed / Low DOC achieves ~1.91× higher cumulative removal proxy before reaching VB = 0.30 mm)", 
             fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout()
fig6_path = FIGURES_DIR / "productivity_vs_tool_life.png"
plt.savefig(fig6_path, dpi=220)
plt.close()
print(f"Saved Figure 6 to: {fig6_path}")

print("\n" + "="*80)
print("PHASE 2 & PHASE 2.5 PIPELINE COMPLETED SUCCESSFULLY!")
print("="*80)
