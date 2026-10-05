import sys
from pathlib import Path
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# STEP 1 & 2: Verify mill.mat structure & README definitions
# -------------------------------------------------------------
MAT_PATH = DATA_DIR / "mill.mat"
assert MAT_PATH.exists(), f"Missing {MAT_PATH}"

mat = sio.loadmat(str(MAT_PATH))
mill = mat['mill']
n_runs = mill.shape[1]

# -------------------------------------------------------------
# STEP 3 & 4: Build clean run-level dataframe & Map Case conditions
# -------------------------------------------------------------
# Material 1 = Cast Iron, Material 2 = Stainless Steel J45
# Cases 1-8: Replicate 1 (First set of inserts)
# Cases 9-16: Replicate 2 (Second set of inserts)
records = []
sensors = ['smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']

for i in range(n_runs):
    item = mill[0, i]
    case_val = int(item['case'][0, 0])
    run_val = int(item['run'][0, 0])
    
    vb_raw = item['VB']
    vb_val = float(vb_raw[0, 0]) if vb_raw.size > 0 else np.nan
    
    time_val = float(item['time'][0, 0]) if item['time'].size > 0 else np.nan
    doc_val = float(item['DOC'][0, 0]) if item['DOC'].size > 0 else np.nan
    feed_val = float(item['feed'][0, 0]) if item['feed'].size > 0 else np.nan
    mat_val = int(item['material'][0, 0]) if item['material'].size > 0 else np.nan
    
    # Check signal corruption
    is_corrupt = False
    for s in sensors:
        sig = item[s].flatten()
        if (np.abs(sig) > 50).any() or (np.abs(sig).mean() < 1e-5):
            is_corrupt = True
            break
            
    is_labeled = not np.isnan(vb_val)
    
    if is_corrupt:
        status = 'CORRUPTED_SIGNAL'
    elif not is_labeled:
        status = 'MISSING_VB'
    else:
        status = 'VALID'
        
    records.append({
        'run_idx': i + 1,
        'case': case_val,
        'replicate': 1 if case_val <= 8 else 2,
        'run': run_val,
        'material_code': mat_val,
        'material_name': 'Cast Iron' if mat_val == 1 else 'Stainless Steel J45',
        'DOC_mm': doc_val,
        'feed_mm_rev': feed_val,
        'speed_m_min': 200.0,
        'spindle_rpm': 826.0,
        'cumulative_time_min': time_val,
        'VB_mm': vb_val,
        'is_labeled': is_labeled,
        'is_corrupted': is_corrupt,
        'status_flag': status,
        'signal_samples': item['smcAC'].size
    })

df_foundation = pd.DataFrame(records)
df_foundation.to_csv(DATA_DIR / "mill_runs_foundation.csv", index=False)
print(f"Generated clean foundation dataframe: {DATA_DIR / 'mill_runs_foundation.csv'} ({len(df_foundation)} rows)")

# -------------------------------------------------------------
# STEP 5 & 6: Reconstruct Trajectory & Flag Missing/Corrupt Runs
# -------------------------------------------------------------
print("\n" + "="*75)
print("STATUS BREAKDOWN ACROSS 167 RUNS:")
print(df_foundation['status_flag'].value_counts())
print("="*75)

# STEP 7: Produce Consolidated Data Audit Table (16 Cases)
case_audit = []
for c in range(1, 17):
    sub = df_foundation[df_foundation['case'] == c]
    labeled = sub[sub['status_flag'] == 'VALID']
    mat_name = sub['material_name'].iloc[0]
    doc = sub['DOC_mm'].iloc[0]
    feed = sub['feed_mm_rev'].iloc[0]
    rep = sub['replicate'].iloc[0]
    
    total_runs = len(sub)
    valid_runs = len(labeled)
    missing_runs = len(sub[sub['status_flag'] == 'MISSING_VB'])
    corrupt_runs = len(sub[sub['status_flag'] == 'CORRUPTED_SIGNAL'])
    
    max_t = sub['cumulative_time_min'].max()
    vb_start = labeled['VB_mm'].iloc[0] if len(labeled) > 0 else np.nan
    vb_end = labeled['VB_mm'].iloc[-1] if len(labeled) > 0 else np.nan
    
    # Time and pass to reach threshold VB >= 0.30 mm
    reach_030 = labeled[labeled['VB_mm'] >= 0.30]
    if len(reach_030) > 0:
        pass_030 = str(reach_030['run'].iloc[0])
        time_030 = f"{reach_030['cumulative_time_min'].iloc[0]:.1f}"
    else:
        pass_030 = "Not reached"
        time_030 = "Not reached"
        
    # Empirical wear rate (mm/min) over the observed cutting time
    if len(labeled) >= 2:
        t_span = labeled['cumulative_time_min'].iloc[-1] - labeled['cumulative_time_min'].iloc[0]
        rate = (vb_end - vb_start) / t_span if t_span > 0 else np.nan
    else:
        rate = np.nan
        
    case_audit.append({
        'Case': c,
        'Rep': rep,
        'Material': 'Cast Iron' if mat_name == 'Cast Iron' else 'Steel J45',
        'DOC': doc,
        'Feed': feed,
        'Passes': total_runs,
        'Valid': valid_runs,
        'Missing': missing_runs,
        'Corrupt': corrupt_runs,
        'MaxTime (min)': max_t,
        'Start VB': vb_start,
        'Final VB': vb_end,
        'Pass to 0.30mm': pass_030,
        'Time to 0.30mm': time_030,
        'Wear Rate (mm/min)': f"{rate:.4f}" if not np.isnan(rate) else "N/A"
    })

audit_table = pd.DataFrame(case_audit)
audit_table.to_csv(DATA_DIR / "case_audit_table.csv", index=False)
print("\n=== CONSOLIDATED DATA AUDIT TABLE (16 CASES) ===")
print(audit_table.to_string())

# -------------------------------------------------------------
# PLOT: VB vs Cumulative Machining Time (Trajectories for 16 Tools)
# -------------------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharey=True, sharex=False)
axes = axes.flatten()

# Grouping conditions to compare Replicate 1 vs Replicate 2
colors_mat = {'Cast Iron': '#1f77b4', 'Steel J45': '#d62728'}

for c in range(1, 17):
    ax = axes[c - 1]
    sub = df_foundation[df_foundation['case'] == c]
    mat = 'Cast Iron' if sub['material_code'].iloc[0] == 1 else 'Steel J45'
    doc = sub['DOC_mm'].iloc[0]
    feed = sub['feed_mm_rev'].iloc[0]
    rep = sub['replicate'].iloc[0]
    
    # Plot valid wear trajectory
    valid = sub[sub['status_flag'] == 'VALID']
    missing = sub[sub['status_flag'] == 'MISSING_VB']
    corrupt = sub[sub['status_flag'] == 'CORRUPTED_SIGNAL']
    
    color = colors_mat[mat]
    ax.plot(valid['cumulative_time_min'], valid['VB_mm'], 'o-', color=color, markersize=5, lw=1.5, label='Measured VB')
    
    # Mark missing runs along the time axis as faint gray crosses
    if len(missing) > 0:
        ax.scatter(missing['cumulative_time_min'], [0.0]*len(missing), marker='x', color='gray', s=30, alpha=0.6, label='Unmeasured Pass')
        
    # Mark corrupt runs
    if len(corrupt) > 0:
        ax.scatter(corrupt['cumulative_time_min'], [0.0]*len(corrupt), marker='^', color='orange', s=50, label='Corrupted Signal')
        
    # Analysis threshold
    ax.axhline(0.30, color='#2c3e50', linestyle='--', alpha=0.7, lw=1, label='Analysis Threshold (0.30 mm)')
    
    # Title & Labels
    ax.set_title(f"Case {c} (Rep {rep}) | {mat}\nDOC={doc}mm, f={feed}mm/rev", fontsize=8.5, fontweight='bold')
    ax.set_xlabel("Cumulative Time (min)", fontsize=7.5)
    ax.set_ylabel("Flank Wear VB (mm)", fontsize=7.5)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.05, 1.6)
    
    if c == 1:
        ax.legend(fontsize=6.5, loc='upper left')

plt.suptitle("Figure: Flank Wear (VB) Progression vs Cumulative Machining Time Across All 16 Tool Inserts\n(Comparing Cast Iron vs Stainless Steel J45 under DOC and Feed Combinations)", fontsize=13, y=0.99)
plt.tight_layout()
fig_path = FIGURES_DIR / "vb_vs_cumulative_time_trajectories.png"
plt.savefig(fig_path, dpi=220)
plt.close()
print(f"\nSaved trajectory visualization to: {fig_path}")
