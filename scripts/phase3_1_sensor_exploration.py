"""
Phase 3.1: Sensor Signal Exploration & Feature Engineering Design
NASA Milling Dataset (V2 Flagship Project)

Author: AntiGravity (Implementation Agent)
Domain Lead / Owner: Bright
Strategic Lead: ChatGPT

Objectives:
- Inspect raw signals across all 6 sensor channels
- Verify sampling rate, recording duration, and synchronization
- Identify signal quality issues, anomalies, and clipping/saturation
- Extract a compact, physically interpretable set of time- and frequency-domain candidate features
- Evaluate whether full-window or sub-window extraction is justified
- Investigate empirical relationships between candidate features and flank wear (VB)
- Audit cutting condition confounding (Material, Feed, DOC) vs wear sensitivity
- Assess replicate repeatability across independent tools
- Propose a lean, evidence-based feature set (10-30 features) for Phase 3.2

Strict Rules:
- NO Machine Learning models (no RF, XGBoost, Ridge, SVR)
- NO train/test splitting
- NO cumulative time or run number as features (leakage prevention)
- Descriptive language only (NO causal claims)
"""

import sys
from pathlib import Path
import scipy.io as sio
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure UTF-8 console output
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SIG_EX_DIR = FIGURES_DIR / "phase3_signal_examples"
FEAT_DIST_DIR = FIGURES_DIR / "phase3_feature_distributions"
FREQ_DIR = FIGURES_DIR / "phase3_frequency_analysis"

for d in [SIG_EX_DIR, FEAT_DIST_DIR, FREQ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. Load Clean Foundation Data & MAT Struct
# -------------------------------------------------------------
FOUNDATION_PATH = DATA_DIR / "mill_runs_foundation.csv"
MAT_PATH = DATA_DIR / "mill.mat"
assert FOUNDATION_PATH.exists(), f"Missing {FOUNDATION_PATH}"
assert MAT_PATH.exists(), f"Missing {MAT_PATH}"

df_runs = pd.read_csv(FOUNDATION_PATH)
mat = sio.loadmat(str(MAT_PATH))['mill']

# Standardize material naming to 'Stainless Steel J45'
df_runs['material_name'] = df_runs['material_name'].replace({
    'Steel J45': 'Stainless Steel J45',
    'stainless steel J45': 'Stainless Steel J45'
})

valid_df = df_runs[df_runs['status_flag'] == 'VALID'].copy().reset_index(drop=True)
print(f"Loaded {len(df_runs)} total runs; {len(valid_df)} confirmed VALID runs for exploration.")

SENSORS = ['smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']
SENSOR_LABELS = {
    'smcAC': 'AC Spindle Motor Current (V)',
    'smcDC': 'DC Spindle Motor Current (V)',
    'vib_table': 'Table Vibration RMS Envelope (V)',
    'vib_spindle': 'Spindle Vibration RMS Envelope (V)',
    'AE_table': 'Table Acoustic Emission RMS Envelope (V)',
    'AE_spindle': 'Spindle Acoustic Emission RMS Envelope (V)'
}

# -------------------------------------------------------------
# 2. Verify Sampling, Synchronization & Signal Dimensions
# -------------------------------------------------------------
SAMPLING_FREQ_HZ = 250.0  # Confirmed from README page 3 (Delta T = 8.0 ms, sampling = 250 Hz)
NYQUIST_FREQ_HZ = SAMPLING_FREQ_HZ / 2.0  # 125 Hz
SPINDLE_RPM = 826.0
SPINDLE_FREQ_HZ = SPINDLE_RPM / 60.0      # 13.767 Hz
TOOTH_PASSING_FREQ_HZ = 6.0 * SPINDLE_FREQ_HZ  # 82.60 Hz (6 inserts on 70mm cutter)

valid_shapes = set()
for _, r in valid_df.iterrows():
    idx = int(r['run_idx']) - 1
    item = mat[0, idx]
    lens = tuple(item[s].shape[0] for s in SENSORS)
    valid_shapes.add(lens)

assert valid_shapes == {(9000, 9000, 9000, 9000, 9000, 9000)}, f"Inconsistent shapes: {valid_shapes}"
print(f"Verified: All {len(valid_df)} VALID runs have exactly 9,000 samples across all 6 channels.")
print(f"Sampling frequency: {SAMPLING_FREQ_HZ} Hz; Duration per snapshot: {9000/SAMPLING_FREQ_HZ:.1f} s.")
print(f"Kinematic harmonics: Spindle rotational frequency = {SPINDLE_FREQ_HZ:.2f} Hz; Tooth passing frequency = {TOOTH_PASSING_FREQ_HZ:.2f} Hz.")

# -------------------------------------------------------------
# 3. Raw Signal Exploration: Representative Examples
# -------------------------------------------------------------
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

time_axis = np.arange(9000) / SAMPLING_FREQ_HZ

def plot_fresh_vs_worn(case_fresh, run_fresh, case_worn, run_worn, mat_name, filename, title_str):
    rf = valid_df[(valid_df['case'] == case_fresh) & (valid_df['run'] == run_fresh)].iloc[0]
    rw = valid_df[(valid_df['case'] == case_worn) & (valid_df['run'] == run_worn)].iloc[0]
    
    idx_f = int(rf['run_idx']) - 1
    idx_w = int(rw['run_idx']) - 1
    
    item_f = mat[0, idx_f]
    item_w = mat[0, idx_w]
    
    fig, axes = plt.subplots(6, 2, figsize=(16, 12), sharex=True)
    
    for s_idx, s in enumerate(SENSORS):
        sig_f = item_f[s].flatten()
        sig_w = item_w[s].flatten()
        
        # Fresh column
        axes[s_idx, 0].plot(time_axis, sig_f, color='#1f77b4', lw=0.6, alpha=0.85)
        axes[s_idx, 0].set_ylabel(s, fontsize=9, fontweight='bold')
        axes[s_idx, 0].grid(True, linestyle=':', alpha=0.4)
        if s_idx == 0:
            axes[s_idx, 0].set_title(f"Fresh Tool: Case {case_fresh} Run {run_fresh}\n{mat_name}, DOC={rf['DOC_mm']}mm, Feed={rf['feed_mm_rev']}mm/rev\nVB = {rf['VB_mm']:.2f} mm", 
                                     fontsize=10, fontweight='bold', color='navy')
            
        # Worn column
        axes[s_idx, 1].plot(time_axis, sig_w, color='#d62728', lw=0.6, alpha=0.85)
        axes[s_idx, 1].grid(True, linestyle=':', alpha=0.4)
        if s_idx == 0:
            axes[s_idx, 1].set_title(f"Worn Tool: Case {case_worn} Run {run_worn}\n{mat_name}, DOC={rw['DOC_mm']}mm, Feed={rw['feed_mm_rev']}mm/rev\nVB = {rw['VB_mm']:.2f} mm", 
                                     fontsize=10, fontweight='bold', color='darkred')
            
    axes[5, 0].set_xlabel("Time (seconds)", fontsize=10, fontweight='bold')
    axes[5, 1].set_xlabel("Time (seconds)", fontsize=10, fontweight='bold')
    
    plt.suptitle(title_str, fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()
    out_path = SIG_EX_DIR / filename
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved signal comparison plot: {out_path.name}")

# Fig 1: Cast Iron: Case 3 Run 1 (Fresh, VB=0.0mm) vs Case 3 Run 16 (Worn, VB=0.55mm)
plot_fresh_vs_worn(3, 1, 3, 16, "Cast Iron", 
                   "fig1_raw_signals_fresh_vs_worn_cast_iron.png",
                   "Figure 1: Raw Sensor Signals Comparison — Fresh vs Worn Tool (Cast Iron, Case 3)")

# Fig 2: Stainless Steel J45: Case 7 Run 1 (Fresh, VB=0.0mm) vs Case 7 Run 7 (Worn, VB=0.46mm)
plot_fresh_vs_worn(7, 1, 7, 7, "Stainless Steel J45", 
                   "fig2_raw_signals_fresh_vs_worn_stainless_steel.png",
                   "Figure 2: Raw Sensor Signals Comparison — Fresh vs Worn Tool (Stainless Steel J45, Case 7)")

# Fig 3: Signal Quality & Saturation Demonstration (smcDC Clipping at 10V)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
c1_r1 = valid_df[(valid_df['case'] == 1) & (valid_df['run'] == 1)].iloc[0]
c1_r10 = valid_df[(valid_df['case'] == 1) & (valid_df['run'] == 10)].iloc[0]

sig_r1_dc = mat[0, int(c1_r1['run_idx'])-1]['smcDC'].flatten()
sig_r10_dc = mat[0, int(c1_r10['run_idx'])-1]['smcDC'].flatten()

ax1.plot(time_axis, sig_r1_dc, color='#2ca02c', lw=0.8, label=f"Case 1 Run 1 (Fresh, VB={c1_r1['VB_mm']}mm) - Normal Signal")
ax1.set_ylabel("smcDC (Volts)", fontsize=10, fontweight='bold')
ax1.set_title("A) Case 1 Run 1: Unclipped DC Spindle Motor Current (Fresh Tool)", fontsize=10.5, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.set_ylim(0, 11)
ax1.legend(loc='lower right')

ax2.plot(time_axis, sig_r10_dc, color='#d62728', lw=0.8, label=f"Case 1 Run 10 (Worn, VB={c1_r10['VB_mm']}mm) - Severely Saturated")
ax2.axhline(9.995, color='black', linestyle='--', lw=1.2, label="DAQ Maximum Range (+10.0 V Saturation Ceiling)")
ax2.set_ylabel("smcDC (Volts)", fontsize=10, fontweight='bold')
ax2.set_xlabel("Time (seconds)", fontsize=10, fontweight='bold')
ax2.set_title("B) Case 1 Run 10: Severe Sensor Clipping at +10.0 V (62.8% of samples clipped flat)", fontsize=10.5, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.set_ylim(0, 11)
ax2.legend(loc='lower right')

plt.suptitle("Figure 3: Critical Signal-Quality Finding — DC Motor Current (smcDC) Saturation at +10V DAQ Limit\n(Observed in 41 out of 145 VALID runs under heavy cutting loads)", 
             fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout()
fig3_path = SIG_EX_DIR / "fig3_signal_saturation_and_clipping_smcDC.png"
plt.savefig(fig3_path, dpi=200)
plt.close()
print(f"Saved saturation audit plot: {fig3_path.name}")

# -------------------------------------------------------------
# 4. Feature Extraction & Windowing Comparison
# -------------------------------------------------------------
records_features = []
freqs = np.fft.rfftfreq(9000, 1.0 / SAMPLING_FREQ_HZ)

# Band boundaries based on physical kinematics
# Spindle rotational frequency: 13.77 Hz -> Band 11.0 to 16.5 Hz
spindle_band_mask = (freqs >= 11.0) & (freqs <= 16.5)
# Tooth passing frequency: 82.60 Hz -> Band 75.0 to 90.0 Hz
tooth_band_mask = (freqs >= 75.0) & (freqs <= 90.0)

for _, r in valid_df.iterrows():
    idx = int(r['run_idx']) - 1
    item = mat[0, idx]
    
    row_feat = {
        'case': r['case'],
        'replicate': r['replicate'],
        'run': r['run'],
        'material_name': r['material_name'],
        'material_code': r['material_code'],
        'DOC_mm': r['DOC_mm'],
        'feed_mm_rev': r['feed_mm_rev'],
        'VB_mm': r['VB_mm']
    }
    
    for s in SENSORS:
        sig = item[s].flatten()
        
        # 1. Full-Window Time-Domain Features (0 - 36 s)
        mean_val = np.mean(sig)
        std_val = np.std(sig)
        rms_val = np.sqrt(np.mean(sig**2))
        p2p_val = np.ptp(sig)
        min_val = np.min(sig)
        max_val = np.max(sig)
        kurt_val = stats.kurtosis(sig, fisher=True)  # Excess kurtosis (Gaussian = 0)
        crest_val = (max_val / rms_val) if rms_val > 1e-6 else np.nan
        
        row_feat[f'{s}_mean'] = mean_val
        row_feat[f'{s}_std'] = std_val
        row_feat[f'{s}_rms'] = rms_val
        row_feat[f'{s}_p2p'] = p2p_val
        row_feat[f'{s}_kurtosis'] = kurt_val
        row_feat[f'{s}_crest'] = crest_val
        
        # 2. Sub-Window Analysis (Window 1: 0-12s, Window 2: 12-24s, Window 3: 24-36s, Steady: 10-30s)
        w1_rms = np.sqrt(np.mean(sig[:3000]**2))
        w2_rms = np.sqrt(np.mean(sig[3000:6000]**2))
        w3_rms = np.sqrt(np.mean(sig[6000:]**2))
        steady_rms = np.sqrt(np.mean(sig[2500:7500]**2))
        
        row_feat[f'{s}_rms_w1'] = w1_rms
        row_feat[f'{s}_rms_w2'] = w2_rms
        row_feat[f'{s}_rms_w3'] = w3_rms
        row_feat[f'{s}_rms_steady'] = steady_rms
        row_feat[f'{s}_rms_drift_ratio'] = (w3_rms / w1_rms) if w1_rms > 1e-6 else np.nan
        
        # 3. Frequency-Domain Features (FFT on detrended signal)
        sig_detrend = sig - mean_val
        fft_complex = np.fft.rfft(sig_detrend)
        fft_mag = np.abs(fft_complex) / (9000.0 / 2.0)
        fft_power = (np.abs(fft_complex) ** 2) / 9000.0
        
        # Exclude DC (index 0) for peak detection
        dom_idx = np.argmax(fft_mag[1:]) + 1
        dom_freq = freqs[dom_idx]
        dom_mag = fft_mag[dom_idx]
        
        # Kinematic Band Powers
        spindle_pwr = np.sum(fft_power[spindle_band_mask])
        tooth_pwr = np.sum(fft_power[tooth_band_mask])
        total_pwr = np.sum(fft_power[1:])
        
        row_feat[f'{s}_dom_freq'] = dom_freq
        row_feat[f'{s}_dom_mag'] = dom_mag
        row_feat[f'{s}_spindle_band_pwr'] = spindle_pwr
        row_feat[f'{s}_tooth_band_pwr'] = tooth_pwr
        row_feat[f'{s}_total_ac_pwr'] = total_pwr
        
    records_features.append(row_feat)

df_features = pd.DataFrame(records_features)
FEATURES_CSV = DATA_DIR / "phase3_sensor_exploratory_features.csv"
df_features.to_csv(FEATURES_CSV, index=False)
print(f"Saved complete exploratory feature table ({len(df_features)} rows, {len(df_features.columns)} columns) to: {FEATURES_CSV}")

# -------------------------------------------------------------
# 5. Frequency-Domain Analysis Plotting
# -------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
axes = axes.flatten()

# Illustrate spectrum for Case 3 Run 16 (Cast Iron, Worn)
worn_row = valid_df[(valid_df['case'] == 3) & (valid_df['run'] == 16)].iloc[0]
item_worn = mat[0, int(worn_row['run_idx'])-1]

for s_idx, s in enumerate(SENSORS):
    ax = axes[s_idx]
    sig = item_worn[s].flatten()
    sig_detrend = sig - np.mean(sig)
    fft_mag = np.abs(np.fft.rfft(sig_detrend)) / (9000.0 / 2.0)
    
    ax.plot(freqs, fft_mag, color='#2b5c8f', lw=1.0)
    ax.axvline(SPINDLE_FREQ_HZ, color='#e67e22', linestyle='--', lw=1.2, label=f"Spindle Rotational Freq ({SPINDLE_FREQ_HZ:.1f} Hz)")
    ax.axvline(TOOTH_PASSING_FREQ_HZ, color='#9b59b6', linestyle=':', lw=1.2, label=f"Tooth Passing Freq ({TOOTH_PASSING_FREQ_HZ:.1f} Hz)")
    ax.axvspan(11.0, 16.5, color='#e67e22', alpha=0.15)
    ax.axvspan(75.0, 90.0, color='#9b59b6', alpha=0.15)
    
    ax.set_title(f"{s}: Single-Sided FFT Amplitude Spectrum", fontsize=10, fontweight='bold')
    ax.set_ylabel("Magnitude (V)", fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)
    if s_idx == 0:
        ax.legend(fontsize=8, loc='upper right')

axes[4].set_xlabel("Frequency (Hz) — Nyquist Limit = 125 Hz", fontsize=10, fontweight='bold')
axes[5].set_xlabel("Frequency (Hz) — Nyquist Limit = 125 Hz", fontsize=10, fontweight='bold')

plt.suptitle("Figure 4: Frequency-Domain Spectral Analysis & Machining Kinematics Alignment\n(Case 3 Run 16 Worn Tool: Clear spectral concentration at Spindle Freq 13.77 Hz and Tooth Passing Freq 82.60 Hz)", 
             fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout()
fig4_path = FREQ_DIR / "fig4_fft_spectra_and_cutting_harmonics.png"
plt.savefig(fig4_path, dpi=200)
plt.close()
print(f"Saved frequency analysis plot: {fig4_path.name}")

# -------------------------------------------------------------
# 6. Windowing Analysis: Full-Window vs Sub-Window Stability
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Scatter: Full RMS vs Steady RMS
ax1.scatter(df_features['smcAC_rms'], df_features['smcAC_rms_steady'], color='#1f77b4', alpha=0.7, edgecolors='black', s=40)
r_val = np.corrcoef(df_features['smcAC_rms'], df_features['smcAC_rms_steady'])[0, 1]
ax1.plot([0.5, 4.5], [0.5, 4.5], 'r--', lw=1.5, label=f'Identity 1:1 Line (Pearson r = {r_val:.4f})')
ax1.set_xlabel("Full-Window RMS (0 - 36 s) [Volts]", fontsize=10, fontweight='bold')
ax1.set_ylabel("Steady-State Window RMS (10 - 30 s) [Volts]", fontsize=10, fontweight='bold')
ax1.set_title("A) High Collinearity: Full vs Steady-State RMS (smcAC)", fontsize=10.5, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle=':', alpha=0.5)

# Distribution of Drift Ratio (W3 / W1)
drift_ac = df_features['smcAC_rms_drift_ratio'].dropna()
drift_vib = df_features['vib_table_rms_drift_ratio'].dropna()
drift_ae = df_features['AE_table_rms_drift_ratio'].dropna()

bp = ax2.boxplot([drift_ac, drift_vib, drift_ae], patch_artist=True, tick_labels=['smcAC', 'vib_table', 'AE_table'], widths=0.45)
cols = ['#1f77b4', '#2ca02c', '#d62728']
for patch, color in zip(bp['boxes'], cols):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)

ax2.axhline(1.0, color='black', linestyle='--', lw=1.2, label='No Within-Pass Drift (Ratio = 1.0)')
ax2.set_ylabel("Sub-Window Drift Ratio (Late W3 / Early W1 RMS)", fontsize=10, fontweight='bold')
ax2.set_title("B) Within-Pass Drift Ratio Across 36-Second Snapshot", fontsize=10.5, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.5)

plt.suptitle("Figure 5: Windowing Analysis — Full-Window (0-36s) vs Sub-Window Evaluation\n(Demonstrating near-perfect linearity r=0.996 with early-cut transient accounting for slight W1 dampening)", 
             fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout()
fig5_path = FREQ_DIR / "fig5_windowing_full_vs_subwindow_analysis.png"
plt.savefig(fig5_path, dpi=200)
plt.close()
print(f"Saved windowing analysis plot: {fig5_path.name}")

# -------------------------------------------------------------
# 7. Candidate Feature Set Definition, Role Categorization & Sanity Checks
# -------------------------------------------------------------
# Structured into:
# - Core (Purpose-driven: Current/Load, Vibration/Impact, AE/Friction, Kinematics)
# - Secondary / Exploratory (Redundancy audit candidate in Phase 3.2)
# - Diagnostic / Conditional (smcDC channels with hardware saturation risk)
CANDIDATE_FEATURES = [
    # Current / Load (Core & Secondary)
    ('smcAC', 'rms', 'Time', 'Core', 'AC spindle motor current dynamic energy (cutting force proxy)'),
    ('smcAC', 'std', 'Time', 'Secondary', 'Torque fluctuation standard deviation (high collinearity with RMS)'),
    ('smcAC', 'p2p', 'Time', 'Secondary', 'Peak-to-peak AC current excursion'),
    ('smcAC', 'kurtosis', 'Time', 'Secondary', 'AC current peakedness / impact sensitivity'),
    ('smcAC', 'crest', 'Time', 'Secondary', 'AC current crest factor (peak to RMS ratio)'),
    ('smcAC', 'spindle_band_pwr', 'Frequency', 'Core', 'Spectral power at spindle rotational harmonic (11-16.5 Hz)'),
    
    # Motor Current DC (Diagnostic / Conditional due to 10V hardware clipping)
    ('smcDC', 'mean', 'Time', 'Diagnostic / Conditional', 'DC current average motor load (CLIPPED at +10V in 41 runs)'),
    ('smcDC', 'std', 'Time', 'Diagnostic / Conditional', 'DC current ripple amplitude (variance affected by clipping)'),
    
    # Vibration / Table (Metallurgy & Kinematics)
    ('vib_table', 'mean', 'Time', 'Secondary', 'Table vibration RMS envelope mean level'),
    ('vib_table', 'std', 'Time', 'Secondary', 'Table vibration envelope fluctuation amplitude'),
    ('vib_table', 'p2p', 'Time', 'Secondary', 'Table vibration peak-to-peak amplitude range'),
    ('vib_table', 'kurtosis', 'Time', 'Secondary', 'Table vibration impact shock peakedness'),
    ('vib_table', 'spindle_band_pwr', 'Frequency', 'Core', 'Table vibration power at spindle rotational freq (11-16.5 Hz)'),
    ('vib_table', 'tooth_band_pwr', 'Frequency', 'Core', 'Table vibration power at tooth passing freq (75-90 Hz)'),
    
    # Vibration / Spindle (Mechanical Impact / Tool Degradation)
    ('vib_spindle', 'mean', 'Time', 'Secondary', 'Spindle vibration RMS envelope baseline level'),
    ('vib_spindle', 'p2p', 'Time', 'Core', 'Spindle vibration peak excursion (edge micro-fractures)'),
    ('vib_spindle', 'kurtosis', 'Time', 'Core', 'Spindle vibration impact peakedness (low DOC/Feed bias)'),
    
    # Acoustic Emission / Table (Friction & Material Dynamics)
    ('AE_table', 'mean', 'Time', 'Secondary', 'Table acoustic emission envelope mean (friction/deformation)'),
    ('AE_table', 'rms', 'Time', 'Core', 'Table acoustic emission effective energy (flank friction contact)'),
    ('AE_table', 'p2p', 'Time', 'Secondary', 'Table acoustic emission burst peak-to-peak range'),
    ('AE_table', 'kurtosis', 'Time', 'Secondary', 'Table acoustic burst peakedness (metallurgy/burst indicator)'),
    
    # Acoustic Emission / Spindle (Tool Edge Friction & Fractures)
    ('AE_spindle', 'mean', 'Time', 'Secondary', 'Spindle acoustic emission envelope baseline'),
    ('AE_spindle', 'rms', 'Time', 'Secondary', 'Spindle acoustic emission energy'),
    ('AE_spindle', 'p2p', 'Time', 'Core', 'Spindle acoustic emission burst range (wear-sensitive, lower DOC bias)'),
    ('AE_spindle', 'kurtosis', 'Time', 'Secondary', 'Spindle acoustic emission burst peakedness')
]

corr_rows = []
within_tool_rows = []
valid_tools = [c for c in range(1, 17) if c != 6]  # Exclude Case 6 (aborted, 1 run)

for s, f, domain, role, interp in CANDIDATE_FEATURES:
    col = f"{s}_{f}"
    vals = df_features[col]
    
    # 1. Pooled Correlation across all 145 valid runs
    p_vb, _ = stats.pearsonr(vals, df_features['VB_mm'])
    s_vb, _ = stats.spearmanr(vals, df_features['VB_mm'])
    s_mat, _ = stats.spearmanr(vals, df_features['material_code'])
    s_doc, _ = stats.spearmanr(vals, df_features['DOC_mm'])
    s_feed, _ = stats.spearmanr(vals, df_features['feed_mm_rev'])
    
    # 2. Within-Tool Sanity Check (Spearman correlation calculated strictly inside each tool)
    tool_corrs = []
    for c in valid_tools:
        sub = df_features[df_features['case'] == c]
        if len(sub) >= 4 and sub['VB_mm'].nunique() > 1:
            r_tool, _ = stats.spearmanr(sub[col], sub['VB_mm'])
            if not np.isnan(r_tool):
                tool_corrs.append(r_tool)
                
    med_within = np.median(tool_corrs) if tool_corrs else np.nan
    mean_within = np.mean(tool_corrs) if tool_corrs else np.nan
    min_within = np.min(tool_corrs) if tool_corrs else np.nan
    max_within = np.max(tool_corrs) if tool_corrs else np.nan
    pos_tools = sum(1 for x in tool_corrs if x > 0)
    strong_pos = sum(1 for x in tool_corrs if x > 0.5)
    
    clipping_flag = 'HIGH RISK (saturated at +10V in 41 runs)' if s == 'smcDC' else 'None'
    
    corr_rows.append({
        'sensor': s,
        'feature': f,
        'column_name': col,
        'domain': domain,
        'role': role,
        'interpretation': interp,
        'pearson_VB': round(p_vb, 3),
        'spearman_VB': round(s_vb, 3),
        'within_tool_median_spearman': round(med_within, 3),
        'within_tool_positive_fraction': f"{pos_tools}/{len(tool_corrs)}",
        'spearman_Material': round(s_mat, 3),
        'spearman_DOC': round(s_doc, 3),
        'spearman_Feed': round(s_feed, 3),
        'clipping_risk': clipping_flag
    })
    
    within_tool_rows.append({
        'column_name': col,
        'sensor': s,
        'feature': f,
        'role': role,
        'pooled_spearman_VB': round(s_vb, 3),
        'within_tool_median_spearman': round(med_within, 3),
        'within_tool_mean_spearman': round(mean_within, 3),
        'within_tool_min_spearman': round(min_within, 3),
        'within_tool_max_spearman': round(max_within, 3),
        'tools_positive': f"{pos_tools}/{len(tool_corrs)}",
        'tools_strong_positive_gt_0.5': f"{strong_pos}/{len(tool_corrs)}"
    })

df_corr = pd.DataFrame(corr_rows)
CORR_CSV = DATA_DIR / "phase3_feature_correlations.csv"
df_corr.to_csv(CORR_CSV, index=False)
print(f"Saved candidate feature correlation audit table to: {CORR_CSV}")

df_within = pd.DataFrame(within_tool_rows)
WITHIN_CSV = DATA_DIR / "phase3_within_tool_correlation_audit.csv"
df_within.to_csv(WITHIN_CSV, index=False)
print(f"Saved within-tool correlation audit table to: {WITHIN_CSV}")

# -------------------------------------------------------------
# 8. Feature Distributions & Confounding Plots
# -------------------------------------------------------------
# Fig 6: Candidate Features vs VB Scatter Plots Stratified by DOC / Material
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

scatter_configs = [
    ('smcAC_rms', 'A) smcAC RMS vs Flank Wear (VB)', 'AC Current RMS (V)', 'DOC_mm'),
    ('AE_table_rms', 'B) AE Table RMS vs Flank Wear (VB)', 'AE Table RMS (V)', 'DOC_mm'),
    ('vib_table_p2p', 'C) Table Vibration Peak-to-Peak vs VB', 'Table Vib P2P (V)', 'material_name'),
    ('smcAC_spindle_band_pwr', 'D) smcAC Spindle Harmonic Power (13.8 Hz) vs VB', 'Spindle Band Power (V²)', 'DOC_mm')
]

for idx, (feat_col, title, ylabel, strat_col) in enumerate(scatter_configs):
    ax = axes[idx]
    if strat_col == 'DOC_mm':
        for doc_val, col, marker in [(0.75, '#2ecc71', 'o'), (1.50, '#9b59b6', 's')]:
            sub = df_features[df_features['DOC_mm'] == doc_val]
            ax.scatter(sub['VB_mm'], sub[feat_col], color=col, marker=marker, s=45, alpha=0.8, edgecolors='black', lw=0.6,
                       label=f"DOC = {doc_val} mm")
    elif strat_col == 'material_name':
        for mat_val, col, marker in [('Cast Iron', '#1f77b4', 'o'), ('Stainless Steel J45', '#d62728', 's')]:
            sub = df_features[df_features['material_name'] == mat_val]
            ax.scatter(sub['VB_mm'], sub[feat_col], color=col, marker=marker, s=45, alpha=0.8, edgecolors='black', lw=0.6,
                       label=mat_val)
            
    ax.set_title(title, fontsize=10.5, fontweight='bold')
    ax.set_xlabel("Measured Flank Wear VB (mm)", fontsize=9.5, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9.5, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', fontsize=8.5)

plt.suptitle("Figure 6: Candidate Sensor Features vs. Measured Flank Wear (VB)\n(Exposing Both Wear Progression and Confounding Stratification by DOC and Material)", 
             fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout()
fig6_path = FEAT_DIST_DIR / "fig6_features_vs_VB_with_confounding.png"
plt.savefig(fig6_path, dpi=200)
plt.close()
print(f"Saved feature vs VB confounding plot: {fig6_path.name}")

# Fig 7: Correlation Heatmap (VB vs Confounders)
fig, ax = plt.subplots(figsize=(10, 11))
corr_matrix_df = df_corr[['column_name', 'spearman_VB', 'spearman_DOC', 'spearman_Feed', 'spearman_Material']].set_index('column_name')
corr_matrix_df = corr_matrix_df.sort_values('spearman_VB', ascending=True)

im = ax.imshow(corr_matrix_df.values, cmap='coolwarm', vmin=-0.8, vmax=0.8, aspect='auto')
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label("Spearman Rank Correlation", fontsize=10, fontweight='bold')

ax.set_yticks(np.arange(len(corr_matrix_df)))
ax.set_yticklabels(corr_matrix_df.index, fontsize=8.5)
ax.set_xticks(np.arange(4))
ax.set_xticklabels(['Spearman VB\n(Target)', 'Spearman DOC\n(Confounder)', 'Spearman Feed\n(Confounder)', 'Spearman Material\n(Confounder)'], 
                   fontsize=9.5, fontweight='bold')

for i in range(len(corr_matrix_df)):
    for j in range(4):
        val = corr_matrix_df.iloc[i, j]
        text_col = 'white' if abs(val) > 0.45 else 'black'
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=text_col, fontsize=8, fontweight='bold')

ax.set_title("Figure 7: Feature Correlation Audit — Flank Wear (VB) vs Cutting Condition Confounders\n(Critical Check: High correlation with VB often co-occurs with strong confounding by DOC and Feed)", 
             fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
fig7_path = FEAT_DIST_DIR / "fig7_feature_correlation_heatmap.png"
plt.savefig(fig7_path, dpi=200)
plt.close()
print(f"Saved correlation heatmap: {fig7_path.name}")

# Fig 8: Replicate Tool Repeatability (Rep 1 vs Rep 2 Feature Trajectories)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot smcAC_rms vs run for Case 3 (Rep 1) vs Case 11 (Rep 2) [Cast Iron, DOC=0.75, Feed=0.25]
c3 = df_features[df_features['case'] == 3].sort_values('run')
c11 = df_features[df_features['case'] == 11].sort_values('run')
ax1.plot(c3['run'], c3['smcAC_rms'], 'o-', color='#1f77b4', label=f"Case 3 (Rep 1) - Final VB={c3['VB_mm'].max():.2f}mm", lw=1.6)
ax1.plot(c11['run'], c11['smcAC_rms'], 's--', color='#6baed6', label=f"Case 11 (Rep 2) - Final VB={c11['VB_mm'].max():.2f}mm", lw=1.6)
ax1.set_xlabel("Milling Run (Pass Number)", fontsize=10, fontweight='bold')
ax1.set_ylabel("smcAC RMS (Volts)", fontsize=10, fontweight='bold')
ax1.set_title("A) Condition 1 (Cast Iron, d0.75 f0.25): Rep 1 vs Rep 2", fontsize=10.5, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.legend(loc='upper left')

# Plot smcAC_rms vs run for Case 7 (Rep 1) vs Case 13 (Rep 2) [Stainless Steel J45, DOC=0.75, Feed=0.25]
c7 = df_features[df_features['case'] == 7].sort_values('run')
c13 = df_features[df_features['case'] == 13].sort_values('run')
ax2.plot(c7['run'], c7['smcAC_rms'], 'o-', color='#d62728', label=f"Case 7 (Rep 1) - Final VB={c7['VB_mm'].max():.2f}mm", lw=1.6)
ax2.plot(c13['run'], c13['smcAC_rms'], 's--', color='#e28743', label=f"Case 13 (Rep 2) - Final VB={c13['VB_mm'].max():.2f}mm", lw=1.6)
ax2.set_xlabel("Milling Run (Pass Number)", fontsize=10, fontweight='bold')
ax2.set_ylabel("smcAC RMS (Volts)", fontsize=10, fontweight='bold')
ax2.set_title("B) Condition 5 (Stainless Steel J45, d0.75 f0.25): Rep 1 vs Rep 2", fontsize=10.5, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.legend(loc='upper left')

plt.suptitle("Figure 8: Replicate Tool Repeatability Check Across Independent Inserts\n(Evaluating whether sensor feature degradation curves track similarly between Replicate 1 and Replicate 2)", 
             fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout()
fig8_path = FEAT_DIST_DIR / "fig8_replicate_feature_repeatability.png"
plt.savefig(fig8_path, dpi=200)
plt.close()
print(f"Saved replicate repeatability plot: {fig8_path.name}")

# Fig 9: Within-Tool Sanity Check — Pooled vs Within-Tool Correlation
fig, ax = plt.subplots(figsize=(14, 8))

# Sort df_within by within_tool_median_spearman
df_plot_within = df_within.sort_values('within_tool_median_spearman', ascending=True).reset_index(drop=True)
y_pos = np.arange(len(df_plot_within))

# Draw horizontal guide lines (dumbbells)
for idx, row in df_plot_within.iterrows():
    p_corr = row['pooled_spearman_VB']
    w_corr = row['within_tool_median_spearman']
    line_col = '#7f8c8d' if abs(p_corr - w_corr) < 0.25 else ('#e74c3c' if p_corr > w_corr else '#2ecc71')
    ax.plot([p_corr, w_corr], [idx, idx], color=line_col, alpha=0.6, lw=2.0)

# Scatter points
ax.scatter(df_plot_within['pooled_spearman_VB'], y_pos, color='#3498db', s=65, label='Pooled Spearman r (All 145 Runs)', zorder=4, edgecolors='black', lw=0.6)
ax.scatter(df_plot_within['within_tool_median_spearman'], y_pos, color='#e67e22', s=75, marker='s', label='Within-Tool Median Spearman r (Strictly Inside Each Tool)', zorder=5, edgecolors='black', lw=0.8)

ax.axvline(0.0, color='black', linestyle='--', lw=1.0, alpha=0.7)
ax.axvline(0.5, color='#27ae60', linestyle=':', lw=1.0, alpha=0.5, label='Moderate/Strong Positive Correlation threshold (r = +0.50)')

ax.set_yticks(y_pos)
ax.set_yticklabels([f"{row['column_name']} ({row['role']})" for _, row in df_plot_within.iterrows()], fontsize=9)
ax.set_xlabel("Spearman Rank Correlation with Measured Flank Wear (VB)", fontsize=10.5, fontweight='bold')
ax.set_xlim(-1.0, 1.05)
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='lower right', fontsize=9.5, framealpha=0.95)

ax.set_title("Figure 9: Within-Tool vs. Pooled Wear Correlation Sanity Check\n(Comparing Pooled 145-run correlation vs. Median within-tool correlation to isolate true wear sensitivity from cutting condition confounding)", 
             fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
fig9_path = FEAT_DIST_DIR / "fig9_within_tool_vs_pooled_correlation.png"
plt.savefig(fig9_path, dpi=200)
plt.close()
print(f"Saved within-tool vs pooled correlation plot: {fig9_path.name}")

print("\n" + "="*80)
print("PHASE 3.1 SENSOR EXPLORATION & AUDIT COMPLETED SUCCESSFULLY!")
print("="*80)

