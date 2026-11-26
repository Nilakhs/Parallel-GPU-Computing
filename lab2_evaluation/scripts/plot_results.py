#!/usr/bin/env python3
"""
Performance Visualization & Scientific Plotting Script
Project: CUDA Warp Divergence Analysis | Parallel GPU Computing (Lab Evaluation 2)
Verified Hardware: NVIDIA RTX 4500 Ada Generation (SM 8.9)
CUDA Toolkit: 13.4 | Host Compiler: MSVC 19.51 | Windows 11 x64

This script generates high-resolution, scientifically accurate figures from timing_data.csv.
All formulas strictly adhere to standard high-performance computing definitions:
  - Baseline: T_baseline = T_divergent = 0.3738 ms
  - Execution-Time Ratio: R = T_variant / T_baseline  (R < 1.0 is faster)
  - Speedup:              S = T_baseline / T_variant  (S > 1.0 is faster)
  - Pct Change:           ((T_baseline - T_variant) / T_baseline) * 100%
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Directory resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_DIR = os.path.join(BASE_DIR, 'graphs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(GRAPHS_DIR, exist_ok=True)

# Publication-grade styling configuration
plt.rcParams.update({
    'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial'],
    'font.size': 10,
    'axes.edgecolor': '#2C3E50',
    'axes.linewidth': 1.2,
    'grid.color': '#BDC3C7',
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'figure.autolayout': True,
    'savefig.dpi': 300
})

# Load benchmark data
csv_path = os.path.join(RESULTS_DIR, 'timing_data.csv')
df = pd.read_csv(csv_path)

# Consistent academic color palette
COLOR_DIVERGENT   = '#E74C3C'  # Coral Red (Divergent Baseline)
COLOR_ALIGNED     = '#2ECC71'  # Emerald Green (Warp-Aligned)
COLOR_DATADEP     = '#3498DB'  # Dodger Blue (Data-Dependent)
COLOR_BRANCHFREE  = '#9B59B6'  # Amethyst Purple (Branch-Free)
PALETTE = [COLOR_DIVERGENT, COLOR_ALIGNED, COLOR_DATADEP, COLOR_BRANCHFREE]

# Baseline constants
T_BASE = float(df[df['kernel_type'] == 'divergent']['avg_time_ms'].iloc[0])

# ==============================================================================
# Graph 1: Kernel Execution Time Comparison
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
bars = ax.bar(df['kernel_name'], df['avg_time_ms'], color=PALETTE, width=0.52,
              edgecolor='#1A252F', linewidth=1.2)

ax.axhline(T_BASE, color='#C0392B', linestyle=':', linewidth=1.4, alpha=0.8,
           label=f'Baseline (Divergent): {T_BASE:.4f} ms')

ax.set_title('CUDA Kernel Execution Latency (Average over 20 Repetitions)\n'
             'NVIDIA RTX 4500 Ada Generation (SM 8.9) · N = 16,777,216 Elements (64 MB FP32)',
             fontsize=11.5, fontweight='bold', pad=14)
ax.set_ylabel('Execution Time (ms) [Lower is Better]', fontsize=11, fontweight='bold')
ax.set_ylim(0, 0.48)
ax.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    pct_diff = ((T_BASE - yval) / T_BASE) * 100.0
    label_text = f'{yval:.4f} ms'
    if abs(pct_diff) > 0.01:
        sign = '+' if pct_diff > 0 else ''
        label_text += f'\n({sign}{pct_diff:.2f}%)'
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008,
            label_text, ha='center', va='bottom', fontsize=9.5, fontweight='bold')

ax.set_xticks(range(len(df['kernel_name'])))
ax.set_xticklabels(df['kernel_name'], rotation=12, ha='right', fontsize=9.5)
ax.legend(loc='upper right', fontsize=9)
plt.savefig(os.path.join(GRAPHS_DIR, '01_execution_time_comparison.png'), bbox_inches='tight')
plt.close()

# ==============================================================================
# Graph 2: SIMT Active Mask State (Theoretical 32-Lane Model)
# ==============================================================================
fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)
warp_lanes = np.arange(32)

aligned_mask    = [1 for _ in warp_lanes]
divergent_mask  = [1 if i % 2 == 0 else 0 for i in warp_lanes]
branchfree_mask = [1 for _ in warp_lanes]

y_positions = [2, 1, 0]
labels = [
    'Warp-Aligned Branch\n((i / 32) % 2 == 0) [100% Active]',
    'Divergent Branch\n(i % 2 == 0) [50% Active, 50% Idle]',
    'Branch-Free Predication\n(Unconditional Compute) [100% Active]'
]

for y_idx, (y_pos, pattern) in enumerate(zip(y_positions, [aligned_mask, divergent_mask, branchfree_mask])):
    for lane in range(32):
        if y_pos == 1:
            color = '#2ECC71' if pattern[lane] == 1 else '#E74C3C'
        elif y_pos == 2:
            color = '#3498DB'
        else:
            color = '#9B59B6'
        rect = patches.Rectangle((lane * 0.9, y_pos - 0.25), 0.78, 0.5,
                                 facecolor=color, edgecolor='#2C3E50', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(lane * 0.9 + 0.39, y_pos, str(lane), ha='center', va='center',
                fontsize=6.5, color='white', fontweight='bold')

ax.set_xlim(-0.5, 32 * 0.9 + 0.5)
ax.set_ylim(-0.8, 2.8)
ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=9.5, fontweight='bold')
ax.set_xlabel('Warp Lane Index (0 to 31 within Warp)', fontsize=10.5, fontweight='bold')
ax.set_title('SIMT Hardware Execution: Theoretical Active Mask & Lane Utilization\n'
             '[Illustrative Model: Shows Active vs Masked Execution States]',
             fontsize=11.5, fontweight='bold', pad=14)
ax.grid(False)

legend_patches = [
    patches.Patch(facecolor='#2ECC71', edgecolor='#2C3E50', label='Active Lane: Path A (Step 1)'),
    patches.Patch(facecolor='#E74C3C', edgecolor='#2C3E50', label='Masked Lane: Serialized (Step 2)'),
    patches.Patch(facecolor='#3498DB', edgecolor='#2C3E50', label='Uniform Warp: All 32 Lanes in Unison'),
    patches.Patch(facecolor='#9B59B6', edgecolor='#2C3E50', label='Predicated Lane: Both Paths Executed')
]
ax.legend(handles=legend_patches, loc='lower right', bbox_to_anchor=(1.0, -0.28), ncol=2, fontsize=8.5)
plt.savefig(os.path.join(GRAPHS_DIR, '02_branch_efficiency_and_active_mask.png'), bbox_inches='tight')
plt.close()

# ==============================================================================
# Graph 3: Rigorous Speedup and Execution-Time Ratio Comparison
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)

# Left Panel: Speedup S = T_base / T_variant (S > 1 means faster)
speedups = df['speedup_vs_baseline']
bars1 = ax1.bar(df['kernel_name'], speedups, color=PALETTE, width=0.5, edgecolor='#1A252F', linewidth=1.2)
ax1.axhline(1.0, color='#C0392B', linestyle='--', linewidth=1.5, label='Baseline (1.000x)')
ax1.set_ylabel('Speedup S = T_baseline / T_variant\n[Values > 1.0 indicate FASTER]', fontsize=10, fontweight='bold')
ax1.set_title('Speedup Relative to Divergent Baseline', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 1.25)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.set_xticks(range(len(df['kernel_name'])))
ax1.set_xticklabels(df['kernel_name'], rotation=18, ha='right', fontsize=8.5)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02,
             f'{yval:.4f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.legend(loc='upper right', fontsize=8.5)

# Right Panel: Execution-Time Ratio R = T_variant / T_base (R < 1 means faster)
ratios = df['execution_time_ratio']
bars2 = ax2.bar(df['kernel_name'], ratios, color=PALETTE, width=0.5, edgecolor='#1A252F', linewidth=1.2)
ax2.axhline(1.0, color='#C0392B', linestyle='--', linewidth=1.5, label='Baseline (1.000x)')
ax2.set_ylabel('Execution-Time Ratio R = T_variant / T_baseline\n[Values < 1.0 indicate FASTER]', fontsize=10, fontweight='bold')
ax2.set_title('Execution-Time Ratio vs Baseline', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 1.25)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.set_xticks(range(len(df['kernel_name'])))
ax2.set_xticklabels(df['kernel_name'], rotation=18, ha='right', fontsize=8.5)

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02,
             f'{yval:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.legend(loc='upper right', fontsize=8.5)

fig.suptitle('Performance Metrics: Speedup (S) vs Execution-Time Ratio (R)\n'
             'Strict distinction between throughput acceleration and latency proportion',
             fontsize=12, fontweight='bold')
plt.savefig(os.path.join(GRAPHS_DIR, '03_relative_speedup_ratios.png'), bbox_inches='tight')
plt.close()

# ==============================================================================
# Graph 4: Throughput (Million Elements Processed Per Second)
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
throughputs = df['throughput_m_elems_sec']
bars = ax.bar(df['kernel_name'], throughputs, color=PALETTE, width=0.52, edgecolor='#1A252F', linewidth=1.2)

ax.set_title('Computational Processing Throughput (Mega-Elements / sec)\n'
             'Array Size: 16.78M FP32 Elements · NVIDIA RTX 4500 Ada Generation',
             fontsize=11.5, fontweight='bold', pad=14)
ax.set_ylabel('Throughput (Million Elements / s) [Higher is Better]', fontsize=10.5, fontweight='bold')
ax.set_ylim(0, 52000)
ax.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 800,
            f'{yval:,.1f} M/s', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

ax.set_xticks(range(len(df['kernel_name'])))
ax.set_xticklabels(df['kernel_name'], rotation=12, ha='right', fontsize=9.5)
plt.savefig(os.path.join(GRAPHS_DIR, '04_throughput_and_giga_elements.png'), bbox_inches='tight')
plt.close()

# ==============================================================================
# Graph 5: Instruction Inflation vs Branch Serialization Trade-off
# ==============================================================================
categories = ['Divergent\n(i % 2)', 'Aligned\n((i/32) % 2)', 'Data-Dep\n(threshold)', 'Branch-Free\n(both paths)']
arithmetic_ops = [8, 8, 8, 16] # 8 FLOPs per thread vs 16 FLOPs for unconditional workA & workB
active_mask_pct = [50, 100, 50, 100]

x = np.arange(len(categories))
width = 0.35

fig, ax1 = plt.subplots(figsize=(9, 5.2), dpi=300)
ax2 = ax1.twinx()

rects1 = ax1.bar(x - width/2, arithmetic_ops, width, label='Theoretical Source FLOPs / Thread', color='#E67E22', edgecolor='#1A252F')
rects2 = ax2.bar(x + width/2, active_mask_pct, width, label='Theoretical Active Lanes (%)', color='#1ABC9C', edgecolor='#1A252F')

ax1.set_ylabel('Theoretical Arithmetic Ops / Thread (Source-Level)', fontsize=10, fontweight='bold', color='#D35400')
ax2.set_ylabel('Theoretical Active Warp Lanes (%)', fontsize=10, fontweight='bold', color='#16A085')
ax1.set_title('Instruction Inflation vs Branch Masking Trade-off\n'
             '[Theoretical Model: Based on Loop Iterations in workA and workB]',
             fontsize=11.5, fontweight='bold', pad=14)
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=9.5, fontweight='bold')
ax1.set_ylim(0, 20)
ax2.set_ylim(0, 120)

for r in rects1:
    ax1.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4, f'{int(r.get_height())}', ha='center', fontsize=9, fontweight='bold')
for r in rects2:
    ax2.text(r.get_x() + r.get_width()/2, r.get_height() + 2, f'{int(r.get_height())}%', ha='center', fontsize=9, fontweight='bold')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

plt.savefig(os.path.join(GRAPHS_DIR, '05_divergence_penalty_analysis.png'), bbox_inches='tight')
plt.close()

# ==============================================================================
# Graph 6: Executive Performance Dashboard
# ==============================================================================
fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.5), dpi=300)

# (0,0): Execution Time
axs[0, 0].bar(df['kernel_name'], df['avg_time_ms'], color=PALETTE, edgecolor='#1A252F')
axs[0, 0].set_title('Execution Latency (ms) [Lower is Better]', fontsize=11, fontweight='bold')
axs[0, 0].set_ylabel('Latency (ms)', fontsize=10)
axs[0, 0].set_ylim(0, 0.48)
axs[0, 0].tick_params(axis='x', rotation=18)
for bar in axs[0, 0].patches:
    axs[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                   f'{bar.get_height():.4f} ms', ha='center', fontsize=8.5, fontweight='bold')

# (0,1): Throughput
axs[0, 1].bar(df['kernel_name'], throughputs, color=PALETTE, edgecolor='#1A252F')
axs[0, 1].set_title('Effective Throughput (Million Elements / sec)', fontsize=11, fontweight='bold')
axs[0, 1].set_ylabel('Throughput (M elem/s)', fontsize=10)
axs[0, 1].set_ylim(0, 52000)
axs[0, 1].tick_params(axis='x', rotation=18)
for bar in axs[0, 1].patches:
    axs[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 800,
                   f'{bar.get_height():,.0f} M/s', ha='center', fontsize=8.5, fontweight='bold')

# (1,0): Speedup
bars_s = axs[1, 0].bar(df['kernel_name'], speedups, color=PALETTE, edgecolor='#1A252F')
axs[1, 0].axhline(1.0, color='red', linestyle='--', linewidth=1.2, label='1.0x Baseline')
axs[1, 0].set_title('Speedup S = T_base / T_variant [> 1.0 is Faster]', fontsize=11, fontweight='bold')
axs[1, 0].set_ylabel('Speedup Factor', fontsize=10)
axs[1, 0].set_ylim(0, 1.25)
axs[1, 0].tick_params(axis='x', rotation=18)
for bar in bars_s:
    axs[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{bar.get_height():.3f}x', ha='center', fontsize=8.5, fontweight='bold')
axs[1, 0].legend(loc='upper right', fontsize=8.5)

# (1,1): Architectural Specs Table
axs[1, 1].axis('off')
table_data = [
    ['Target GPU Model', 'NVIDIA RTX 4500 Ada Generation'],
    ['Compute Architecture', 'Ada Lovelace (SM 8.9)'],
    ['Array Elements (N)', '16,777,216 (2^24 FP32)'],
    ['Grid / Block Config', '65,536 Blocks / 256 Threads/Block'],
    ['Repetitions Timed', '20 Runs (CUDA Events)'],
    ['Baseline (Divergent)', f"{df[df['kernel_type']=='divergent']['avg_time_ms'].iloc[0]:.4f} ms (1.000x)"],
    ['Warp-Aligned', f"{df[df['kernel_type']=='aligned']['avg_time_ms'].iloc[0]:.4f} ms (0.945x speedup)"],
    ['Data-Dependent', f"{df[df['kernel_type']=='data_dependent']['avg_time_ms'].iloc[0]:.4f} ms (1.025x speedup)"],
    ['Branch-Free Predicated', f"{df[df['kernel_type']=='branch_free']['avg_time_ms'].iloc[0]:.4f} ms (0.946x speedup)"]
]
tbl = axs[1, 1].table(cellText=table_data, colLabels=['Benchmark Parameter', 'Verified Metric / Value'],
                      loc='center', cellLoc='left')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.05, 1.35)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#34495E')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#F8F9F9' if row % 2 == 0 else '#FFFFFF')

fig.suptitle('PGC Lab 2: Comprehensive CUDA Warp Divergence Evaluation Dashboard',
             fontsize=13.5, fontweight='bold', y=0.98)
plt.savefig(os.path.join(GRAPHS_DIR, '06_comprehensive_evaluation_dashboard.png'), bbox_inches='tight')
plt.close()

# ==============================================================================
# Graph 7: Warp Lane Execution Timeline (Theoretical Serialization Model)
# ==============================================================================
fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=300)

ax.barh(y=1, width=1.0, left=0.0, height=0.4, color='#E74C3C', edgecolor='black',
        label='Phase 1: Even Lanes Active (workA), Odd Lanes Masked (Idle)')
ax.barh(y=1, width=1.0, left=1.0, height=0.4, color='#E67E22', edgecolor='black',
        label='Phase 2: Odd Lanes Active (workB), Even Lanes Masked (Idle)')

ax.barh(y=0, width=1.0, left=0.0, height=0.4, color='#2ECC71', edgecolor='black',
        label='Unified Phase: All 32 Lanes Active in Unison (workA or workB)')

ax.set_yticks([0, 1])
ax.set_yticklabels(['Warp-Aligned Warp\n(Single Unified Phase)',
                    'Divergent Warp\n(Serialized Execution)'],
                   fontsize=10, fontweight='bold')
ax.set_xlabel('Instruction Execution Phase / Clock Time Unit', fontsize=10.5, fontweight='bold')
ax.set_title('SIMT Hardware Execution Model: Warp Serialization vs Uniform Execution\n'
             '[Theoretical Model: Illustrates Why Divergence Serializes Execution Paths]',
             fontsize=11.5, fontweight='bold', pad=14)
ax.set_xlim(0, 2.3)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
ax.set_xticklabels(['0', 'T / 2', 'T (Aligned Completes)', '1.5 T', '2T (Divergent Completes)'])
ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.legend(loc='upper right', fontsize=8.5)

plt.savefig(os.path.join(GRAPHS_DIR, '07_warp_lane_execution_timeline.png'), bbox_inches='tight')
plt.close()

print("[SUCCESS] All 7 high-resolution evaluation graphs regenerated successfully.")
