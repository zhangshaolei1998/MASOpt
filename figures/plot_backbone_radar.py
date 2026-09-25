"""Rebuild the compact six-panel radar from the archived paired-score CSV.

Run with Python, NumPy, and Matplotlib. backbone_radar.csv preserves the
Initial/Full scores previously reported in the manuscript's detailed table.
All panels retain the same zoomed radius, 99--102, with Initial = 100.
The manuscript caption explains the scale; the graphic contains no gain,
mean-gain, or radial-tick annotations.
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
with (ROOT / 'backbone_radar.csv').open(encoding='utf-8', newline='') as stream:
    rows = list(csv.DictReader(stream))
names = ['GPT-5.3-codex', 'GPT-5.4-mini', 'Gemini-3.6-flash',
         'Grok-4.5', 'Qwen3-4B', 'Qwen3.5-9B']
datasets = ['HumanEval', 'MBPP', 'GSM8K', 'MATH', 'HotpotQA', 'DROP']
scores = {(r['backbone'], r['dataset']): (float(r['initial']), float(r['full']))
          for r in rows}
assert len(scores) == len(rows) == 36

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(2, 3, figsize=(9.0, 3.95),
                         subplot_kw={'projection': 'polar'})
fig.subplots_adjust(left=.08, right=.92, bottom=.10, top=.88,
                    wspace=.62, hspace=.64)
blue, red = '#5C88C9', '#D24C48'
angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
closed = np.r_[angles, angles[0]]
for ax, name in zip(axes.flat, names):
    paired = np.array([scores[name, dataset] for dataset in datasets])
    initial, full = paired[:, 0], paired[:, 1]
    assert np.all(initial > 0) and np.all(paired <= 100)
    relative = 100*full/initial
    assert np.all((relative >= 99) & (relative <= 102)), name
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_ylim(99, 102)
    ax.set_yticks([99, 100, 101, 102])
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(datasets, fontsize=9.5)
    ax.tick_params(axis='x', pad=4)
    ax.grid(color='#D7DDE5', linewidth=.55)
    ax.spines['polar'].set_color('#BCC6D3')
    ax.spines['polar'].set_linewidth(.6)
    base = np.full(7, 100.)
    curve = np.r_[relative, relative[0]]
    ax.fill(closed, base, color=blue, alpha=.12, zorder=2)
    ax.fill(closed, curve, color=red, alpha=.09, zorder=3)
    ax.plot(closed, base, color=blue, linewidth=1.4, linestyle='--',
            marker='o', markersize=2.4, zorder=5)
    ax.plot(closed, curve, color=red, linewidth=1.5,
            marker='o', markersize=2.8, zorder=6)
    ax.text(.5, -.26, name, transform=ax.transAxes, ha='center', va='top',
            fontsize=10, fontweight='bold', color='#222D3C')
fig.legend([Line2D([0], [0], color=blue, lw=1.5, ls='--', marker='o', ms=3),
            Line2D([0], [0], color=red, lw=1.5, marker='o', ms=3)],
           ['Initial', 'MASOpt (Full)'], loc='upper center',
           bbox_to_anchor=(.5, 1.025), ncol=2, frameon=False, fontsize=11,
           handlelength=2.1, columnspacing=1.8)
out = ROOT / 'backbone_radar.pdf'
fig.savefig(out, bbox_inches='tight', pad_inches=.06)
plt.close(fig)
print(f'Wrote {out}; all 36 paired scores are unchanged.')
