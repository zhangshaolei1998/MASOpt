"""Build the equal-size workflow and prompt-preference figures.

The workflow scores and ALFWorld preferences are the author's 2026-09-25
revision; other preference rows are unchanged from the manuscript table.
The reported HotpotQA gain differs from subtraction of the rounded scores
by 0.0001 pp. Preserve the supplied values and display gains to two decimals.
Run this file with Python and Matplotlib; the CSV files are the data source.
"""
from pathlib import Path
import csv
from decimal import Decimal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
BLUE, GRAY, RED = '#3C70B6', '#8A9BB2', '#E77D68'
TIE = '#BCC1CA'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.linewidth': .7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def read_csv(name):
    with (ROOT / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def canvas():
    # Identical physical dimensions and plot heights keep the two figures aligned.
    fig = plt.figure(figsize=(4.0, 2.75))
    ax = fig.add_axes([.235, .21, .725, .59])
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=3, width=.65)
    ax.set_axisbelow(True)
    return fig, ax


def workflow():
    rows = read_csv('architecture_adaptation.csv')
    for row in rows:
        difference = Decimal(row['full']) - Decimal(row['initial'])
        assert abs(difference - Decimal(row['reported_gain_pp'])) <= Decimal('.0001')
    initial = np.array([float(row['initial']) for row in rows])
    full = np.array([float(row['full']) for row in rows])
    x = np.arange(len(rows))
    fig, ax = canvas()
    width, offset = .35, .24
    ax.bar(x-offset, initial, width, color=GRAY, label='Initial', zorder=3)
    ax.bar(x+offset, full, width, color=BLUE, label='MASOpt (Full)', zorder=3)
    for i, row in enumerate(rows):
        ax.text(x[i]-offset, initial[i]+1.4, f'{initial[i]:.2f}',
                ha='center', va='bottom', fontsize=9.0, color='#343C48')
        ax.text(x[i]+offset, full[i]+3.6,
                f"+{float(row['reported_gain_pp']):.2f}",
                ha='center', va='bottom', fontsize=10.2, fontweight='bold')
    ax.set_xticks(x, [row['dataset'] for row in rows])
    ax.set_xlim(-.65, len(rows)-.35)
    ax.set_ylim(0, 88)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_ylabel('Task score (%)')
    ax.grid(axis='y', color='#DFE3E9', linestyle='--', linewidth=.55)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.10), ncol=2,
              frameon=False, fontsize=9.6, handlelength=1.2,
              handletextpad=.45, columnspacing=1.3, borderaxespad=0)
    fig.savefig(ROOT/'architecture_adaptation.pdf')
    plt.close(fig)


def preference():
    rows = read_csv('prompt_preference.csv')
    for row in rows:
        assert sum(Decimal(row[k]) for k in ['masopt_win', 'tie', 'maspob_win']) == 100
    fig, ax = canvas()
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for key, color, label in [('masopt_win', BLUE, 'MASOpt wins'),
                              ('tie', TIE, 'Tie'),
                              ('maspob_win', RED, 'MASPOB* wins')]:
        values = np.array([float(row[key]) for row in rows])
        ax.barh(y, values, left=left, height=.69, color=color, label=label, zorder=3)
        for i, value in enumerate(values):
            if value >= 8:
                ax.text(left[i]+value/2, y[i], f'{value:.2f}',
                        ha='center', va='center', fontsize=8.6 if value < 16 else 10,
                        fontweight='bold',
                        color='white' if key == 'masopt_win' else '#20242A')
            elif value > 0:
                # Place the 2.29% minority above its narrow segment, with a leader.
                ax.annotate(f'{value:.2f}', (left[i]+value/2, y[i]-.35),
                            xytext=(95, y[i]-.57), textcoords='data',
                            ha='right', va='bottom', fontsize=8.8,
                            arrowprops={'arrowstyle':'-', 'lw':.6, 'color':'#646B73'})
        left += values
    ax.set_yticks(y, [row['dataset'] for row in rows])
    ax.set_ylim(len(rows)-.5, -.85)
    ax.set_xlim(0, 100)
    ax.set_xticks([0,20,40,60,80,100])
    ax.set_xlabel('Pairwise prompt preference (%)', labelpad=5)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=5)
    ax.grid(axis='x', color='#DFE3E9', linestyle='--', linewidth=.55)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.10), ncol=3,
              frameon=False, fontsize=9.0, handlelength=1.0,
              handletextpad=.35, columnspacing=.7, borderaxespad=0)
    fig.savefig(ROOT/'prompt_quality.pdf')
    plt.close(fig)


if __name__ == '__main__':
    workflow()
    preference()
    print('Rebuilt the two equal-size experiment figures from the CSV sources.')
