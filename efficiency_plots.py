"""
Reproduces the base ARJ8S paper's Figs. 1-3 (classical efficiency index,
computational efficiency index, computational performance), extended to
I-ARJ8S.

Journal-standard formatting (Elsevier/Applied Numerical Mathematics author
guidelines): line-art figures are saved as vector PDF, not raster, so they
scale to any print size without pixelation; no title is baked into the
image (the LaTeX \\caption already provides this -- a redundant in-figure
title is a common desk-reject-level formatting complaint); font sizes are
set for readability at final print width (~8-9cm after LaTeX scaling, not
the much larger on-screen preview size); every line uses both a distinct
color AND a distinct dash/marker style so the figure remains readable if
printed or photocopied in black-and-white.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'legend.fontsize': 11.5,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.linewidth': 0.9,
    'lines.linewidth': 1.8,
    'pdf.fonttype': 42,   # embed fonts as scalable outlines, not bitmaps
    'ps.fonttype': 42,
})

# Distinct (color, dash pattern, marker) per method so the figure survives
# both color and grayscale reproduction.
STYLES = {
    'Newton':              dict(color='#1b1b1b', ls='-',  marker=None,  markevery=0),
    'ARJ8S':                dict(color='#0072B2', ls='-',  marker='o',   markevery=0.12),
    'CTT8S':                dict(color='#D55E00', ls='--', marker='s',   markevery=0.12),
    'ZO8S':                 dict(color='#009E73', ls='-.', marker='^',   markevery=0.12),
    'EH9S':                 dict(color='#CC79A7', ls=':',  marker='D',   markevery=0.12),
    'I-ARJ8S (best case)':  dict(color='#E69F00', ls='-',  marker='*',   markevery=0.12),
}


def plot_series(ax, n, series_dict, ylabel):
    for name, y in series_dict.items():
        st = STYLES[name]
        ax.plot(n, y, label=name, color=st['color'], linestyle=st['ls'],
                 marker=st['marker'], markevery=st['markevery'] or None,
                 markersize=6, markerfacecolor='none', markeredgewidth=1.3)
    ax.set_xscale('log')
    ax.set_xlabel(r'$n$')
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc='best', ncol=2 if len(series_dict) > 4 else 1)
    ax.grid(True, which='both', alpha=0.25, linewidth=0.6)
    ax.tick_params(direction='in', top=True, right=True)


n = np.arange(10, 101)

methods_I = {
    'Newton': (2, n**2 + n),
    'ARJ8S': (8, n**2 + 4 * n),
    'CTT8S': (8, n**2 + 3 * n),
    'ZO8S': (8, 2 * n**2 + 3 * n),
    'EH9S': (9, n**2 + 10 * n),
    'I-ARJ8S (best case)': (8, n**2 + 5 * n),
}
series = {name: p ** (1.0 / d) for name, (p, d) in methods_I.items()}

fig, ax = plt.subplots(figsize=(5.4, 4.1))
plot_series(ax, n, series, r'$I$')
fig.tight_layout(pad=0.4)
fig.savefig('efficiency_index.pdf')
plt.close(fig)
print("Saved efficiency_index.pdf")

methods_CI = {
    'Newton': (2, n**2 + n, (1 / 3) * (n**3 - n) + n**2),
    'ARJ8S': (8, n**2 + 4 * n, (1 / 3) * (n**3 - n) + 4 * n**2 + 5 * n),
    'CTT8S': (8, n**2 + 3 * n, (1 / 3) * (n**3 - n) + 3 * n**2 + 8 * n),
    'ZO8S': (8, 2 * n**2 + 3 * n, (2 / 3) * (n**3 - n) + 3 * n**2 + 4 * n),
    'EH9S': (9, n**2 + 10 * n, (1 / 3) * (n**3 - n) + 7 * n**2 + 11 * n),
    'I-ARJ8S (best case)': (8, n**2 + 5 * n, (1 / 3) * (n**3 - n) + 4 * n**2 + 6 * n),
}
series = {name: p ** (1.0 / (d + op)) for name, (p, d, op) in methods_CI.items()}

fig, ax = plt.subplots(figsize=(5.4, 4.1))
plot_series(ax, n, series, r'$CI$')
fig.tight_layout(pad=0.4)
fig.savefig('computational_index.pdf')
plt.close(fig)
print("Saved computational_index.pdf")

n_log = np.unique(np.logspace(1, 6, 400).astype(int))
n_log = n_log[n_log >= 2].astype(float)


def E_phi(d, p, n):
    return 1 - (d / p) * (2.0 / (n * (n + 1)))


def T_phi(op, p, n):
    return 1 - (op / p) * (6.0 / (n**3 + 3 * n**2 - n))


methods_CP = {
    'ARJ8S': (8, lambda n: n**2 + 4 * n, lambda n: (1 / 3) * (n**3 - n) + 4 * n**2 + 5 * n),
    'CTT8S': (8, lambda n: n**2 + 3 * n, lambda n: (1 / 3) * (n**3 - n) + 3 * n**2 + 8 * n),
    'ZO8S': (8, lambda n: 2 * n**2 + 3 * n, lambda n: (2 / 3) * (n**3 - n) + 3 * n**2 + 4 * n),
    'EH9S': (9, lambda n: n**2 + 10 * n, lambda n: (1 / 3) * (n**3 - n) + 7 * n**2 + 11 * n),
    'I-ARJ8S (best case)': (8, lambda n: n**2 + 5 * n,
                             lambda n: (1 / 3) * (n**3 - n) + 4 * n**2 + 6 * n),
}
series = {}
for name, (p, dfun, opfun) in methods_CP.items():
    d = dfun(n_log)
    op = opfun(n_log)
    Ephi = E_phi(d, p, n_log)
    Tphi = T_phi(op, p, n_log)
    series[name] = 100 * 0.5 * (Ephi + Tphi)

fig, ax = plt.subplots(figsize=(5.4, 4.1))
plot_series(ax, n_log, series, r'$CP$ (%)')
fig.tight_layout(pad=0.4)
fig.savefig('computational_performance.pdf')
plt.close(fig)
print("Saved computational_performance.pdf")
