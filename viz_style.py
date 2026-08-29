import matplotlib.pyplot as plt
import seaborn as sns

BG = '#071119'
SURFACE = '#0B1B26'
SURFACE_ALT = '#102733'
BORDER = '#24404D'
GRID = '#1B3542'
TEXT = '#F3F7F8'
MUTED = '#A8BBC1'
ACCENT = '#62C7D0'
ACCENT_LIGHT = '#8ADBE1'
ACCENT_DARK = '#3E8D9B'
SUCCESS = '#65C891'
WARNING = '#E8BD68'
DANGER = '#EF8D8D'

# Deliberately cool-neutral. Green, amber, and red are reserved for status.
PALETTE = [ACCENT, '#50AEB9', '#7CCDD3', '#467986', '#325D6A']

def apply_style():
    sns.set_theme(style='darkgrid', palette=PALETTE, font='DejaVu Sans')
    plt.rcParams.update({
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'figure.facecolor': BG,
        'savefig.facecolor': BG,
        'axes.facecolor': SURFACE,
        'axes.edgecolor': BORDER,
        'axes.labelcolor': MUTED,
        'axes.titlecolor': TEXT,
        'axes.titlesize': 13,
        'axes.titleweight': 600,
        'axes.titlepad': 16,
        'axes.labelsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.size': 10,
        'text.color': TEXT,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'grid.color': GRID,
        'grid.alpha': 0.7,
        'grid.linewidth': 0.7,
        'legend.facecolor': SURFACE_ALT,
        'legend.edgecolor': BORDER,
        'legend.labelcolor': TEXT,
        'legend.frameon': True,
        'legend.framealpha': 1,
    })

def save(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
