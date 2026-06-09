#!/usr/bin/env python3
"""
Plot configuration for the npj Clean Water Perspective.

Color palette follows Nature's official colorblind-safe scheme.
Ref: https://www.nature.com/documents/natrev-artworkguide.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# Color Palette — Macaron (blue-dominant, low saturation, high brightness)
# =============================================================================

# Macaron 7-color qualitative palette
MACARON_COLORS = {
    'baby_blue':   '#7EB6D9',  # 婴儿蓝
    'sky_blue':    '#A8D8EA',  # 天空蓝
    'mint':        '#98D8C8',  # 薄荷绿
    'lavender':    '#C7CEEA',  # 薰衣草
    'sakura':      '#F2B5D4',  # 樱花粉
    'peach':       '#FFD1A9',  # 蜜桃橙
    'cream':       '#FFF3B0',  # 奶油黄
}

# Legacy alias for backward compatibility
NATURE_COLORS = MACARON_COLORS

# Model colors
MODEL_COLORS = {
    'qwen3.5-plus': '#0077BB',      # Blue
    'claude-sonnet-4.6': '#33BBEE',  # Cyan
    'gemini-3.1-pro': '#009988',     # Teal
}

MODEL_LABELS = {
    'qwen3.5-plus': 'Qwen 3.5-Plus',
    'claude-sonnet-4.6': 'Claude Sonnet 4.6',
    'gemini-3.1-pro': 'Gemini 3.1 Pro',
}

# Sub-field colors (Macaron palette mapped to 7 sub-fields)
SUBFIELD_COLORS = {
    'WWTP': '#7EB6D9',        # 婴儿蓝 — largest sub-field, primary color
    'monitoring': '#A8D8EA',  # 天空蓝 — water quality, blue family
    'control': '#98D8C8',     # 薄荷绿 — process control, cool transition
    'sludge': '#C7CEEA',      # 薰衣草 — sludge, purple-blue tone
    'membrane': '#F2B5D4',    # 樱花粉 — membrane systems, warm contrast
    'coagulation': '#FFD1A9', # 蜜桃橙 — core argument, warm highlight
    'DBP': '#FFF3B0',         # 奶油黄 — smallest sub-field, light
}

# Ordered sub-field list (by corpus size, descending)
SUBFIELD_ORDER = ['WWTP', 'monitoring', 'control', 'sludge', 'membrane', 'coagulation', 'DBP']

# Fig 3 comparison colors (macaron, matching SUBFIELD_COLORS)
FIG3_COLORS = {
    'coagulation': '#FFD1A9',  # 蜜桃橙 — coagulation pathway (warm)
    'control': '#98D8C8',      # 薄荷绿 — control pathway (cool)
    'others': '#C7CEEA',       # 薰衣草 — others/corpus baseline
}

# Prompt version colors
PROMPT_COLORS = {
    'v1': '#0077BB',
    'v2': '#33BBEE',
    'fewshot': '#009988',
}

PROMPT_LABELS = {
    'v1': 'Zero-shot (v1)',
    'v2': 'Optimized (v2)',
    'fewshot': 'Few-shot',
}

# Boolean field colors
BOOL_COLORS = ['#0077BB', '#D6EAF8']

# Sequential blue palette for heatmaps
HEATMAP_CMAP = 'Blues'

# =============================================================================
# Figure sizes (Nature Portfolio style: single column 89 mm, double column 183 mm)
# =============================================================================

FIGURE_SIZE_SINGLE = (3.5, 2.8)    # Single column (~89mm)
FIGURE_SIZE_1_5 = (5.5, 4.0)       # 1.5 column
FIGURE_SIZE_DOUBLE = (7.2, 5.0)    # Double column (~183mm)
FIGURE_SIZE_FULL = (7.2, 8.0)      # Full page double column
DPI = 300

# =============================================================================
# Font Sizes (Nature style: 7-8pt for labels, 8-9pt for titles)
# =============================================================================

FONT_SIZES = {
    'title': 10,
    'axis_label': 9,
    'tick_label': 8,
    'legend_title': 8,
    'legend': 7,
    'annotation': 7,
    'significance': 8,
}

# =============================================================================
# Line Widths
# =============================================================================

LINE_WIDTHS = {
    'main': 1.5,
    'secondary': 1.0,
    'grid': 0.3,
    'spine': 0.8,
    'box': 1.0,
    'whisker': 1.0,
    'bracket': 0.8,
}

# =============================================================================
# Style Functions
# =============================================================================

def apply_plot_style():
    """Apply Ocean Blue style to matplotlib rcParams."""
    plt.rcParams.update({
        'figure.dpi': DPI,
        'figure.facecolor': 'white',
        'font.family': 'Arial',
        'font.size': 8,
        'axes.titlesize': FONT_SIZES['title'],
        'axes.titleweight': 'bold',
        'axes.labelsize': FONT_SIZES['axis_label'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': LINE_WIDTHS['spine'],
        'axes.grid': False,
        'xtick.labelsize': FONT_SIZES['tick_label'],
        'ytick.labelsize': FONT_SIZES['tick_label'],
        'legend.fontsize': FONT_SIZES['legend'],
        'legend.title_fontsize': FONT_SIZES['legend_title'],
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#CCCCCC',
    })


def create_figure(size='double', nrows=1, ncols=1):
    """Create figure with standard settings."""
    apply_plot_style()
    sizes = {
        'single': FIGURE_SIZE_SINGLE,
        '1.5': FIGURE_SIZE_1_5,
        'double': FIGURE_SIZE_DOUBLE,
        'full': FIGURE_SIZE_FULL,
    }
    figsize = sizes.get(size, FIGURE_SIZE_DOUBLE)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=DPI)
    return fig, axes


def save_figure(fig, filepath, dpi=None):
    """Save figure with standard settings."""
    fig.savefig(filepath, dpi=dpi or DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {filepath}")
