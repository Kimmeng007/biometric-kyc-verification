"""ROC plotting shared by all three evaluations (static PNGs for the README)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.metrics import OPERATING_POINTS, far_frr_curve

# Validated categorical slots 1-3 (fixed order: the modality keeps its colour everywhere).
COLORS = {"Face": "#2a78d6", "Voice": "#eb6834", "Fused (face + voice)": "#1baf7a"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlecolor": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})


FLOOR = 1e-4  # error rates of exactly 0 are drawn on this floor of the log axes


def _pct(v, _pos=None):
    return f"{100 * v:g}%"


def plot_roc_panels(panels: dict, out_path: Path, suptitle: str) -> None:
    """ROC curves in the biometric DET convention.

    panels = {panel title: {series name: (scores, labels, eer)}}.
    x = FAR (impostors accepted), y = FRR (genuine customers rejected), both log scale,
    so near-perfect systems stay distinguishable. Lower-left is better; the dotted
    diagonal is FAR == FRR, where each curve crosses it is that system's EER.
    """
    from matplotlib.ticker import FuncFormatter

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 5.4), sharey=True, squeeze=False)
    for ax, (title, series) in zip(axes[0], panels.items()):
        ax.plot([FLOOR, 1], [FLOOR, 1], color=AXIS, lw=1, ls=(0, (1, 2)), zorder=1)
        for far_op in OPERATING_POINTS:
            ax.axvline(far_op, color=AXIS, lw=1, ls=(0, (3, 3)), zorder=1)
            ax.text(far_op * 1.12, 0.05, f"FAR {100 * far_op:g}%", color=MUTED, fontsize=8,
                    va="bottom", ha="left", transform=ax.get_xaxis_transform())
        for name, (scores, labels, eer) in series.items():
            _, far, frr = far_frr_curve(scores, labels)
            keep = far >= FLOOR * 0.8  # below this FAR there are too few impostors to measure
            ax.plot(far[keep], np.clip(frr[keep], FLOOR, 1), color=COLORS[name], lw=2,
                    solid_capstyle="round", label=f"{name}  EER {100 * eer:.2f}%", zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(FLOOR * 0.8, 1)
        ax.set_ylim(FLOOR * 0.8, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(_pct))
        ax.yaxis.set_major_formatter(FuncFormatter(_pct))
        ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("False Accept Rate - impostors let in")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(series),
                        frameon=False, fontsize=8.5, labelcolor=INK, handlelength=1.6, columnspacing=1.2)
        for line in leg.get_lines():
            line.set_linewidth(3)
    axes[0][0].set_ylabel("False Reject Rate - genuine customers blocked")
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
