from __future__ import annotations

import matplotlib.pyplot as plt

MU_SB = "μ"
DEG_SB = "°"


def apply_plot_style():
    plt.style.use("default")
    plt.rc("figure", figsize=[5, 5])
    plt.rc("font", size=14, family="Arial")
    plt.rc("axes", labelsize=14, titlesize=14)
    plt.rc("legend", fontsize=12)
    plt.rc("xtick", labelsize=11)
    plt.rc("ytick", labelsize=11)
    plt.rc("lines", linewidth=2)
    plt.rc("figure", dpi=100)
    plt.rc("savefig", dpi=150, bbox="tight")
