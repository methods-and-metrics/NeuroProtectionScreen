#!/usr/bin/env python3

"""
Reachability phase diagrams for the diffusion–reaction model in a two-sided slab.

Steady-state midpoint fraction (normalized to boundary):
    Css = 1 / cosh(d / λ),    λ = sqrt(D/k),   d = L/2
Reachability condition Css ≥ f implies:
    k ≤ kcrit(d,f) = D * (acosh(1/f) / d)^2

This script plots kcrit versus midpoint distance d for multiple thresholds f, and overlays
drug-specific k values.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def k_crit_two_sided_slab(D_um2_s: float, d_mm: np.ndarray, f: float) -> np.ndarray:
    d_um = d_mm * 1000.0  # mm -> µm
    a = np.arccosh(1.0 / f)
    return D_um2_s * (a / d_um) ** 2

def d_max_for_threshold(D_um2_s: float, k_s: float, f: float) -> float:
    a = math.acosh(1.0 / f)
    lam_um = math.sqrt(D_um2_s / k_s)
    d_um = a * lam_um
    return d_um / 1000.0

def make_phase_plot(drug_name: str, D_um2_s: float, k_s: float, out_png: str, out_pdf: str):
    d_mm = np.linspace(1.0, 40.0, 400)
    thresholds = [(0.1, "10%"), (0.01, "1%"), (0.001, "0.1%")]
    linestyles = ["-", "--", ":"]

    plt.figure(figsize=(7.8, 5.2))
    for (f, lab), ls in zip(thresholds, linestyles):
        plt.plot(d_mm, k_crit_two_sided_slab(D_um2_s, d_mm, f), linestyle=ls,
                 label=f"Reachable if k ≤ kcrit (Css ≥ {lab})")

    plt.axhline(k_s, linestyle="-.")
    plt.axvline(25.0, linestyle="-.")  # d=25 mm corresponds to L=5 cm

    plt.yscale("log")
    plt.xlabel("Midpoint diffusion distance d = L/2 (mm)")
    plt.ylabel("k (s⁻¹)  [reachability boundary]")
    plt.title(f"Phase diagram (two-sided slab): {drug_name}\nD={D_um2_s:.2f} µm²/s, k={k_s:.2e} s⁻¹")
    plt.xlim(1, 40)

    # annotate maximum reachable distances at the drug’s k
    y_positions = [3e-4, 8e-5, 2e-5]
    for (f, lab), y in zip(thresholds, y_positions):
        dmax = d_max_for_threshold(D_um2_s, k_s, f)
        plt.text(2.0, y, f"Max d for Css≥{lab}: {dmax:.1f} mm", fontsize=9)

    plt.legend(frameon=False, fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.show()

if __name__ == "__main__":
    edaravone = dict(name="Edaravone (high-consumption scenario)", D=342.39, k=3.50e-5)
    nxy059   = dict(name="NXY-059 (low-consumption scenario)", D=263.70, k=3.83e-9)

    make_phase_plot(edaravone["name"], edaravone["D"], edaravone["k"],
                    "PhaseDiagram_Edaravone.png", "PhaseDiagram_Edaravone.pdf")
    make_phase_plot(nxy059["name"], nxy059["D"], nxy059["k"],
                    "PhaseDiagram_NXY059.png", "PhaseDiagram_NXY059.pdf")
