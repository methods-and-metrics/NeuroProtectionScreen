# ------------------------------------------------------------
# penumbra_gradient_heatmap_all_drugs.py
# Code to generate Supplementary Figure S6
# (cross-drug steady-state rim-to-core gradient heatmap)
# ------------------------------------------------------------
#
# Model: two-sided 1D slab (0 <= x <= L), with C(0)=C(L)=1 (step boundary upper bound).
# Steady-state solution:
#   C(x) = cosh((x - L/2)/lambda) / cosh((L/2)/lambda),  where lambda = sqrt(D/k).
#
# We plot the half-slab from the lesion edge to the centre:
#   s = distance from the edge, 0 <= s <= d, where d = L/2.
# The steady-state fraction is:
#   C(s)/C_edge = cosh((s - d)/lambda) / cosh(d/lambda).
#
# D and k are taken from the parameter sets used in the main text (Table 1) and
# derived in SI Table S2.

import math
import numpy as np
import matplotlib.pyplot as plt

# Parameters (Table S2)
DRUGS = ["Edaravone", "Fasudil", "MgSO4", "Uric Acid", "NXY-059", "Minocycline"]

D_um2_s = {
    "Edaravone": 342.39,
    "Fasudil": 288.44,
    "MgSO4": 387.28,
    "Uric Acid": 346.47,
    "NXY-059": 263.70,
    "Minocycline": 248.16,
}

k_s = {
    "Edaravone": 3.50e-5,
    "Fasudil": 4.81e-4,
    "MgSO4": 3.703e-5,
    "Uric Acid": 1.44e-4,
    "NXY-059": 3.83e-9,
    "Minocycline": 8.37e-6,
}

# Geometry / distance
L_mm = 50.0
d_mm = L_mm / 2.0
d_um = d_mm * 1000.0

# Threshold (fraction of edge concentration)
f = 0.01  # 1%

# Sampling grid (edge to centre)
nx = 800
s_mm = np.linspace(0.0, d_mm, nx)
s_um = s_mm * 1000.0

def steady_state_fraction_profile(s_um, d_um, D_um2_s, k_s):
    lam_um = math.sqrt(D_um2_s / k_s)
    return np.cosh((s_um - d_um) / lam_um) / np.cosh(d_um / lam_um)

def max_depth_at_threshold_mm(D_um2_s, k_s, d_mm, f):
    """Maximum depth from rim with C/C_edge >= f at steady state (two-sided slab)."""
    lam_um = math.sqrt(D_um2_s / k_s)
    z = (d_mm * 1000.0) / lam_um
    y = f * math.cosh(z)
    # If y < 1, then even the centre is above threshold; report full depth.
    if y < 1.0:
        return d_mm
    # Otherwise solve cosh((d - s)/lambda) = f*cosh(d/lambda)
    s_mm = d_mm - (lam_um / 1000.0) * math.acosh(y)
    return max(0.0, min(d_mm, s_mm))

# Build heatmap matrix in log10 space for dynamic range
# (reversed order so the first listed drug appears at the top of the axis)
DRUGS_PLOT = list(reversed(DRUGS))
M = np.zeros((len(DRUGS_PLOT), nx), dtype=float)
s_thresh_mm = []

for i, drug in enumerate(DRUGS_PLOT):
    frac = steady_state_fraction_profile(s_um, d_um, D_um2_s[drug], k_s[drug])
    frac = np.clip(frac, 1e-18, 1.0)
    M[i, :] = np.log10(frac)
    s_thresh_mm.append(max_depth_at_threshold_mm(D_um2_s[drug], k_s[drug], d_mm, f))

# Plot
plt.figure(figsize=(7.6, 3.6))
ax = plt.gca()

im = ax.imshow(
    M,
    aspect="auto",
    origin="lower",
    extent=[0.0, d_mm, -0.5, len(DRUGS_PLOT) - 0.5],
    vmin=-14,
    vmax=0,
)

ax.set_yticks(range(len(DRUGS_PLOT)))
ax.set_yticklabels(DRUGS_PLOT)
ax.set_xlabel("Distance from lesion edge toward centre (mm)")
ax.set_ylabel("Drug")
ax.set_title("Steady-state rim-to-core gradients (two-sided slab; L = 5 cm; step boundary upper bound)")

# Mark max depth where C/C_edge >= 1% at steady state
for i, s_thr in enumerate(s_thresh_mm):
    if s_thr >= d_mm - 1e-3:
        ax.scatter([d_mm], [i], marker=">", s=50, color="white", clip_on=False)
    else:
        ax.plot([s_thr, s_thr], [i - 0.35, i + 0.35], linewidth=2.0, color="white", clip_on=False)

# Mark centre depth
ax.axvline(d_mm, linestyle=":", linewidth=1.0, color="white", clip_on=False)
ax.set_xlim(0.0, d_mm + 0.5)

cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("log10(C / C_edge) at steady state")

ax.text(
    0.01,
    0.02,
    "White tick/arrow: maximum depth with C/C_edge >= 1%",
    transform=ax.transAxes,
    fontsize=8,
    color="white",
    ha="left",
    va="bottom",
)

plt.tight_layout()
plt.savefig("figure_penumbra_gradient_edge_to_center_heatmap.png", dpi=300)
plt.show()
