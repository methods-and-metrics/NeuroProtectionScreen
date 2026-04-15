# ------------------------------------------------------------
# penumbra_gradient_figures_minocycline.py
# Code to generate Supplementary Figures S4 and S5
# (spatial gradient strips + distance x time heatmap)
# ------------------------------------------------------------

import math
import numpy as np
import matplotlib.pyplot as plt

# Minocycline parameters from Table S2 (effective diffusivity D and consumption k)
D_um2_s = 248.16      # um^2/s
k_s     = 8.37e-6     # 1/s

# Human-scale slab used in the main text: L = 5 cm (two-sided; centre at 25 mm)
L_um = 50_000.0       # 50 mm = 5 cm
dx_um = 100.0         # 100 um grid (501 nodes across 5 cm, matching SI discretisation)
nx = int(round(L_um / dx_um)) + 1
dx = L_um / (nx - 1)

# Explicit stability-controlled timestep (same form as SI Section S1.5)
dt = 0.9 / (k_s + 2.0 * D_um2_s / (dx * dx))

alpha = D_um2_s * dt / (dx * dx)
assert alpha <= 0.5, f"Unstable explicit scheme (alpha={alpha:.3f})"

mid = nx // 2
x_um = np.linspace(0.0, L_um, nx)
x_mm_half = (x_um[:mid+1]) / 1000.0  # edge (0) -> centre (25 mm)

def step_boundary(t_s: float) -> float:
    # Step boundary upper bound: C_edge = 1 for t > 0
    return 1.0

def run_simulation(t_end_h: float, sample_every_s: float):
    """Simulate Ct = D*Cxx - k*C with two-sided Dirichlet step boundary.
    Returns: times_h, C_half_history (time x depth), where depth is 0..25 mm."""
    t_end_s = t_end_h * 3600.0
    nsteps = int(math.ceil(t_end_s / dt))
    sample_every = max(1, int(round(sample_every_s / dt)))

    C = np.zeros(nx, dtype=float)
    C[0] = 1.0
    C[-1] = 1.0

    times_h = []
    hist = []

    for step in range(nsteps + 1):
        t = step * dt

        if step % sample_every == 0 or step == nsteps:
            times_h.append(t / 3600.0)
            hist.append(C[:mid+1].copy())

        if step == nsteps:
            break

        b = step_boundary(t)

        C_new = C.copy()
        C_new[1:-1] = (
            C[1:-1]
            + alpha * (C[2:] - 2.0 * C[1:-1] + C[:-2])
            - (k_s * dt) * C[1:-1]
        )
        C_new[0] = b
        C_new[-1] = b
        C = C_new

    return np.asarray(times_h), np.vstack(hist)

# --------------------------
# Figure S4: spatial strips
# --------------------------
# Selected time points (hours) to show the rim-to-core gradient
strip_times_h = [6, 24, 72, 120]
t_end_h = max(strip_times_h)
t_h, C_hist = run_simulation(t_end_h=t_end_h, sample_every_s=600.0)  # 10 min samples

# Extract the profiles closest to the target times
strip_profiles = []
strip_labels = []
for th in strip_times_h:
    idx = int(np.argmin(np.abs(t_h - th)))
    prof = C_hist[idx, :]
    strip_profiles.append(prof)
    strip_labels.append(f"t = {th:g} h (centre {100*prof[-1]:.3g}%)")

strip_profiles = np.vstack(strip_profiles)

fig, ax = plt.subplots(figsize=(9.5, 3.2), dpi=200, constrained_layout=True)
im = ax.imshow(
    strip_profiles,
    aspect="auto",
    origin="lower",
    extent=[x_mm_half[0], x_mm_half[-1], 0, strip_profiles.shape[0]],
)
ax.set_xlabel("Distance from lesion edge (mm)  ->  centre at 25 mm")
ax.set_yticks(np.arange(strip_profiles.shape[0]) + 0.5)
ax.set_yticklabels(strip_labels)
ax.set_title("Minocycline: rim-to-core spatial concentration gradient (two-sided slab, L = 5 cm; step boundary)")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Normalised concentration  C(x,t) / C_edge")
fig.savefig("minocycline_spatial_gradient_selected_times_L5cm_step.png", dpi=300)

# ---------------------------------------
# Figure S5: distance x time penetration
# ---------------------------------------
# Full time window for heatmap (0 to 120 h). Use 10 min sampling for plot.
t_end_h = 120.0
t_h, C_hist = run_simulation(t_end_h=t_end_h, sample_every_s=600.0)  # 10 min

# Compute time to reach 1% at each depth (first-passage time)
f_thr = 0.01
t_to_1pct = np.full(C_hist.shape[1], np.nan)
for j in range(C_hist.shape[1]):
    idxs = np.where(C_hist[:, j] >= f_thr)[0]
    if len(idxs) > 0:
        t_to_1pct[j] = t_h[idxs[0]]

fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200, constrained_layout=True)
im = ax.imshow(
    C_hist,
    origin="lower",
    aspect="auto",
    extent=[x_mm_half[0], x_mm_half[-1], t_h[0], t_h[-1]],
)
ax.set_xlabel("Distance from lesion edge (mm)  ->  centre at 25 mm")
ax.set_ylabel("Time since boundary exposure (h)")
ax.set_title("Minocycline: time-resolved rim-to-core penetration (two-sided slab, L = 5 cm)")

# Overlay depth-dependent time-to-1% curve
ax.plot(x_mm_half, t_to_1pct, linewidth=2, label="Time to reach 1% at each depth")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Normalised concentration  C(x,t) / C_edge (step boundary)")

ax.legend(frameon=False, loc="upper right")
fig.savefig("minocycline_distance_time_heatmap_L5cm_step.png", dpi=300)
plt.close("all")
