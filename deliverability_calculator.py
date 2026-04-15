#!/usr/bin/env python3
"""
Deliverability (reachability) calculator for diffusion–reaction transport in hypoperfused tissue.

Core model:
    ∂C/∂t = D ∇²C − k C

Steady-state "center fraction" for constant boundary concentration (normalized):
- Slab (two-sided, thickness L):      Css_center = 1 / cosh((L/2)/λ)
- Cylinder (radial, radius R):        Css_center = 1 / I0(R/λ)
- Sphere (radial, radius R):          Css_center = (R/λ) / sinh(R/λ)
where λ = sqrt(D/k) is the penetration length.

This script supports:
- Analytic steady-state reachability for a chosen threshold f (e.g., 0.01)
- Optional time-to-threshold under a step boundary condition for the slab (explicit FD)

Units:
- D in µm^2/s
- k in 1/s
- L and R in µm (internal); use mm in CLI inputs.

Note: This is an "upper bound" calculator when used with step boundary conditions.
"""

from __future__ import annotations
import argparse
import math
import numpy as np

# -----------------------
# Core analytic functions
# -----------------------

def penetration_length_um(D_um2_s: float, k_s: float) -> float:
    if k_s <= 0:
        return float("inf")
    return math.sqrt(D_um2_s / k_s)

def css_center_slab_two_sided(D_um2_s: float, k_s: float, L_um: float) -> float:
    """Two-sided slab of full thickness L (midpoint at L/2)."""
    lam = penetration_length_um(D_um2_s, k_s)
    if not math.isfinite(lam):
        return 1.0
    z = (L_um / 2.0) / lam
    if z > 700:
        return 0.0
    return 1.0 / math.cosh(z)

def css_center_cylinder(D_um2_s: float, k_s: float, R_um: float) -> float:
    """Radial cylinder with boundary at r=R and regularity at r=0."""
    lam = penetration_length_um(D_um2_s, k_s)
    if not math.isfinite(lam):
        return 1.0
    z = R_um / lam
    # I0 available via numpy
    return 1.0 / float(np.i0(z))

def css_center_sphere(D_um2_s: float, k_s: float, R_um: float) -> float:
    """Radial sphere with boundary at r=R and regularity at r=0."""
    lam = penetration_length_um(D_um2_s, k_s)
    if not math.isfinite(lam):
        return 1.0
    z = R_um / lam
    if z > 700:
        return 0.0
    # center limit: z/sinh(z)
    return float(z / math.sinh(z))

def reachable(css: float, f: float) -> bool:
    return css >= f

# --------------------------------------
# Optional: time-to-threshold (slab only)
# --------------------------------------

def simulate_center_fraction_slab_step(
    D_um2_s: float,
    k_s: float,
    L_um: float,
    dx_target_um: float,
    t_end_h: float,
    max_points: int = 8000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Explicit FD solver for Ct = D*Cxx - k*C on [0,L] with two-sided step boundary:
        C(0,t)=C(L,t)=1 for t>0, C(x,0)=0
    Returns (time_h, center_fraction).
    """
    Nx = int(round(L_um / dx_target_um)) + 1
    dx = L_um / (Nx - 1)

    dt = 0.9 / (k_s + 2.0 * D_um2_s / (dx * dx))
    t_end_s = t_end_h * 3600.0
    n_steps = int(math.ceil(t_end_s / dt))
    save_every = max(1, n_steps // max_points)

    C = np.zeros(Nx, dtype=float)
    C[0] = 1.0
    C[-1] = 1.0
    center_idx = Nx // 2

    alpha = D_um2_s * dt / (dx * dx)
    t = 0.0

    times_h = []
    frac = []

    for step in range(n_steps + 1):
        if step % save_every == 0 or step == n_steps:
            times_h.append(t / 3600.0)
            frac.append(C[center_idx])

        if step == n_steps:
            break

        C_new = C.copy()
        C_new[1:-1] = (
            C[1:-1]
            + alpha * (C[2:] - 2.0 * C[1:-1] + C[:-2])
            - (k_s * dt) * C[1:-1]
        )
        C_new[0] = 1.0
        C_new[-1] = 1.0
        C = C_new
        t += dt

    return np.asarray(times_h), np.asarray(frac)

def time_to_fraction(time_h: np.ndarray, frac: np.ndarray, f: float) -> float | None:
    idx = np.where(frac >= f)[0]
    return None if len(idx) == 0 else float(time_h[idx[0]])

# -----------------------
# Convenience conversions
# -----------------------

def mm_to_um(x_mm: float) -> float:
    return x_mm * 1000.0

def equiv_radius_from_volume_ml(V_ml: float) -> float:
    """Equivalent sphere radius (mm) for volume in mL (1 mL = 1 cm^3 = 1000 mm^3)."""
    V_mm3 = V_ml * 1000.0
    R_mm = (3.0 * V_mm3 / (4.0 * math.pi)) ** (1.0/3.0)
    return R_mm

# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Diffusion–reaction deliverability calculator (reachability + optional time-to-threshold).")
    ap.add_argument("--geometry", choices=["slab", "cylinder", "sphere"], default="slab", help="Geometry for steady-state reachability.")
    ap.add_argument("--distance_mm", type=float, required=True, help="Characteristic distance in mm: slab uses full thickness L; cylinder/sphere use radius R.")
    ap.add_argument("--D_um2_s", type=float, required=True, help="Effective diffusion coefficient D (µm^2/s).")
    ap.add_argument("--k_s", type=float, required=True, help="Effective first-order consumption k (1/s).")
    ap.add_argument("--threshold", type=float, default=0.01, help="Threshold fraction f (default 0.01 for 1%%).")
    ap.add_argument("--time_to_threshold", action="store_true", help="Also estimate time-to-threshold for slab under step boundary (upper bound).")
    ap.add_argument("--t_end_h", type=float, default=120.0, help="End time (h) for time-to-threshold simulation (slab only).")
    ap.add_argument("--dx_um", type=float, default=100.0, help="Spatial step (µm) for time-to-threshold simulation.")
    ap.add_argument("--volume_ml", type=float, default=None, help="Optional infarct/core volume (mL) to convert to equivalent sphere radius; overrides distance_mm if provided.")
    args = ap.parse_args()

    if args.volume_ml is not None:
        R_mm = equiv_radius_from_volume_ml(args.volume_ml)
        print(f"[info] volume_ml={args.volume_ml:g} mL -> equivalent sphere radius R={R_mm:.2f} mm")
        args.geometry = "sphere"
        args.distance_mm = R_mm

    f = args.threshold
    D = args.D_um2_s
    k = args.k_s

    if args.geometry == "slab":
        css = css_center_slab_two_sided(D, k, mm_to_um(args.distance_mm))
    elif args.geometry == "cylinder":
        css = css_center_cylinder(D, k, mm_to_um(args.distance_mm))
    else:
        css = css_center_sphere(D, k, mm_to_um(args.distance_mm))

    lam_um = penetration_length_um(D, k)
    t12_h = (math.log(2.0) / k) / 3600.0 if k > 0 else float("inf")

    print(f"geometry={args.geometry}  distance={args.distance_mm:g} mm")
    print(f"D={D:g} µm^2/s  k={k:g} 1/s  ->  λ={lam_um/1000.0:.3f} mm  (consumption t1/2={t12_h:.2f} h)")
    print(f"Css_center/C_boundary={css:.6g}")
    print(f"Reachable at f={f:g}?  {'YES' if css >= f else 'NO (NR)'}")

    if args.time_to_threshold:
        if args.geometry != "slab":
            print("[warn] time-to-threshold simulation currently implemented only for slab; skipping.")
            return
        t, frac = simulate_center_fraction_slab_step(D, k, mm_to_um(args.distance_mm), args.dx_um, args.t_end_h)
        tt = time_to_fraction(t, frac, f)
        if tt is None:
            print(f"t_to_{f:g}: not reached by {args.t_end_h:g} h")
        else:
            print(f"t_to_{f:g}: {tt:.2f} h (step-boundary upper bound)")

if __name__ == "__main__":
    main()
