# Run: streamlit run streamlit_app_deliverability.py

import math
import numpy as np
import streamlit as st

st.set_page_config(page_title="Stroke Neuroprotection Deliverability Calculator", layout="centered")

st.title("Deliverability (Reachability) Calculator")
st.caption("Upper-bound screening tool for whether a systemically available drug can physically reach the lesion center via diffusion in hypoperfused tissue.")

def penetration_length_um(D_um2_s: float, k_s: float) -> float:
    return math.sqrt(D_um2_s / k_s) if k_s > 0 else float("inf")

def css_slab(D, k, L_um):
    lam = penetration_length_um(D, k)
    if not math.isfinite(lam):
        return 1.0
    z = (L_um/2)/lam
    return 1/math.cosh(z)

def css_cyl(D, k, R_um):
    lam = penetration_length_um(D, k)
    if not math.isfinite(lam):
        return 1.0
    z = R_um/lam
    return 1/float(np.i0(z))

def css_sph(D, k, R_um):
    lam = penetration_length_um(D, k)
    if not math.isfinite(lam):
        return 1.0
    z = R_um/lam
    return float(z/math.sinh(z))

geo = st.selectbox("Geometry", ["Slab (two-sided)", "Cylinder (radial)", "Sphere (radial)"])
dist_mm = st.number_input("Characteristic distance (mm)", min_value=0.1, value=25.0, step=0.5,
                          help="Slab uses full thickness L. Cylinder/Sphere use radius R (center-to-rim).")
D = st.number_input("D (µm²/s)", min_value=1.0, value=300.0, step=10.0)
k = st.number_input("k (1/s)", min_value=0.0, value=1e-5, format="%.2e")
f = st.number_input("Threshold f (fraction of boundary)", min_value=1e-6, max_value=1.0, value=0.01, format="%.4f")

dist_um = dist_mm*1000.0
lam_um = penetration_length_um(D, k)
lam_mm = lam_um/1000.0 if math.isfinite(lam_um) else float("inf")
t12_h = (math.log(2)/k)/3600 if k > 0 else float("inf")

if "Slab" in geo:
    css = css_slab(D, k, dist_um)
elif "Cylinder" in geo:
    css = css_cyl(D, k, dist_um)
else:
    css = css_sph(D, k, dist_um)

st.subheader("Results")
col1, col2, col3 = st.columns(3)
col1.metric("Penetration length λ (mm)", f"{lam_mm:.3f}" if math.isfinite(lam_mm) else "∞")
col2.metric("Consumption half-life ln2/k (h)", f"{t12_h:.2f}" if math.isfinite(t12_h) else "∞")
col3.metric("Css_center / C_boundary", f"{css:.3g}")

if css >= f:
    st.success(f"Reachable at threshold f = {f:g} (steady-state criterion).")
else:
    st.error(f"NOT reachable (NR): Css_center < f. Even at infinite time, center remains below threshold under this D,k,distance.")

st.caption("Interpretation: This is a steady-state reachability screen under constant boundary concentration. "
           "Time-to-threshold within hours may still fail even if reachable, when diffusion distances are large.")
 
