import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.utils.dataframe import dataframe_to_rows

# -----------------------
# Parameters (from Table 1 / Table S2)
# -----------------------
drugs = [
    ("Edaravone",   342.39, 3.50e-05),
    ("Fasudil",     288.44, 4.81e-04),
    ("MgSO4",       387.28, 3.703e-05),
    ("Uric Acid",   346.47, 1.440e-04),
    ("NXY-059",     263.70, 3.83e-09),
    ("Minocycline", 248.16, 8.37e-06),
]

# Clinical slab thicknesses (L) in cm (full thickness edge-to-edge)
L_cm_values = [1, 2, 5]

def css_center_fraction_two_sided_slab(D_um2_s: float, k_s: float, L_cm: float) -> float:
    """
    Analytic steady-state midpoint fraction for a symmetric two-sided slab
    with clamped boundaries (edge concentration fixed).
        Css_center / C_edge = 1 / cosh( (L/2) / sqrt(D/k) )
    D in µm^2/s, k in 1/s, L in cm.
    """
    L_um = L_cm * 10_000.0  # cm -> µm
    if k_s <= 0:
        return 1.0
    lam_um = math.sqrt(D_um2_s / k_s)  # penetration length λ in µm
    z = (L_um / 2.0) / lam_um
    if z > 700:  # avoid cosh overflow
        return 0.0
    return 1.0 / math.cosh(z)

# -----------------------
# Build data table (% of edge)
# -----------------------
rows = []
for name, D, k in drugs:
    row = {"Drug": name}
    for L_cm in L_cm_values:
        row[f"L = {L_cm} cm"] = 100.0 * css_center_fraction_two_sided_slab(D, k, L_cm)
    rows.append(row)

df = pd.DataFrame(rows)

# -----------------------
# PNG figure (matplotlib)
# -----------------------
fig_png_path = "/mnt/data/Fig3_panelA_steady_state_percent_edge_L1cm_2cm_5cm.png"

x = np.arange(len(df["Drug"]))
width = 0.26

plt.figure(figsize=(10, 4.8))
for i, L_cm in enumerate(L_cm_values):
    plt.bar(x + (i - 1) * width, df[f"L = {L_cm} cm"], width=width, label=f"L = {L_cm} cm")

plt.xticks(x, df["Drug"], rotation=0)
plt.ylabel("Steady-state centre concentration (% of edge)")
plt.xlabel("Drug")
plt.ylim(0, 105)  # linear axis in %
plt.legend(frameon=False, ncols=3, loc="upper right")
plt.tight_layout()
plt.savefig(fig_png_path, dpi=300)
plt.close()

# -----------------------
# Excel workbook with data + editable chart
# -----------------------
xlsx_path = "/mnt/data/Fig3_panelA_steady_state_percent_edge_L1cm_2cm_5cm.xlsx"

wb = Workbook()
ws = wb.active
ws.title = "Data"

for r in dataframe_to_rows(df, index=False, header=True):
    ws.append(r)

ws.column_dimensions["A"].width = 16
ws.column_dimensions["B"].width = 12
ws.column_dimensions["C"].width = 12
ws.column_dimensions["D"].width = 12
ws.freeze_panes = "A2"

chart = BarChart()
chart.type = "col"
chart.grouping = "clustered"
chart.overlap = 0
chart.title = "Figure 3A: Steady-state centre concentration (% of edge)"
chart.y_axis.title = "% of edge"
chart.x_axis.title = "Drug"

data_ref = Reference(ws, min_col=2, max_col=4, min_row=1, max_row=1 + len(df))
cats_ref = Reference(ws, min_col=1, min_row=2, max_row=1 + len(df))
chart.add_data(data_ref, titles_from_data=True)
chart.set_categories(cats_ref)
chart.y_axis.scaling.min = 0
chart.y_axis.scaling.max = 105

ws.add_chart(chart, "F2")
wb.save(xlsx_path)

print("Wrote:", fig_png_path)
print("Wrote:", xlsx_path)
