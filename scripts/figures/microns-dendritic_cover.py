"""MICrONS dendritic clustering cover: raw near-pair clustering vs what presynaptic soma geometry alone produces,
and the geometry-adjusted estimate, from saved result JSON only.

Sources (repo ~/projects/microns-dendritic, public at github.com/rohit-ravi2/microns-dendritic):
  results/calibration/strict_*.json      planted_geometry["0.0"].delta_std  (soma-position labels, naive within-cell shuffle)
  results/primary/summary.json           delta_d (naive raw contrast), beta_d, d_upper95 (geometry-adjusted, 95% null interval)
  results/auto/A11_auto_*.json           held-out auto-co-registration tier
  results/robust_summary.csv             ASP (spine-only) and AE (Euclidean-window) broad rows
  results/v1dd/V1DD_all_natural_movie.json  second animal; geometry_check.naive_delta_d is its planted-geometry contrast
The interval is symmetric: lower = 2 * beta_d - d_upper95 (robust.py computes beta +/- 1.96 null SD).
"""
import csv, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = (sys.argv[2] if len(sys.argv) > 2 else "/home/rohit/projects/microns-dendritic") + "/results/"
out = sys.argv[1] if len(sys.argv) > 1 else "public/images/projects/microns-dendritic-cover.png"
load = lambda p: json.load(open(R + p))

prim = load("primary/summary.json")
rows_a = []  # (label, planted, raw, beta, upper)
for key, lab in (("strict_excitatory_rf", "Pyramidal · RF location"), ("strict_excitatory_feat", "Pyramidal · tuning feature"),
                 ("strict_inhibitory_rf", "Interneuron · RF location"), ("strict_inhibitory_feat", "Interneuron · tuning feature")):
    cal = load(f"calibration/{key}.json")["planted_geometry"]["0.0"]["delta_std"]
    p = prim[key]
    rows_a.append((lab, cal, p["delta_d"], p["beta_d"], p["d_upper95"]))

rob = {r["tag"]: r for r in csv.DictReader(open(R + "robust_summary.csv"))}
rows_b = []
for key, lab in (("A11_auto_excitatory_rf", "Held-out tier · pyr RF"), ("A11_auto_inhibitory_feat", "Held-out tier · int feature")):
    a = load(f"auto/{key}.json")
    rows_b.append((lab, None, a["delta_d"], a["beta_d"], a["d_upper95"]))
for key, lab in (("ASP_broad_excitatory_rf", "Spine-only · pyr RF"), ("AE_broad_inhibitory_rf", "Euclidean windows · int RF")):
    r = rob[key]
    rows_b.append((lab, None, float(r["delta_d"]), float(r["beta_d"]), float(r["d_upper95"])))
v = load("v1dd/V1DD_all_natural_movie.json")
rows_b.append(("Second animal (V1DD) · movie corr.", v["geometry_check"]["naive_delta_d"], v["delta_d"], v["beta_d"], v["d_upper95"]))

INK, MUTED, GRID, BAR, RAW = "#1f2328", "#59636e", "#e6e8eb", "#4a6fa5", "#c8553d"
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), dpi=200, sharex=True)
for ax, rows, title in ((axes[0], rows_a, "MICrONS, four pre-registered tests"),
                        (axes[1], rows_b, "Replications")):
    ys = list(range(len(rows)))[::-1]
    for y, (lab, planted, raw, beta, up) in zip(ys, rows):
        lo = 2 * beta - up
        ax.plot([lo, up], [y - 0.14] * 2, color=BAR, lw=2.2, solid_capstyle="round")
        ax.plot(beta, y - 0.14, "o", color=BAR, ms=6.5)
        ax.plot(raw, y + 0.1, "o", color=RAW, ms=6.5)
        if planted is not None:
            ax.plot(planted, y + 0.1, "o", mfc="white", mec=INK, mew=1.6, ms=8)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_title(title, fontsize=11.5, color=INK, pad=8)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    ax.set_xlabel("near-pair similarity excess (standardized d)", color=MUTED, fontsize=10)
    ax.set_ylim(-0.7, len(rows) - 0.4)
axes[0].set_xlim(-0.08, 0.23)

h = [plt.Line2D([], [], ls="", marker="o", mfc="white", mec=INK, mew=1.6, ms=8),
     plt.Line2D([], [], ls="", marker="o", color=RAW, ms=6.5),
     plt.Line2D([], [], ls="-", marker="o", color=BAR, ms=6.5, lw=2.2)]
fig.legend(h, ["labels = presynaptic soma position only, naive shuffle", "real labels, naive within-cell contrast",
               "real labels, geometry-adjusted (95% null interval)"],
           loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
fig.suptitle("The raw clustering is the size geometry alone produces; once geometry is regressed out, nothing detectable remains",
             fontsize=13, color=INK, y=0.985)
fig.tight_layout(rect=(0, 0.07, 1, 0.97))
fig.savefig(out, facecolor="white")
for r in rows_a + rows_b:
    print(r[0], [None if x is None else round(x, 4) for x in r[1:]])
