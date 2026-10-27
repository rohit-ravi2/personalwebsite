"""asd-myelin-timing cover: which arms of the ladder survived their own controls, and the white-matter sign fix.

Every value is read from a small result file in ~/projects/asd-myelin/outputs/ (paths below). Left panel: -log10 p
before the control that decided each arm (hollow) and after it (filled). Right panel: autism x white-matter neurite
density (ICVF) genetic correlation as first reported, after the BIG40 allele-orientation fix, and in Gu et al. 2025.

Sources
  phase1/P1_RARE_BURDEN_V2_REPORT.txt      CHROMATIN_REG / GATE6 LoF: Fisher p (vs siblings) -> matchP (1,000 matched sets)
  phase1/P1_RARE_APOPTOSIS_REPORT.txt      APOP_PROCESS LoF: Fisher p -> matchP
  phase1/P1_PSEUDOTIME_REPORT.txt          M1 naive p -> M2 (+LOEUF/length/expression) p
  apoptosis/sfari_stratified.json          SFARI_ALL APOPTOSIS / OLIGO_functional p, LOEUF x CDS matched (3,000 draws; p=0 -> < 1/3000)
  apoptosis/pleiotropy_control.json        same lists, + GO-annotation-count matched null
  testB/pair_correlations.csv              IN-PV x OPC p, unadjusted
  testB/pair_correlations_techadj.csv      IN-PV x OPC p, RIN/PMI/mito/ngene/age adjusted
  phase15/p15_summary_table.tsv            PRIMARY_OL_stage p_adj -> ADJ_composition p_adj
  phase16/opc_se_matched.json              OPC CpG Fisher p -> SE-decile Mantel-Haenszel p
  phase14/p14_sign_fix.log                 corrected ICVF mean rg (-0.1020) and Gu 2025 S4 mean (-0.0733)
  rg/RG_REAL_ASD_REPORT.txt                15 ICVF tract rgs as first reported (mean +0.102), retracted for sign (E-007)
Usage: python asd_myelin_survival.py <out.png>
"""
import csv, json, math, os, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = os.path.expanduser("~/projects/asd-myelin/outputs/")
J = lambda f: json.load(open(O + f))


def report_row(path, module, cls):
    """(p, matchP) from a carrier-burden report row."""
    for line in open(O + path):
        f = line.split()
        if len(f) >= 8 and f[0] == module and f[1] == cls:
            return float(f[5]), float(f[7])
    raise KeyError((path, module, cls))


def perm_p(rows, lst, st, key="set", n_perm=None):
    r = next(x for x in rows if x["list"] == lst and x[key] == st)
    return r["p"] if r["p"] > 0 else 1.0 / n_perm


pt = open(O + "phase1/P1_PSEUDOTIME_REPORT.txt").read()
stage_naive = float(re.search(r"M1 naive.*?p=([0-9.e-]+)", pt).group(1))
stage_cond = float(re.search(r"M2 \+LOEUF.*?p=([0-9.e-]+)", pt).group(1))

chrom = report_row("phase1/P1_RARE_BURDEN_V2_REPORT.txt", "CHROMATIN_REG", "LoF")
gate6 = report_row("phase1/P1_RARE_BURDEN_V2_REPORT.txt", "GATE6_NEUROGENIC_GLIOGENIC_SWITCH", "LoF")
apop_dn = report_row("phase1/P1_RARE_APOPTOSIS_REPORT.txt", "APOP_PROCESS", "LoF")

ss, pc = J("apoptosis/sfari_stratified.json"), J("apoptosis/pleiotropy_control.json")
apop = (perm_p(ss, "SFARI_ALL", "APOPTOSIS", n_perm=3000), perm_p(pc, "SFARI_ALL", "APOPTOSIS"))
olig = (perm_p(ss, "SFARI_ALL", "OLIGO_functional", n_perm=3000), perm_p(pc, "SFARI_ALL", "OLIGO"))


def pair_p(f, a, b):
    for r in csv.DictReader(open(O + f)):
        if {r["a"], r["b"]} == {a, b}:
            return float(r["p"])
    raise KeyError((f, a, b))


pv_opc = (pair_p("testB/pair_correlations.csv", "IN-PV", "OPC"), pair_p("testB/pair_correlations_techadj.csv", "IN-PV", "OPC"))

p15 = {r["test"]: r for r in csv.DictReader(open(O + "phase15/p15_summary_table.tsv"), delimiter="\t")}
clock = (float(p15["PRIMARY_OL_stage"]["p_adj"]), float(p15["ADJ_composition"]["p_adj"]))

p16 = J("phase16/opc_se_matched.json")
opc_dmr = (p16["opc_vs_others"]["CpG"]["fisher_p"], p16["LEG_A_se_matched_CpG"]["q10"]["mantel_haenszel"]["MH_p"])

sf = open(O + "phase14/p14_sign_fix.log").read()
rg_fixed = float(re.search(r"ours corrected ICVF mean\s*=\s*([-0-9.]+)", sf).group(1))
rg_gu = float(re.search(r"Gu\s+AutoPtx WM tract mean = ([-0-9.]+)", sf).group(1))
icvf_rows = [l.split() for l in open(O + "rg/RG_REAL_ASD_REPORT.txt") if l.split()[1:2] == ["ICVF"]]
rg_first = sum(float(f[3]) for f in icvf_rows) / len(icvf_rows)  # mean over the 15 ICVF tracts as first reported
assert len(icvf_rows) == 15

# (label, p before control, p after control, outcome, control name)
SURV, DIED, OPEN = "survived", "died", "open"
arms = [
    ("Chromatin regulators, de novo LoF", chrom[0], chrom[1], SURV, "matched gene sets"),
    ("Oligodendrocyte genes in SFARI list", olig[0], olig[1], SURV, "annotation-count null"),
    ("OPC methylation excess", opc_dmr[0], opc_dmr[1], OPEN, "precision matching"),
    ("Oligodendrocyte maturation stage", stage_naive, stage_cond, DIED, "constraint/length/expression"),
    ("Neurogenic→gliogenic switch, de novo", gate6[0], gate6[1], DIED, "matched gene sets"),
    ("Apoptosis genes in SFARI list", apop[0], apop[1], DIED, "annotation-count null"),
    ("Apoptosis genes, de novo LoF", apop_dn[0], apop_dn[1], DIED, "matched gene sets"),
    ("PV interneuron × OPC co-dysregulation", pv_opc[0], pv_opc[1], DIED, "technical covariates"),
    ("Oligodendrocyte clock, autism vs control", clock[0], clock[1], DIED, "cell composition"),
]

INK, MUTED, GRID, BAR, ALT, NULL = "#1f2328", "#59636e", "#e6e8eb", "#4a6fa5", "#c9822b", "#9aa3ad"
COL = {SURV: BAR, DIED: NULL, OPEN: ALT}
CAP = 12.0  # -log10 p axis cap; the methylation arm runs past it and is labelled

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=200, gridspec_kw={"width_ratios": [1.8, 1]})
ax = axes[0]
for i, (lab, p0, p1, out, ctrl) in enumerate(arms):
    y = len(arms) - 1 - i
    x0, x1 = -math.log10(p0), -math.log10(p1)
    c = COL[out]
    ax.plot([min(x0, CAP), min(x1, CAP)], [y, y], color=c, lw=2, zorder=1)
    ax.scatter([min(x0, CAP)], [y], s=46, facecolor="white", edgecolor=c, lw=1.6, zorder=2)
    ax.scatter([min(x1, CAP)], [y], s=52, color=c, zorder=3)
    if max(x0, x1) > CAP:
        ax.annotate("", xy=(CAP, y), xytext=(CAP - 1.2, y), arrowprops=dict(arrowstyle="->", color=c, lw=1.6))
        ax.text(CAP - 1.3, y, f"off scale: p = {p0:.0e} → {p1:.0e}", ha="right", va="center", fontsize=8, color=MUTED)
    ax.text(CAP + 0.25, y, ctrl, ha="left", va="center", fontsize=8, color=MUTED)
ax.axvline(-math.log10(0.05), color=INK, lw=1, ls="--")
ax.text(-math.log10(0.05) + 0.1, len(arms) - 0.45, "p = 0.05", fontsize=8.5, color=MUTED)
ax.set_yticks(range(len(arms))); ax.set_yticklabels([a[0] for a in arms][::-1], fontsize=9)
ax.set_xlim(0, CAP); ax.set_ylim(-0.6, len(arms) - 0.2)
ax.set_xlabel("−log10 p    (hollow: before the control that decided it; filled: after)", color=MUTED, fontsize=9.5)
ax.set_title("Which arms survived their own controls", fontsize=12, color=INK, pad=10)
for out, lab in ((SURV, "survived"), (OPEN, "meaningful signal: grows under precision matching"), (DIED, "died")):
    ax.scatter([], [], color=COL[out], s=40, label=lab)
ax.legend(frameon=False, fontsize=8.3, loc="lower right", labelcolor=MUTED, bbox_to_anchor=(1.0, 0.0))

ax = axes[1]
vals = [rg_first, rg_fixed, rg_gu]
names = ["As first\nreported", "After allele-\norientation fix", "Gu et al.\n2025"]
cols = [NULL, BAR, ALT]
for k, v in enumerate(vals):
    ax.bar(k, v, width=0.6, color=cols[k])
    ax.text(k, v + (0.006 if v > 0 else -0.006), f"{v:+.3f}", ha="center", va="bottom" if v > 0 else "top", fontsize=9.5, color=INK)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(-0.15, 0.15)
ax.set_ylabel("mean genetic correlation, autism × white-matter ICVF", color=MUTED, fontsize=9.5)
ax.set_title("White matter: the sign fix", fontsize=12, color=INK, pad=10)
ax.text(1, -0.135, "corrected: 15/15 tracts negative,\nno frontal gradient", ha="center", fontsize=8, color=MUTED)

for ax in axes:
    ax.grid(axis="x" if ax is axes[0] else "y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
fig.tight_layout()
fig.savefig(sys.argv[1], facecolor="white")
for a in arms:
    print(f"{a[0]:45s} {a[1]:.3g} -> {a[2]:.3g}  {a[3]}")
print("rg first/fixed/gu", rg_first, rg_fixed, rg_gu)
