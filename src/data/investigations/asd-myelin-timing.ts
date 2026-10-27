import type { InvestigationEntry } from "./types";

// Every number here is traced to a file in ~/projects/asd-myelin in docs/stage1/asd-myelin-timing_FACTS.md.
export const ENTRIES: InvestigationEntry[] = [
  {
    n: 1,
    date: "2026-09-16",
    q: "Does autism share genetic architecture with frontal white matter?",
    verdict: "ruled-out",
    tag: "Genetic correlation · later voided",
    tried:
      "A genetic correlation (rg: how far the common variants that raise one trait also raise another, estimated from GWAS summary statistics by LD-score regression) between an autism GWAS and fractional anisotropy (FA) in 10 UK Biobank white-matter tracts from the Oxford BIG40 release. Before running it I wrote down what each outcome would mean: a frontal-specific negative rg would support the hypothesis, a null would argue against a common-variant cause.",
    found:
      "All ten correlations sat near zero, with the frontal and long-range tracts closest to zero. The same run reported an autism heritability of 0.348.",
    ruled:
      "Nothing, in the end. Both the trait (entry 4) and, for the follow-up, the instrument (entry 3) turned out to be wrong.",
    why:
      "FA is a weak proxy for myelin. And 0.348 is far above the heritability I later measured on the real autism GWAS (0.122). That mismatch was the first sign of the file problem, and I did not read it.",
    next:
      "Not close myelin on one weak measure: rerun on a myelin-sensitive diffusion measure, and look directly at lipid and myelin genes.",
  },
  {
    n: 2,
    date: "2026-09-16",
    q: "Do lipid and myelin genes carry autism risk at their own loci?",
    verdict: "ruled-out",
    tag: "Locus scan + fine-mapping · later voided",
    tried:
      "Take the strongest association within 100 kb of each of 45 cholesterol-synthesis, lipid-transport and myelin genes, then fine-map the hits (SuSiE: a Bayesian method that gives each variant a probability of being the causal one).",
    found:
      "Three loci were genome-wide significant: LRP1 (p = 1.2e-18), SREBF2 (p = 2.9e-10) and MOG (p = 1.9e-10). The APOE ε4 SNP was null (p = 0.31). Fine-mapping dissolved all three: the LRP1 signal belongs to STAT6 next door, MOG sits inside the immune MHC region, and the SREBF2 signal had no attributable variant.",
    ruled: "That the lipid loci were lipid signals.",
    why:
      "STAT6 and the MHC are classic allergy loci. That was the real clue, and I misread it for a week: the 'autism' file was an allergic-disease GWAS (entry 4).",
    next:
      "Before chasing methylation of these genes, rebuild the white-matter test on a better measure.",
  },
  {
    n: 3,
    date: "2026-09-16",
    q: "Does autism track neurite density in frontal tracts?",
    verdict: "ruled-out",
    tag: "Pipeline error",
    tried:
      "Redo the white-matter test on ICVF (intracellular volume fraction: a NODDI diffusion measure of neurite density that follows myelinated-axon content more closely than FA) in frontal tracts.",
    found:
      "Before any result: the download script had mapped BIG40 file numbers to the catalogue's row index instead of its phenotype column. The eight 'ICVF' files were tensor eigenvalues (L2/L3) of the internal capsule and cingulum, not ICVF at all, and the sample size was wrong (33,224 instead of 31,355). Several downloads had also silently truncated.",
    ruled: "No conclusion had been drawn. The test was rebuilt on 15 verified ICVF tracts.",
    why:
      "The mislabelled inputs would have produced a confident 'myelin' result about a measure the test never touched. Only the failed downloads stopped it.",
    next: "Check every identifier against the catalogue and rerun.",
  },
  {
    n: 4,
    date: "2026-09-16",
    q: "Was the autism GWAS actually autism?",
    verdict: "ruled-out",
    tag: "Input error",
    tried: "A positive control: do the known autism loci show up in the file?",
    found:
      "They did not. The lead autism SNP rs910805 had p = 0.082 instead of the published 2e-09, and the top loci were TSLP, IL33, IL1RL1 and ORMDL3. The file was byte-identical to the Ferreira 2017 allergic-disease GWAS (N = 242,569), renamed when it was downloaded. In the real iPSYCH-PGC autism GWAS (Grove 2019), rs910805 matches exactly (p = 2.041e-09) and liability heritability is 0.122.",
    ruled:
      "Every genetic result from 10 to 14 September: the FA null, the 0.348 heritability, the three lipid loci and their fine-mapping.",
    why:
      "The fine-mapped STAT6 and MHC signals were allergy biology. Checking known loci before anything else would have caught it on day one.",
    next:
      "Rerun everything on the real GWAS, and keep the allergy GWAS as a deliberate comparison trait.",
  },
  {
    n: 5,
    date: "2026-09-16",
    q: "On the real GWAS, do APOE or the lipid loci matter?",
    verdict: "ruled-out",
    tag: "Candidate loci + eQTL",
    tried:
      "The same scan with an empirical regional p-value (the locus minimum compared against random regions), plus a lookup of whether each lead SNP changes its gene's expression in adult prefrontal cortex (eQTL, PsychENCODE) or its methylation in fetal brain (mQTL).",
    found:
      "Autism positive controls were strong (XRN2/KIZ p = 2.0e-09, MACROD2 p = 2.8e-08). APOE ε4 p = 0.21. LRP1 and SREBF2 fell to p = 1.2e-3 and 4.3e-4 (empirical 0.16 and 0.11). DHCR7 and FDFT1 were nominal (empirical 0.018 and 0.021) and did not survive correction. At these loci the autism lead SNP was not itself an eQTL (SREBF2: p = 0.38), and there were no fetal mQTL hits.",
    ruled:
      "APOE as a route (it had been set outside the scope from the start) and the lipid loci, which were allergy artefacts.",
    why: "With the right trait, cholesterol and myelin genes look like the rest of the genome.",
    next: "Single loci are low-powered. Test the gene sets as wholes.",
  },
  {
    n: 6,
    date: "2026-09-16",
    q: "Do lipid, cholesterol or oligodendrocyte gene sets carry common-variant risk?",
    verdict: "ruled-out",
    tag: "Gene-set analysis (MAGMA)",
    tried:
      "MAGMA (combines SNP associations into a score per gene, then tests whether a gene set scores higher than other genes, correcting for gene size and SNP density) on 19 sets, compared against 160 random sets of matched size.",
    found:
      "The SFARI autism gene list worked as a positive control (p = 1.2e-05). Every lipid set leaned the wrong way: lipid transport β = −0.037 (p = 0.81), cholesterol synthesis p = 0.69, a curated myelin core β = −0.22 (p = 0.92). OPC and oligodendrocyte marker sets reached p = 0.078 and 0.079, weaker than the best of the 160 random sets (p = 0.0036).",
    ruled: "Lipid and cholesterol genes as a common-variant entry point.",
    why:
      "I first read a smooth decline (OPC > oligodendrocyte > myelin) as support for a timing story; the next day it turned out to track gene constraint, not lineage. A planned replication on a second autism GWAS (Matoba 2020) failed silently: its log ends in a missing-file error that the wrapper script did not catch, so that replication never ran.",
    next: "Ask whether partitioned heritability can see the oligodendrocyte lineage at all.",
  },
  {
    n: 7,
    date: "2026-09-16",
    q: "Can partitioned heritability resolve the oligodendrocyte lineage?",
    verdict: "partial",
    tag: "S-LDSC · uninformative",
    tried:
      "Stratified LD-score regression (S-LDSC: estimates how much heritability sits in SNPs near a gene set, over and above a baseline model), with LD scores built locally for 22 annotations.",
    found:
      "The SFARI set validated the pipeline (enrichment 1.61, coefficient z = 2.94). Excitatory neurons, where autism signal is known to sit, reached only z = 0.86. Oligodendrocytes (0.37), OPCs (0.29), lipid transport (0.17) and myelin (−0.44) were null. Six annotations, including chromatin regulators, were never built.",
    ruled: "Nothing. These nulls carry no information.",
    why:
      "A test that cannot detect the neuronal signal cannot rule out an oligodendrocyte one. A larger autism GWAS is the reopen trigger.",
    next: "Use a design with more resolution: place genes on the oligodendrocyte maturation axis.",
  },
  {
    n: 8,
    date: "2026-09-16",
    q: "With the right GWAS and the right measure, does autism track white matter?",
    verdict: "partial",
    tag: "Genetic correlation (BIG40 ICVF)",
    tried: "rg between real autism liability and 27 BIG40 measures: ICVF in 15 tracts, FA in 10, L2 in 2.",
    found:
      "As first reported: mean ICVF rg +0.102, positive in all 15 tracts (strongest, left cingulum, p = 0.007; none survived Bonferroni). Anterior and posterior tracts differed by 0.011: no frontal gradient.",
    ruled: "Frontal specificity, the anatomical core of the starting hypothesis.",
    why:
      "The sign was wrong (entry 18): every correlation was inverted by an allele-orientation error. The absence of a frontal gradient survived the fix.",
    next:
      "I rewrote the question that day: is there a subset of autism with an abnormal oligodendrocyte developmental trajectory, with timing as a first-class variable? Premature maturation that exhausts the progenitor pool could later look like hypomyelination.",
  },
  {
    n: 9,
    date: "2026-09-17",
    q: "Does autism risk concentrate at a stage of oligodendrocyte maturation?",
    verdict: "ruled-out",
    tag: "Maturation axis, atlas 1",
    tried:
      "In a human infant prefrontal atlas (Klavert et al. 2026; 3,021 oligodendrocyte-lineage cells), give each of 13,644 genes a stage score from OPC through committed precursor (COP) to oligodendrocyte, then regress each gene's autism association (MAGMA Z) on it. Pre-registered shape: an inverted U peaking near COP. Controls: gene constraint (LOEUF, how intolerant a gene is to loss of function), coding length and expression breadth; 1,000 stage-label permutations; the same model on neuron, astrocyte and microglia axes.",
    found:
      "Naive association p = 0.004. Adding constraint, length and expression: p = 0.64; permutation p = 0.70. The quadratic term had the wrong sign. The neuron axis, run identically, gave t = +3.36, p = 0.0008.",
    ruled: "Common-variant risk localised to an oligodendrocyte maturation stage.",
    why:
      "Long, constrained, broadly expressed genes are associated with nearly every brain trait, and they are the genes progenitors express. The 'stage' effect was a property of the genes. The neuron result shows the test can see a real cell-type signal.",
    next: "Rare variants have larger effects. Test those, then replicate in an atlas with more mature cells.",
  },
  {
    n: 10,
    date: "2026-09-17",
    q: "Do rare de novo mutations hit oligodendrocyte or lipid genes?",
    verdict: "kept",
    tag: "De novo burden",
    tried:
      "Compare autistic children with their unaffected siblings in the Simons Simplex Collection (2,140 vs 1,614) for carrying a new loss-of-function (LoF) mutation in each gene module. Each module gets a synonymous control: mutations that leave the protein unchanged cannot be causal, so they should not be enriched.",
    found:
      "The first version counted mutations rather than people, and synonymous mutations came out enriched, which is impossible; I retracted it. Counting carriers: SFARI genes LoF 231 vs 44 (OR 3.96); chromatin regulators LoF 128 vs 40 (OR 2.41, q = 3e-06, matched-set p = 0.011) with synonymous OR 1.06 and missense 1.07. Committed-precursor genes: 8 vs 2 carriers (q = 0.77). Cholesterol sets: missense OR 0.72–0.75.",
    ruled: "Lipid genes as a rare-variant route. The oligodendrocyte modules are underpowered, not ruled out.",
    why:
      "Chromatin LoF burden is the established autism result (Satterstrom 2020): loss-of-function-specific and not oligodendrocyte-specific. The synonymous control is what caught the counting error.",
    next: "One module looked enormous: the switch from making neurons to making glia.",
  },
  {
    n: 11,
    date: "2026-09-17",
    q: "Is the neurogenic-to-gliogenic switch a rare-variant hotspot?",
    verdict: "ruled-out",
    tag: "Synonymous control",
    tried:
      "The switch module's LoF burden, against 1,000 gene sets matched on coding length and constraint, and against its own synonymous row.",
    found:
      "LoF 23 vs 1 carrier, OR 17.35, q = 5e-04: the largest effect in the project. Against matched sets it was ordinary (p = 0.16), and its synonymous row was OR 6.03. OPC marker genes had the same defect (LoF OR 6.03, synonymous OR 2.71, matched p = 0.030).",
    ruled: "The switch module and the OPC-marker burden.",
    why:
      "A control that cannot be real came out positive, so the module was measuring gene length and constraint, not biology.",
    next: "Back to common variants, with modules large enough to detect something.",
  },
  {
    n: 12,
    date: "2026-09-17",
    q: "Does any mechanistic gate carry common-variant risk at adequate power?",
    verdict: "ruled-out",
    tag: "Gate competition",
    tried:
      "Regress gene-level autism Z jointly on 26 mechanistic modules (cholesterol synthesis and trafficking, sphingolipids, peroxisome, oligodendrocyte maturation, mitochondria, iron, Wnt, ERK, mTOR, immune niche, chromatin), with constraint, length, expression and SNP-count covariates and 1,000 matched sets.",
    found:
      "Version 1 used modules of 11–30 genes, far too small to detect an effect the size of SFARI's (β = +0.13); I withdrew it as uninformative. Rebuilt with 160 or more genes per module: SFARI β = +0.13 (p = 0.001); cholesterol synthesis p = 0.45, trafficking p = 0.65, oligodendrocyte maturation p = 0.48, chromatin p = 0.74; peroxisome/plasmalogen depleted (β = −0.17, p = 0.045); mitochondria p = 0.063. Four nested contrasts returned no value because their modules fell below the size floor.",
    ruled:
      "Lipid synthesis and trafficking as common-variant gates. Conditioning on oligodendrocyte genes moved them by about 1%, so they are absent in their own right, not hidden behind the lineage.",
    why:
      "Chromatin is null for common variants but strong for rare ones (entry 10): a real common/rare split. Fate-switch and OPC-timing modules are too small to test at this level.",
    next: "Turn to the immune version of the predisposition, starting with whether the cells can hear cytokines.",
  },
  {
    n: 13,
    date: "2026-09-17",
    q: "Can inflammatory cytokines act directly on OPCs?",
    verdict: "ruled-out",
    tag: "Receptor substrate",
    tried:
      "If inherited immune reactivity damages myelin through cytokines, OPCs need the receptors. Mean expression (log-CPM) of 25 immune receptors per cell type in the infant prefrontal atlas, with microglia as the positive control.",
    found:
      "IL6R 0.015 on OPCs vs 0.811 on microglia; IL1R1 near zero in every cell type (0.005); TNFRSF1A 0.033 vs 0.215; complement receptors absent from OPCs. 15 of 25 receptors present on OPCs vs 20 of 25 on microglia. Two exceptions: the type-I interferon receptor IFNAR1 at parity (0.174 vs 0.194), and the adenosine receptor ADORA1 higher on OPCs (0.137 vs 0.015).",
    ruled: "Classical inflammatory cytokines acting directly on OPCs at baseline.",
    why:
      "The classical receptors face microglia, so any effect on this lineage would have to be relayed. Receptor induction during inflammation was not tested.",
    next: "Ask whether the immune predisposition is in the genome at all.",
  },
  {
    n: 14,
    date: "2026-09-17",
    q: "Is there an inherited immune route into the oligodendrocyte lineage?",
    verdict: "ruled-out",
    tag: "Immune genetics",
    tried:
      "Three tests: immune submodules (MHC, interferon, TNF, IL-6/IL-17, TLR, complement, microglia, mast cells) in the common-variant regression; a symmetric cell-type test running the autism and allergy GWAS through the same atlas; and de novo burden in immune modules.",
    found:
      "No immune submodule carried positive autism signal (IL-6/IL-17 p = 0.50; TLR depleted, p = 0.043). Autism localised to neurons (t = +3.08); allergy to microglia (t = +6.30) and away from neurons (t = −4.77). Neither localised to OPCs, and allergy was depleted there (t = −2.08). rg(autism, allergy) = +0.094 (p = 0.011). Rare: TNF/NF-κB LoF OR 3.47 (q = 0.074), with its synonymous row also raised (OR 1.51).",
    ruled: "A child-genome immune route into the oligodendrocyte lineage.",
    why:
      "The double dissociation shows the instrument resolves cell types. Maternal genotype and prenatal exposure are invisible to a child's GWAS, so those versions stay open. The planned allergy–autism overlap tests (local correlation, colocalisation) were written but never run.",
    next: "Quantify what a myelin change would actually do to signal timing.",
  },
  {
    n: 15,
    date: "2026-09-17",
    q: "How much timing does myelin buy?",
    verdict: "kept",
    tag: "Conduction model",
    tried:
      "Simulate 20,000 axons per scenario with realistic diameter and myelin-thickness (g-ratio) distributions and compute σ_d, the spread of arrival times, on tracts of 20 to 300 mm. Compare it with the windows that matter: about 3 ms for feed-forward inhibition, 8 ms for coincidence detection, 25 ms for a gamma cycle.",
    found:
      "Moderate hypomyelination adds 5.3 ms of spread on a 90 mm association tract and 1.0 ms on a 20 mm fibre. Hypermyelination removes spread (−2.8 ms, ×0.86). Hypermyelinated axons with more variable sheaths return almost to baseline (×1.05).",
    ruled:
      "My own pre-written expectation that under- and over-myelination converge on more spread. The model said the opposite.",
    why:
      "Different autism genes push myelination in opposite directions in animal models (PTEN early; CHD8, NLGN3, TCF4 under-myelinated). A mixed cohort cancels in the mean and in the variance, which predicts weak, inconsistent whole-cohort myelin findings and makes stratification a precondition. The parameters are typical but uncited, so this is a scaling argument; the physics is Pajevic and colleagues'.",
    next: "Replicate the maturation-axis null in a second atlas with mature cells.",
  },
  {
    n: 16,
    date: "2026-09-17",
    q: "Does the stage null hold in an atlas that reaches mature cells?",
    verdict: "ruled-out",
    tag: "Maturation axis, atlas 2",
    tried:
      "Repeat the stage test in GSE155488 (10,391 oligodendrocyte-lineage cells from fetal, 2-year, 13-year and adult donors, staged OPC → COP → newly formed → myelin-forming → mature), with four maturation scores and 1,000 permutations. Then trace each gene programme along the stages.",
    found:
      "All scores null, naive and conditioned (pseudotime correlation p = 0.86 and 0.53; permutation p = 0.81); even the naive effect was absent. Along the lineage, cholesterol, sphingolipid and plasmalogen synthesis all peak at COP, while structural myelin genes are the only programme rising steadily to maturity.",
    ruled: "Stage localisation of common-variant risk, now in two atlases.",
    why:
      "Lipid synthesis looks like an output switched on at commitment, so a supply failure would show as a failure to commit, not a thin sheath at the end. That reading is descriptive: two scripts staged this atlas differently and were never reconciled, and the mouse version of the offset is known (Marques 2016).",
    next: "Genetics shows where risk starts. Methylation can show which cells end up perturbed.",
  },
  {
    n: 17,
    date: "2026-09-17",
    q: "Which cell type is most epigenetically perturbed in autistic cortex?",
    verdict: "partial",
    tag: "Cell-type methylation",
    tried:
      "Single-nucleus methylation of autistic and control prefrontal cortex (GSE298243, Eyring et al. 2026; 8 cell types, 2.5 million tested regions): the rate of diagnosis-associated differentially methylated regions (DMRs) per cell type.",
    found:
      "OPCs had the highest rate: 272 of 12,515 CpG regions (2.2%) vs 1.1% across the other seven types, OR 1.92, p = 1.5e-19.",
    ruled: "Nothing yet. I promoted this to the program's keystone.",
    why:
      "A cell type that is not where inherited risk lives but is the most perturbed is what a downstream convergence point would look like. OPC estimates are also the noisiest in the dataset, which needed checking.",
    next: "Check the signal's content and its noise before leaning on it (entry 19).",
  },
  {
    n: 18,
    date: "2026-09-21",
    q: "Why does my white-matter correlation disagree with Gu et al. 2025?",
    verdict: "partial",
    tag: "Allele orientation",
    tried:
      "Reconcile my positive correlation with Gu et al. (2025, Molecular Psychiatry), who reported a negative one on the same UK Biobank data.",
    found:
      "BIG40 files declare a1 as the reference allele; my preprocessing had used it as the effect allele. Re-prepped, the flip was exact: ICVF mean rg +0.102 → −0.102 (15/15 negative), FA +0.060 → −0.060, and the loading on a GenomicSEM common white-matter factor +0.114 → −0.113 (p = 0.022). Best FDR across tracts 0.064. On Gu's tract definitions my estimates are indistinguishable from theirs (all p ≥ 0.33); Gu report 14 of 15 tracts negative, mean −0.073.",
    ruled: "The positive result. There never was one.",
    why:
      "The corrected result is a weak, global, negative correlation: higher autism liability with lower neurite density, everywhere rather than frontally. That replicates Gu 2025 with less power; it is not a new finding. The same bug sits in a bidirectional Mendelian randomisation run (genu ICVF → autism, nominal p = 0.026, failing correction; the reverse direction had only two instruments), which has not been redone with the fix.",
    next: "Apply the same scrutiny to the methylation keystone.",
  },
  {
    n: 19,
    date: "2026-09-21",
    q: "Is the OPC methylation excess real?",
    verdict: "kept",
    tag: "Methylation, adjudicated",
    tried:
      "Three alternative explanations. Content: which genes the significant OPC regions sit in. Noise: OPC estimates have the largest standard errors, which inflates threshold-crossing, so compare each region only with regions of equal precision (10 standard-error deciles, Mantel–Haenszel). Composition: whether the cell types were tested on comparable regions.",
    found:
      "Content first: 3 of 272 significant regions carry a gene name (two olfactory pseudogenes and a lncRNA), and no module was testable, so I downgraded the result to noise. The matched test reversed that: the excess grows under precision matching, OR 1.92 → 2.86 (95% CI 2.50–3.29, p = 4e-56), and rises with stringency (1.25 at p < 0.05, 2.45 at p < 1e-5). Three named regions is what the 2.2% annotation rate predicts (6.1 expected, p = 0.30), and mapping by coordinates gives 205 genes with no pathway enrichment. But each cell type was tested on its own regions: of 124,166 CpG regions, 2,103 are shared by any two types and none by all eight.",
    ruled: "Measurement noise and annotation emptiness as explanations. The excess is a real signal.",
    why:
      "Noise would shrink under precision matching and fade with stringency; this excess grows under both. That makes it the strongest perturbation signal in the program. What remains open is specificity: each cell type was tested on its own regions, so ranking OPCs above the other seven needs a shared region set, refit from the per-donor files with diagnosis permuted.",
    next: "Refit on a shared region set from the per-donor files (pending); meanwhile, look for the maturation trajectory directly in autistic tissue.",
  },
  {
    n: 20,
    date: "2026-09-21",
    q: "Is oligodendrocyte maturation shifted in autistic cortex?",
    verdict: "partial",
    tag: "Maturation clock",
    tried:
      "Turn the atlas-2 stage markers into a clock that places any oligodendrocyte-lineage nucleus between OPC and mature, and apply it to 25,494 such nuclei from autistic and control cortex (Velmeshev et al. 2019; 15 autistic and 16 control donors, ages 4–22). Prespecified controls: the same score on astrocytes and excitatory neurons, ambient RNA, cell composition.",
    found:
      "The clock transfers: OPC and oligodendrocyte clusters separate blind (d = 5.4), and in control donors stage rises with age (ρ = +0.77, p = 5e-04). The primary comparison is null: p = 0.16 (permutation 0.17), the astrocyte control shifts more (d = −0.72) than the primary (d = −0.41), and adjusting for composition gives p = 0.72. Post hoc, an age-by-diagnosis interaction within oligodendrocytes (permutation p = 0.011) survived leave-one-out, but the excitatory-neuron control also reached p = 0.021.",
    ruled: "A group-level stage shift detectable at this sample size.",
    why:
      "The score partly reads OPC fraction, nuclei recovery and ambient RNA. The smallest effect detectable at 80% power is d = 1.04 (0.43 stage units), about the clock's entire change between ages 2 and 13 (0.44); d = 0.5 would need about 64 donors per group. An age interaction is the shape the trajectory idea predicts, so it stays a lead, not a finding.",
    next: "A new hypothesis: failed apoptosis and hypomyelination share genes.",
  },
  {
    n: 21,
    date: "2026-10-07",
    q: "Do failed-apoptosis and oligodendrocyte genes converge in autism?",
    verdict: "ruled-out",
    tag: "Apoptosis × oligodendrocyte",
    tried:
      "The idea: failed apoptosis leaves prefrontal cortex hyperconnected, hypomyelination causes excitation–inhibition (E/I) imbalance, and the two share genes. Tested with MAGMA on the 33-gene intersection of the apoptosis and oligodendrocyte GO sets, and with de novo burden.",
    found:
      "Common variants: the intersection had β = −0.006, p = 0.51, below two of its five size-matched random sets; the oligodendrocyte signal sat in the non-apoptosis genes (p = 0.037). De novo: apoptosis LoF OR 1.94 (p = 0.003), but against matched sets its synonymous row was more extreme than its LoF row (p = 0.009 vs 0.106). Random 31-gene control sets also reached matched p = 0.001.",
    ruled: "Apoptosis–oligodendrocyte convergence in common and de novo variants.",
    why:
      "The last line is a pipeline finding: the matched-set null is miscalibrated for small modules, so every small-module matched p in the burden pipeline, including entry 11's, is untrustworthy.",
    next: "Try the highest-confidence autism genes against a matched background.",
  },
  {
    n: 22,
    date: "2026-10-07",
    q: "Do high-confidence autism genes favour apoptosis or oligodendrocyte annotations?",
    verdict: "partial",
    tag: "Annotation-count control",
    tried:
      "Enrichment among the 125 gold-standard genes (Satterstrom 2020 ∪ Fu 2022) and the SFARI list, against genes matched on constraint and coding length; then a second null that also matches how many GO annotations each gene has.",
    found:
      "Gold genes: 4 apoptosis-and-oligodendrocyte genes vs 1.1 expected (p = 0.019), but dropping any one of CTNNB1, NF1, PTEN or SCN2A loses it; it is the mTOR-pathway disorders rediscovered (3 vs 0.6, p = 0.014). The chromatin positive control first returned zero hits because of a wrong GO term name; fixed, p = 0.0004. In the SFARI list, apoptosis looked robust (1.29×, p < 0.001) until annotation count was matched: 1.05×, p = 0.23. Oligodendrocyte genes held: 1.62× → 1.39× (p = 0.020), and 2.38× (p = 0.0045) in the highest-confidence SFARI tier.",
    ruled: "Apoptosis genes as a source of autism risk: heavily studied genes simply carry more annotations.",
    why:
      "Constraint and length matching does not remove annotation bias, so GO enrichments in autism need this control. The oligodendrocyte arm is the one that survived it.",
    next: "If the apoptotic machinery itself is mutated in autism, the executioner caspases should show it.",
  },
  {
    n: 23,
    date: "2026-10-07",
    q: "Is the pruning machinery itself a locus of risk?",
    verdict: "ruled-out",
    tag: "Caspases + developmental timing",
    tried:
      "Synaptic pruning uses the apoptotic caspase machinery without killing the cell (Ertürk et al. 2014), so test the machinery precisely: executioner caspases, IAP brakes, complement. Then ask when the surviving oligodendrocyte genes are expressed (BrainSpan, 524 samples).",
    found:
      "Zero executioner caspases among 1,231 SFARI genes (0.4 expected); brakes and complement null (p = 0.61 and 0.38). Oligodendrocyte genes on the SFARI list rise after birth (log2 +0.37 postnatal vs prenatal, peaking in early adulthood), while SFARI genes overall do not (−0.11).",
    ruled: "Pruning and apoptosis genes as loci of risk. Pruning deficits can still be a downstream phenotype.",
    why:
      "The surviving oligodendrocyte arm is postnatal and reads as a modifier. Bulk expression of these genes also tracks oligodendrocyte abundance, which rises after birth by itself.",
    next: "Stop starting from hypotheses: scan all 5,768 GO terms and see what autism genes converge on.",
  },
  {
    n: 24,
    date: "2026-10-07",
    q: "Unprompted, what do autism genes converge on?",
    verdict: "partial",
    tag: "Unbiased GO scan",
    tried: "Every GO term against the SFARI list and its high-confidence tier, under the annotation-count-matched null, with FDR across 5,768 terms.",
    found:
      "SFARI list: 190 terms at q < 0.05, overwhelmingly synaptic. High-confidence tier: 62 terms. Ensheathment of neurons ranked 6th (3.03×, q < 0.001), through an mTOR/RAS group (MTOR, PTEN, TSC1, NF1, HRAS) and a node-of-Ranvier group (SCN2A, SCN8A, ANK2); the β-catenin/TCF complex ranked 59th (9.6×, q = 0.047). Apoptosis terms were near misses (q ≈ 0.14); pruning and autophagy ranked 508th or lower.",
    ruled: "Pruning as a convergence point.",
    why:
      "Myelination appeared without being asked for, but only in the high-confidence tier: across the full SFARI list, ensheathment ranks 1,167th (1.22×, q = 0.89). Quoting '6th of 5,768' alone would overstate it.",
    next: "Test the β-catenin/TCF signal properly.",
  },
  {
    n: 25,
    date: "2026-10-08",
    q: "Is β-catenin/TCF a real oligodendrocyte-facing hub?",
    verdict: "partial",
    tag: "β-catenin/TCF",
    tried:
      "Test the nuclear complex against canonical Wnt; leave each gene out in turn; test direction (are autism genes over-represented among brakes on oligodendrocyte differentiation relative to promoters?); then check where the genes are expressed, using control-donor pseudobulk from Velmeshev 2019 and fetal OPCs from GSE155488.",
    found:
      "β-catenin/TCF: 5 genes vs 0.5 expected in the high-confidence tier and 4 vs 0.4 in the gold set, surviving removal of any single gene; canonical Wnt as a whole is null (p = 0.28). Direction: 5 of 14 brakes vs 3 of 28 promoters (p = 0.017), but it dies on every single-gene removal. Expression: CTNNB1 and PTEN are slightly depleted in the oligodendrocyte lineage (log2 −0.19, −0.26) and SCN2A strongly (−1.15); TCF7L2 is enriched (+1.28) and TCF4 is expressed in 76% of fetal OPCs.",
    ruled: "β-catenin/TCF as an oligodendrocyte-expressed hub.",
    why:
      "It is the only result in the program that survives leave-one-out, but it reflects pathway membership, not oligodendrocyte expression. (HGNC TCF4 is the Pitt–Hopkins gene, not Wnt's 'TCF-4', which is TCF7L2.) Oligodendrocyte-enriched autism genes outside the framework: VEZF1 (+1.93), TEK/TIE2 (+1.37), GNAI1 (+1.29).",
    next: "Look for a dimensional imaging signal, and for whether interneuron myelination has been tested.",
  },
  {
    n: 26,
    date: "2026-10-08",
    q: "Is there a dimensional white-matter signal in children?",
    verdict: "partial",
    tag: "HBN-POD2 pilot + literature",
    tried:
      "HBN-POD2 diffusion data are open but diagnosis is gated, so test the dimensional form on the 168 children who also have open psychopathology factor scores: axonal water fraction (AWF, a kurtosis-diffusion measure of intra-axonal water) against the general p-factor, adjusted for age and sex. Separately, check whether anyone has measured E/I after altering myelin on fast-spiking interneurons.",
    found:
      "AWF vs p-factor r = −0.152 (p = 0.049); attention, internalising and externalising null; FA, MK and MD null (all |r| ≤ 0.101). The experiment exists: Benamer et al. (2020) disrupted GABA signalling to OPCs and found abnormal myelination of fast-spiking interneurons and an E/I imbalance, with total myelin unchanged.",
    ruled: "My claim that this measurement was missing. The model now predicts altered myelin organisation, not less myelin.",
    why:
      "The p-factor is general psychopathology, not autism, and AWF is axonal rather than myelin-specific, so this supports only the form of a dimensional claim.",
    next: "A pre-registered test of the interneuron–oligodendrocyte link in human autistic tissue.",
  },
  {
    n: 27,
    date: "2026-10-08",
    q: "Are PV interneurons and OPCs co-dysregulated in autistic brains?",
    verdict: "ruled-out",
    tag: "Pre-registered test",
    tried:
      "Pseudobulk the full Velmeshev matrix (104,559 nuclei) into 525 donor × cell-type profiles, score each donor's dysregulation per cell type, and correlate PV-interneuron with OPC and oligodendrocyte dysregulation across autistic donors. Pre-registered: support needs a positive correlation that also ranks high among all 119 cell-type pairs; a value typical of all pairs means a global factor, i.e. null.",
    found:
      "Unadjusted, PV × OPC r = +0.64 (p = 0.014), 58th of 119 pairs: the median. After adjusting for RNA quality (RIN), post-mortem interval, mitochondrial fraction, genes detected and age: r = +0.52 (p = 0.055), 36th of 119; VIP × OPC was stronger (+0.57); PV × oligodendrocytes −0.06. The median pair correlation fell from 0.63 to 0.28.",
    ruled: "PV × oligodendrocyte-lineage co-dysregulation as a human signature.",
    why:
      "More than half of the apparent cross-cell-type coupling was technical. The instrument is weak: 14 donors per pair, and transcription cannot see myelin organisation.",
    next: "Return to the original question once more: lipid transport, in genetics and in methylation.",
  },
  {
    n: 28,
    date: "2026-10-08",
    q: "Last look: is lipid transport involved at all?",
    verdict: "ruled-out",
    tag: "Lipid transport, final",
    tried: "Every genetic instrument in the program, plus lipid-gene methylation by cell type in the GSE298243 data.",
    found:
      "MAGMA: lipid transport p = 0.84, cholesterol transport 0.20, cholesterol synthesis 0.85, APOE cluster 0.89, against oligodendrocyte markers at p = 0.049 in the same run. The best lipid-transport GO term on the SFARI list was fatty-acid transport, 307th (q = 0.17). Methylation: in OPCs 2 of 260 lipid-synthesis regions significant (OR 0.33, p = 0.13); in oligodendrocytes 6 of 254 (OR 1.72, p = 0.18; 2 hyper, 4 hypo); none of 26 tests below q = 0.1; transporter sets too small to test.",
    ruled:
      "Lipid-transport genetics, across MAGMA, both GO scans and the gold-standard genes. The methylation result is an underpowered null, not a refutation.",
    why:
      "With entry 16, the consistent reading is that the lipid machinery is not broken; it is gated by differentiation state and switched on at commitment.",
    next: "What remains open, and the data it needs, is below.",
  },
];
