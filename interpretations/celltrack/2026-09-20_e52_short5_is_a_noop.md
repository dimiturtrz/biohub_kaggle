# E52 — SHORT5 is a no-op on the champion's output, and this proxy cannot price a prune at all

**Verdict: the champion already removes every component of ≤5 nodes — the minimum component size in its
submission is exactly 6, on all four movies. Harvested as a post-pass, SHORT5 deletes zero nodes. Its
only possible effect is ORDERING (raw geff, before postprocessing), which is E37's repair axis. Separately,
this measurement exposed that the annotated-fraction of these movies makes a cached submission unable to
price ANY prune: only 1.4 % of predicted nodes match GT, so "FP" is overwhelmingly "unannotated".**

## What was harvested

`howonkang/biohub-0947-short5-prepp-r1`, `apply_short5_raw`: treat prediction edges as undirected,
union-find, delete every connected component with ≤5 nodes plus incident edges, from the RAW geff BEFORE
the champion's postprocessing. Screened in as one of exactly three functions that kernel adds over 0947
(env-var and `def`-set difference; the sibling `pawanmali/biohub-zhincez947-nodiv-ablation-v1` added zero
and was skipped without spending a run).

## The measurement

`kaggle/short_component_prune.py` over `runs/kaggle_out/celltrack-public-0947/submission.csv`
(122 808 nodes, the four `TEST_MOVIES`), GT from `train/*.geff`, matched by per-frame KD-tree at 2.87 µm
(cell separation) and again at the official 7 µm.

Component-size histogram, every movie:

    min component size = 6      sizes 6,7,8,… populated; nothing below 6 exists

So `component ≤ 2`, `≤ 3`, `≤ 5` each remove **0 nodes**. The champion's `OUTPUT_MIN_TRACK_LEN = 6` is
already binding, and `OUTPUT_KEEP_DIVISION_COMPONENTS = 1` spares nothing in practice — no short
division component survives to be spared.

Extending past the cut for shape only: `≤ 8` removes 4 798 nodes at 0.35 % collateral, `≤ 12` removes
11 343 at 1.18 % (2.87 µm ruler). Those are *not* a SHORT5 result; they are the next rung of the same
knob, i.e. a stricter `OUTPUT_MIN_TRACK_LEN`, which the champion already swept.

## The denominator problem this surfaced

Only **1 701 of 122 808** predicted nodes (1.39 %) lie within 2.87 µm of any GT cell; 2.21 % at 7 µm.
That is not a 98 % false-positive rate — it is the **annotation sparsity** (E51's table: annotated frac
0.0016–0.135). An unmatched predicted node is usually a real cell nobody labelled.

Consequence, and it is the general point: **a cached submission cannot price a pruner's FP-recall**, because
its negative class is "unannotated", not "false". The collateral column stays valid — it is computed over
matched-GT nodes, a denominator that means what it says — but the recall column is uninterpretable. E47's
spec needs both, so a prune has to be priced where the FP set is known (a candidate set with GT
correspondence), the way E47/E48/E49 were, not on a submission.

## What this closes and what it does not

**Closes:** SHORT5 as a post-hoc element. There is nothing for it to delete. No submission slot is owed it.

**Does not close:** SHORT5 applied *pre*-postprocessing, where it removes short components before the
champion's bridging/repair can extend them. That is a different claim — it says the repair step is
net-harmful on short components — and E37 already priced the whole repair axis at a ~1.1 % ceiling on 0947
with only 935 bridgeable end→start pairs. A pre-pass could only redistribute inside that ceiling, so it is
sub-bar by E37's own number, and testing it needs a full kernel rerun. Not worth a slot.
