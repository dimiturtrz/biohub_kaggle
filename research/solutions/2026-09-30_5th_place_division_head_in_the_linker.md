# 5th place — a division head *inside* the linker, priced at +0.024 private

Source: topic **#744549** (5th of 4,020, gold).

**This is the measured price of the survivor `E61` named.** Our division work ended with one mechanism still
standing: *"re-parenting inside the LINKER"* — every other division lever we tried was post-processing on a
graph that had already been committed. 5th put the division head in the linker and it is their single
largest stage.

## 1. The stage-by-stage ledger

| stage | public | private |
|---|---|---|
| **linker new-cell + division heads** | +0.020 | **+0.024** |
| **detection confidence as the cell cost in the ILP** | +0.008 | **+0.020** |
| **ZebraHub pretraining (of the *linker*)** | +0.014 | **+0.016** |
| 5-fold ensemble | +0.008 | +0.007 |
| ILP2 + "no new divisions" | +0.003 | +0.007 |
| 8-view TTA | +0.007 | +0.005 |
| **smoothing** (a line through t−4…t+4, 80% line + 20% original, then fill single-frame gaps) | +0.004 | +0.005 |
| repairs | +0.001 | +0.002 |

Two things to read off this table before anything else:

1. **The two biggest private gains are both larger on private than on public** (+0.024 vs +0.020, +0.020 vs
   +0.008) — the opposite of the near-universal inversion every other write-up reports. Mechanism-driven
   stages transferred; tuned knobs inverted.
2. **Detection confidence as the ILP cell cost is +0.020 private for what is essentially plumbing** —
   passing a number the detector already computed into the objective instead of treating every candidate
   node as equally real.

## 2. The linker, and its four heads

A transformer over two frames: **self-attention within each frame, cross-attention between t and t+1**. It
learns four things at once:

1. a **link score** for each candidate pair,
2. a **16-dim appearance/identity similarity** embedding,
3. a **new-cell** head (does this node start a track),
4. a **division** head, **with a higher loss weight on division examples**.

The division decision is therefore made *in the same forward pass and the same context* as the link decision
— the parent's competing children, the neighbours that would be stranded, and the appearance similarity are
all visible to it. Every division mechanism we tested saw a graph the ILP had already fixed.

Contrast with 14th place, which reaches the same place from the other direction: their ILP is priced so it
**cannot** fork (`edge = −1.0`, `division = 1.2`), so all divisions come from downstream geometric proposers
plus learned vetoes. 5th moves the decision *upstream* instead of adding proposers *downstream*. Both are
"the division model", and neither is a post-processing knob — which is exactly the axis our own sweeps could
not reach.

## 3. The staged ILP — a division gate built out of three solves

1. **ILP1** at `p ≥ 0.9` — conservative; its divisions are the trusted set.
2. **ILP2** at `p ≥ 0.6` — fills gaps, and also adds false divisions.
3. **Remove every division ILP1 did not have.** ILP2's gap-filling is kept, its new forks are not.
4. **ILP3** re-links only.

And a detail worth stealing outright: **weak links at 0.2–0.3 help during the solve and are removed after.**
They give the solver alternative routes so a good edge is not forced to compete against nothing, then they
leave before they can be scored. This is the shape of thing we never tried — a candidate set that differs
*between* the solve and the output.

## 4. ZebraHub pretraining splits by what you pretrain

| team | pretrained | result |
|---|---|---|
| 5th | the **linker**, on 4 ZebraHub datasets | **+0.014 public / +0.016 private** |
| hjyact (#744485) | the **detector** | no gain |

External data is not one lever. The linker is data-starved in a way the detector is not: the detector sees
every nucleus in every frame, while the linker sees only the sparse annotated edges. Our external-data entry
should read "pretraining the **linker** on ZebraHub is worth +0.016 private; pretraining the detector is
worth nothing" — not "external data, never evaluated".

## 5. Drift and duplicate frames — a CV-vs-LB explanation, priced

They found whole-image jumps where **"after shifting back by 14 µm the cells match"**, plus duplicated
frames. In validation they removed those links from *both* prediction and GT.

> The drift fix was a big CV gain and **no LB gain → the hidden test has no large drift**; "one reason why
> many people saw CV and LB that did not match."

And the honest ending: training **without duplicated frames and with drift corrected raised private by
0.011 but dropped public**, so they did not ship it. A second team losing ~0.011 private to public-board
tuning, on top of Korokke3's 0.007 and 89th's 0.007.

## 6. What this changes for us

| our verdict | status after 5th |
|---|---|
| `E61`: division recall unreachable from post-processing; survivor = re-parenting inside the linker | **confirmed and priced: +0.024 private** — the largest single stage in a gold solution |
| division arc is denominator-bound, then also model-bound | **model-bound, and the model belongs in the linker**, co-trained with link scoring |
| external data never evaluated | **split by target**: linker pretraining +0.016 private, detector pretraining nil |
| `jm68` two-pass ILP sized sub-floor | our two-pass was two solves of the **same** objective; 5th's three solves each change the **candidate set and the division rule** between stages |
| smoothing / post-proc closed | a trivial line fit over t−4…t+4 at 80/20 is **+0.005 private**; cf. 14th, where removing linefit smoothing costs **−0.019** |
