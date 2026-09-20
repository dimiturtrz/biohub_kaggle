# E62 — the frontier's fuse weight is worse than ours, and the moment alignment is inert

**2026-09-20. Closes bd `xoap` and `3gyw`.** Both frontier kernels that run a bidirectional edge fuse
weight the reverse direction far below half — evgendvorkin at 0.30, `evg0942` at 0.15 — while we have
always taken the plain harmonic mean, `2fr/(f+r)`, which is the same fuse at `w=0.5`. A month-old issue
flagged the discrepancy and was never run. It is now run, on the test-4 proxy, seven cells.

| cell | proxy score | vs shipped |
|---|---|---|
| bidirectional **off** | 0.9343 | −0.0032 |
| **w=0.5, align off — SHIPPED** | **0.9375** | — |
| w=0.30, align off | 0.9361 | −0.0014 |
| w=0.15, align off | 0.9343 | −0.0032 |
| w=0.5, align on | 0.9370 | −0.0005 |
| w=0.30, align on | 0.9361 | −0.0014 |
| w=0.15, align on | 0.9343 | −0.0032 |

**The shipped cell is the best cell.** The sweep carries its own control — `w=0.5 / align off` is
arithmetically the fuse we already ship — so this is a like-for-like comparison and not a re-baselining.
Nothing here is a lever: the full spread across seven cells is 0.0032, well under the ~0.01–0.02 noise
floor, and the direction that would have been a "win" is the one we were already in.

## w=0.15 reproduces bidirectional-off exactly, and that is arithmetic, not a bug

The 0.15 cells return 0.9343 with clamped 0.9317 and bonus +0.0026 — every digit identical to the
bidirectional-off cell. That is what the weighted harmonic mean is supposed to do. `w` is the REVERSE
direction's share, and

    fuse = forward * reverse / ((1 - w) * reverse + w * forward)   →   forward   as w → 0

so the donors' weight is a small correction on a nearly forward-only score. At that lean the pair ranking
moves too little to change any ILP decision, and the tracker's output is bit-identical to never having run
the reverse pass at all. **The donor constant does not say "weight the reverse lightly"; on our candidate
distribution it says "do not run the reverse pass".** Same shape as the c3 gate bracket that lost recall
4→2 when transplanted: a constant fitted to another pipeline's score distribution carries that
distribution's meaning, not ours.

## The moment alignment is inert — which refutes the plumbing suspicion, not just the knob

`align_reverse_moments` puts the reverse logits on the forward logits' centre and spread before the
softmax, exactly as `evg0942` does, because the two directions normalise over different axes and their raw
scales need not agree. The suspicion it was built to test was a real one: memory records that *every*
bidirectional variant has read flat, and a fuse that shifts the whole probability mass under a fixed
candidate threshold would produce precisely that. It does not. Alignment is identical to four decimals at
w=0.15 and w=0.30 and costs 0.0005 at w=0.5. **The flat bidirectional arms were not flat because of a
scale mismatch** — that explanation is now closed, and the remaining explanation is that the reverse pass
carries little the forward pass does not already have.

## What this leaves

The fuse axis is priced and it is not where the gap is. Bidirectional-on beats off by 0.0032 — real in
sign, sub-floor in size, and already banked. The far side of the weight (w > 0.5, where the reverse pass
would outweigh the forward one) is the only unmeasured part of this axis and is being probed for
completeness, not as a candidate; w=1.0 doubles as a control on whether the reverse direction is as
informative as the forward one.

**Carry forward:** a donor constant is a statement about the donor's distribution. Before porting one, ask
what it reduces to at our numbers — here, arithmetic alone showed 0.15 means "off", which the sweep then
confirmed to every digit.
