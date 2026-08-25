# Directional-PE margin gain is confounded with detected-corpus finetuning until the bias grows

2026-08-25. Long directional-PE arm (`directional_pe_long_v1.pt`): warm pilkwang pack + fresh zero-init
relative-position bias, full finetune, `--det-weight 0`, `--detected-videos 40`, 4 epochs, lr 5e-5,
selection-objective `confusor_margin`.

## The instrument (proxy-independent — the saturated proxy is banned above 0.89)

`confusor_margin = mean_p_true − mean_p_chosen`, read straight off the eval's mislink diagnostics:

| point | P_true | P_chosen | margin | node R | mislinks |
|---|---|---|---|---|---|
| init (warm pack, 0 train) | 0.185 | 0.581 | **−0.396** | 0.980 | 64 |
| step 1245 (1/3 epoch) | 0.116 | 0.289 | **−0.173** | 0.894 | 40 |

Margin narrowed +0.223 in 1/3 epoch. Looks like a win — but it is NOT attributable to directional-PE.

## Why it is not the directional term (yet)

Checkpoint bias params at step 1245 (`transformer_state`):
- `bias.weight` (RADIAL) absmean = 0.00105
- `bias.direction_weight` (DIRECTIONAL) absmean = 0.00039

Both are essentially still at zero-init. The v3 long-train reference had radial 0.0625 / directional 0.0119
(60× larger). So at 1/3 epoch neither PE term is load-bearing — the margin gain is the **detected-corpus
finetuning effect**, the same mechanism memory already logs as dw0: +margin held-out, **−0.028 LB** (proxy
anti-correlated at the top). A margin gain from THIS source does not clear the submission gate.

## What actually tests bottleneck B

The directional axis only becomes testable once `direction_weight` is load-bearing. Two things gate a real
verdict:
1. **Growth** — the bias must grow to a magnitude comparable to radial. At lr 5e-5 from zero it grows slowly
   (~0.0004 at 1/3 epoch → ~0.005 extrapolated at 4 epochs, still < v3's 0.0119). 4 epochs is likely
   insufficient; continue via `--init-weights directional_pe_long_v1.pt` for more epochs (full non-LoRA ckpt,
   loads clean) or raise the rate.
2. **Attribution** — even with a grown bias, a margin gain is confounded with the corpus effect. A MATCHED
   radial-only arm (`--relative-position` WITHOUT `--relative-position-directional`, same corpus/steps)
   isolates the directional delta. Only `margin(directional) − margin(radial-only)` attributes to bottleneck B.

## The regime self-destructs — the endpoint was never readable

Run killed at step 4980. After the step-1245 eval the detector COLLAPSED: node R 0.980→0.894 (step1245)→
**0.211** (step3735), det loss ~9, mislinks 0, `P true nan`. `best` froze at −0.1730 (later evals are nan and
never beat it). This is **BN-pollution** (memory `recall-collapse = BN-pollution`): warm pilkwang pack ships BN,
`--det-weight 0` leaves the detector unsupervised, so its BN running stats drift on the detected-corpus
distribution and the head decays. The single readable eval (step1245) is itself on an already-degrading
detector (0.894), with the bias still at zero-init — so it was never a clean directional-PE test.

## Status

Directional-PE remains **TRIED, not DEMONSTRATED** — not refuted. The warm-pack + det0 + detected-corpus regime
CANNOT test it: BN-pollution kills the detector before the zero-init bias grows. The only regime that both
(a) holds node R (GN, no BN stats to pollute — memory `dw0-synth-velocity-recall-collapse`) and (b) lets the
bias grow is **from-scratch `--norm group`** — a 3-5h train for a knob-sized rider on bottleneck B. Poor
wall-clock÷gain vs bottleneck A (finer122 detection-merge). Parked, not killed. No submission on any margin
from this run.
