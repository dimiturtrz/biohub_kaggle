# Every training conclusion this campaign made was measured on an unknown scale

Four separate defects surfaced today, all in the same place and all the same kind. None is a bug in the
usual sense — every one of them ran, passed its tests, and produced numbers. They just produced numbers
whose *scale* nobody had checked, and I drew conclusions from those numbers for a week.

## The four

**1. A "run" was a sixth of a first pass.** The joint trainer takes `--steps`. Every joint experiment used
3000 of them. The dataset holds **18024 GT pairs** and the loop consumes one pair per step, so a nominal
epoch is ~18000 steps and every run this campaign performed was **~17% of a single epoch** — against the
published pack's fifty. The learning-rate finding, the cosine-vs-flat comparison, the +0.0075 held-out
gain: all measured inside a partial first pass, then written up as properties of the recipe.

**2. Sampling is with replacement, so coverage never reaches 100%.** `PairDataset.__getitem__` draws
i.i.d.:

```python
rng = np.random.default_rng(self._seed + index)
target = self._targets[int(rng.integers(len(self._targets)))]
```

That is not a shuffled epoch. After `n` draws over `n` items the expected coverage is `1 − e⁻¹ ≈ 63%`, so
even a *full* nominal epoch never sees ~37% of the pairs while drawing others three and four times. "One
epoch" and "the model has seen the data" are not the same statement here, and only one of them is true.

**3. `patience` counts eval windows, and the window size had no flag.** `eval_every` was a config field
with no CLI argument, so every run silently used 500. Patience is therefore denominated in a unit the
caller could not set: `patience 5` means 2500 steps (14% of an epoch) at `eval_every 500`, and five whole
epochs at 9000. Same number, two orders of magnitude apart.

**4. I stopped runs during dips — twice — having already written down not to.** The 3000-step cosine run
dipped to 0.8356 at step 1000 and reached its *best* 0.8646 at 2000. That memory says verbatim: *"a DIP —
do not stop here, two earlier short runs did and I wrongly concluded the optimum is ~500 steps."* Then I
killed the contrastive run at step 2500 of 18000 and recorded it as refuted.

## Why they are one failure, not four

Each is a **units** question that nobody asked: steps per what, coverage of what, patience over what,
converged relative to what. The code was audited for correctness — the losses are decoupled and tested,
the split is honest, the selector is now provably deterministic — but the *scale of the axis those results
live on* was inherited from a default and never examined.

That is a more dangerous class of defect than a crash, because it is invisible in exactly the way that
matters: the curves look like curves. A run that has seen 17% of its data plateaus, dips, and recovers,
and every one of those shapes is legible as convergence if you do not know where you are on the x-axis.

## What it changes

The training results are not *wrong*, they are **unbounded below**: every one of them is a lower bound
measured early. Specifically:

- "joint warm-start at lr 1e-4 does not beat pilkwang" — measured over 17% of an epoch.
- "cosine beats flat" — two partial passes compared against each other.
- "+0.0075 on the test four" — real, held out, and still only what 3000 steps bought.
- "contrastive degrades the model" — **withdrawn**; 2500 steps of an 18000-step run is not a verdict.

None of these needs re-refuting. They need re-*measuring* at a scale where the word "converged" means
something.

## The fixes, and the one that generalises

`--eval-every` is now exposed (the immediate unblock). The config should speak in **epochs** outright —
`steps_per_epoch` is derivable from the pair count the trainer already logs, so nothing needs supplying and
no magic number enters; that is filed. The run line should print both scales (`epoch 3.5/15, step 63000`)
so a curve can never again be read at the wrong resolution.

The sampling hole gets a better answer than "shuffle instead": per-pair **difficulty-weighted sampling with
optimistic initialisation** — every pair starts at maximum difficulty, so unseen pairs are drawn first for
free, and once seen they fall to their measured value so the budget flows to what the model gets wrong.
One array, one rule, no phase switch, no decay constant, and coverage stops being a lottery.

The generalisable lesson is narrower than "check your units". It is: **a default you did not choose is not
a decision you made.** `steps=1500`, `eval_every=500`, `patience=5`, sampling-with-replacement — every one
arrived as a default, and every one silently defined what "a training run" meant here. The gates in this
repo check that code is correct. Nothing checks that a number means what the person reading it thinks it
means, and four separate conclusions this week were shaped by that gap.
