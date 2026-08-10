# The joint trainer's first controlled answer, and the split that makes it mean something

The competitive path we had left was our own weights: train the detector and the edge transformer *together*
on real consecutive pairs, the way the published pack was trained, instead of fine-tuning one head with the
backbone frozen. Every earlier retrain froze the U-Net — `lna` on synthetic pairs, `lna` again on real
positives, `zni` on mined hard negatives — and all three degraded. Joint was the one axis never tried, so it
was the one worth building.

It now has an answer, and the answer is no. What follows is why the *no* is trustworthy, which took more work
than the run itself.

## The split had to be fixed first

The trainers were selecting checkpoints on **fold-0**, a k-fold split of the 199 training videos built before
we knew the competition's test set is four specific movies that live *inside* `train/`. Two things were wrong
with it. It is leaked for anything pilkwang-based — they trained on 180 of the 199, covering ~93% of fold-0's
validation — and it *misranks*: the same weights score 0.837 through the fold-0 motion-linker eval and 0.893
through the real pipeline. Selecting on a metric that misranks is worse than not selecting.

Removing it gained data rather than costing any:

|            | before (fold-0) | after |
|------------|-----------------|-------|
| train      | 158             | **191** |
| validation | 41 (leaked)     | **4** (densely annotated) |
| test       | —               | **4** (the competition movies, untouched) |

The four validation movies are the *densely* annotated ones. That choice matters more than it looks: our
annotation of the test four is sparse (52 / 51 / 861 / 1229 nodes) while the hidden evaluation annotates those
same movies densely. Selecting against our sparse labels is therefore selecting against the wrong target — the
documented anti-transfer that cost this campaign three separate times (node-count farming −0.007, smooth 0.3
−0.003, min-track-length). Pushing selection onto densely annotated held-out videos points the pressure at
labels shaped like the ones that actually score us, and leaves the test four as an estimate nothing was fitted
to.

A calibration figure fell out of it. The *same* untrained weights score **0.8474 on the dense validation four**
and **0.9100 on the sparse test four**, against a true leaderboard of 0.892. The denser set is both harsher and
closer to reality; the sparse four flatter a pipeline through the node-count bonus. That is worth remembering
whenever a proxy number looks generous.

## The run, and the control that made it readable

Warm-started from the published pack, 3000 steps, detection and association optimised together, selected
through the shipped `CellTracker` at the shipped threshold. The result:

- initial eval, before any training: **0.8474**
- no window of 3000 steps ever beat it; early stopping fired at step 2500
- save-best therefore kept the **initial** weights

Which produced a confusing surface: the "trained" checkpoint and a freshly mounted pack scored *identically*,
to four decimals, with identical node recall and node-count ratio. Identical scores are a fact about weights,
not about tracking, so the checkpoint was compared tensor-by-tensor against the pack — byte-identical. The
resume snapshot held genuinely different weights (max delta 8.9e-3), so training had moved the model; it had
simply moved it somewhere worse, and the machinery correctly refused to save it.

**Joint warm-start fine-tuning at lr 1e-4 does not improve on the published weights.** That is consistent with
every warm-start fine-tune here: the published weights already fit these frames, so the gradient pulls away
from their optimum rather than toward a better one.

## What this does not settle

The refutation is of one configuration, not of joint training:

- **From scratch** was never run, and it is the case with no warm-start optimum to fight.
- **A much lower learning rate** (1e-5, 1e-6) is the standard response when fine-tuning a converged model
  degrades it, and 1e-4 was inherited from the detector recipe rather than chosen for this.
- **`det_weight`** was never swept; the detection term may simply be dominating and dragging the association
  head with it.

Recording those explicitly matters, because a single refuting run kills a *use*, not an asset — and the
temptation after a negative result is to file the whole direction away.

## The instrument was the real deliverable

The run also exposed that neither logging path worked. Metrics went to mlflow, which had been inert since the
day it was wired — `enable_system_metrics_logging()` sets a flag, then every `start_run()` raised because
`psutil` was absent, and the "tracking must never break a run" guard swallowed it. Console summaries went to a
`__main__` logger that sits outside the `celltrack` tree the log handlers configure, so a module run with
`python -m` logged into the void while every *imported* module kept printing. Both are fixed, and the guard now
says so out loud when it disables itself.

The lesson generalises past this repo: **a component whose contract is "never break the caller" must still be
loud about being broken**, or it converts a hard failure into an invisible one. The number this run was meant
to produce was recoverable only because the checkpoint survived on disk.
