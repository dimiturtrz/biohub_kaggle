"""Does the learned patch encoder separate the confusor cells that raw correlation and the trunk features cannot?

THE ONE QUESTION that licenses spending a submission on xqjk. `appearance_identity` established the bottleneck is
a discriminator gap and that neither the `(1, 4, 4)` trunk features nor a fixed raw-box correlation rank the true
successor above chance on the dense-movie confusor (raw TOP-1 0.161 against chance 0.255). The encoder's whole
reason to exist is that a LEARNED read of the full-resolution voxels, trained only to be discriminative, might see
what a fixed correlation and an invariance-trained trunk both miss. This asks it directly, on the same pairs and
the same choice sets, so the comparison is like-for-like: only the scorer changes.

THE READ. The encoder passes the gate if it ranks the true successor above chance on the MISLINKED confusor pairs
— that is the population the shipped tracker gets wrong, and lifting it is the whole thesis. The CORRECT-edge
population is the control: an encoder that also separates the easy edges (where the rival is merely the nearest
alternative) is measuring identity, while one that separates neither is too weak and one that separates only the
easy edges has learned nearness, not identity. The raw-correlation baseline runs alongside as the floor the
encoder must clear to have added anything a fixed read did not.

This is GT-free of the leaderboard but not of the local truth: it uses the annotated true successor to know which
candidate is right, exactly as the raw diagnosis does. It is a mechanism gate, not a proxy score — it says whether
the embedding carries the missing signal, and only if it does is a trained-encoder edge head worth a slot.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float

from celltrack.analysis.appearance_identity import (
    BoxScorer,
    IdentityDiagnosis,
    IdentitySimilarity,
    InGateRanking,
)
from celltrack.analysis.localisation import CandidatePairs
from celltrack.eval.dense_diagnosis import DENSE_MOVIE
from celltrack.models.patch_encoder import PatchEncoder, PatchEncoderConfig
from celltrack.operating_point import TrackerConfig
from core.paths import DataRoot

logger = logging.getLogger(__name__)

# The raw-correlation TOP-1 the prior `appearance_identity` diagnosis measured on this dense-movie confusor set.
_DOCUMENTED_RAW_TOP1 = 0.161
# Pre-registered BEFORE any number is read (sibling, 2026-08-24): the raw-corr arm rides the SAME `_rank` path
# as the encoder arm, so reproducing its documented value witnesses the shared harness. A raw-corr TOP-1 outside
# this band means the shared path moved (chance 0.255 sits OUTSIDE it) — HALT before the encoder number exists,
# so a moved harness can never masquerade as an encoder result on the n~37 set where a bug lands effect-sized.
_WITNESS_BAND = 0.05


class EncoderGate:
    """The one confusor-separation question that licenses spending a submission on xqjk — scorer, loader, report.

    It reads the encoder as a `BoxScorer` (cosine of L2-normalised embeddings), rebuilds it from a checkpoint, and
    reports its TOP-1 against chance beside the raw-correlation floor, on the mislinked confusor and the correct-edge
    control. It is a mechanism gate, not a proxy: it says whether the embedding carries the missing signal.
    """

    @staticmethod
    def scorer(encoder: PatchEncoder, device: str) -> BoxScorer:
        """A `BoxScorer` that embeds source and candidate boxes and ranks by cosine — the encoder's learned identity.

        The embeddings are L2-normalised HERE, at read time: the encoder emits vectors in R^d so `SIGReg` can shape
        their distribution, and a direction is what a similarity wants, so normalisation belongs at the point cosine
        is taken rather than baked into the encoder.
        """

        @torch.no_grad()
        def score(
            source: Float[np.ndarray, "z y x"], candidates: Float[np.ndarray, "c z y x"]
        ) -> Float[np.ndarray, "c"]:
            boxes = np.concatenate([source[None], candidates], axis=0)[:, None]  # (1+c, 1, z, y, x)
            tensor = torch.from_numpy(boxes.astype(np.float32)).to(device)
            embeddings = encoder(tensor)
            unit = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            similarity = unit[0] @ unit[1:].T
            return similarity.cpu().numpy()

        return score

    @staticmethod
    def load(checkpoint: Path, device: str) -> PatchEncoder:
        """Rebuild the encoder from a checkpoint carrying its config beside the weights."""
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        encoder = PatchEncoder(PatchEncoderConfig(**state["config"]))
        encoder.load_state_dict(state["encoder"])
        return encoder.to(device).eval()

    @staticmethod
    def _rank(
        diagnosis: IdentityDiagnosis, pairs: CandidatePairs, gate_um: float, scorer: BoxScorer, radius_um: float
    ) -> InGateRanking:
        """Rank one population's true successors under a scorer, over the same in-gate choice sets."""

        def read(rows: np.ndarray) -> object:
            return diagnosis._boxes(rows, radius_um)  # noqa: SLF001

        return InGateRanking.of(diagnosis.prediction, diagnosis.spacing, read, pairs, gate_um, scorer)  # type: ignore[arg-type]

    @staticmethod
    def _log_ranking(label: str, name: str, ranking: InGateRanking) -> None:
        """One TOP-1-against-chance line for a (population, scorer) arm."""
        logger.info(
            "%-10s %-9s n=%4d  TOP-1 %.3f (chance %.3f)  median rank %.1f",
            label,
            name,
            ranking.count(),
            ranking.top_one(),
            ranking.chance(),
            float(np.median(ranking.true_rank)) if ranking.count() else float("nan"),
        )

    @staticmethod
    def _witness(diagnosis: IdentityDiagnosis, mislinked: CandidatePairs, gate_um: float, radius_um: float) -> bool:
        """Run the raw-correlation arm FIRST and check it reproduces the documented TOP-1 within the pre-registered
        band. A witness only witnesses if its verdict lands BEFORE the number it licenses: raw-corr and the encoder
        ride the identical `_rank` path, so a raw-corr that reproduces 0.161 proves the shared harness is intact —
        and a raw-corr that misses means the harness moved, so the encoder number must not be read at all."""
        ranking = EncoderGate._rank(diagnosis, mislinked, gate_um, IdentitySimilarity.raw_correlation_scorer, radius_um)
        EncoderGate._log_ranking("mislinked", "raw-corr", ranking)
        intact = abs(ranking.top_one() - _DOCUMENTED_RAW_TOP1) <= _WITNESS_BAND
        logger.info(
            "witness    raw-corr  TOP-1 %.3f vs documented %.3f +/- %.3f  ->  %s",
            ranking.top_one(),
            _DOCUMENTED_RAW_TOP1,
            _WITNESS_BAND,
            "harness intact, reading encoder" if intact else "HARNESS MOVED — encoder arm NOT read",
        )
        return intact

    @staticmethod
    def _report(diagnosis: IdentityDiagnosis, encoder: PatchEncoder, device: str, radius_um: float) -> None:
        """Witness the shared harness on raw-corr FIRST, then read the encoder on both populations only if intact."""
        gate_um = TrackerConfig.shipped().linker.gate_um
        prediction, truth, matching = diagnosis.prediction, diagnosis.truth, diagnosis.matching
        mislinked = CandidatePairs.mislinked(prediction, truth, matching)
        if not EncoderGate._witness(diagnosis, mislinked, gate_um, radius_um):
            return
        learned = EncoderGate.scorer(encoder, device)
        populations = (
            ("mislinked", mislinked),
            ("correct", CandidatePairs.correct(prediction, truth, matching, diagnosis.spacing, gate_um)),
        )
        for label, pairs in populations:
            EncoderGate._log_ranking(label, "encoder", EncoderGate._rank(diagnosis, pairs, gate_um, learned, radius_um))


def main() -> None:
    """Ask whether the trained patch encoder ranks the confusor's true successor above chance."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Learned patch-encoder confusor-separation gate.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--movie", default=DENSE_MOVIE)
    parser.add_argument("--radius", type=float, default=3.25)
    args = parser.parse_args()

    diagnosis = IdentityDiagnosis._of(DataRoot.from_config(args.config), args.device, args.movie)  # noqa: SLF001
    encoder = EncoderGate.load(args.checkpoint, args.device)
    EncoderGate._report(diagnosis, encoder, args.device, args.radius)  # noqa: SLF001


if __name__ == "__main__":
    main()
