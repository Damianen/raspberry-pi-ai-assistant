"""Pure identity logic: cosine matching against enrolled people and majority voting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def best_match(
    embedding: np.ndarray,
    people: Sequence[tuple[str, np.ndarray]],
    threshold: float,
) -> tuple[str, float] | None:
    """Highest-cosine enrolled person at or above the threshold, or None."""
    best: tuple[str, float] | None = None
    for name, enrolled in people:
        score = cosine_similarity(embedding, enrolled)
        if best is None or score > best[1]:
            best = (name, score)
    if best is None or best[1] < threshold:
        return None
    return best


@dataclass(frozen=True)
class Decision:
    """Outcome of one identity vote; name is None when nobody won the majority."""

    name: str | None
    score: float


class IdentityVoter:
    """Collects (name | None, score) samples; a vote completes every `samples_needed`.

    The winner needs a strict majority of all samples (unknowns count against
    it); its score is the mean over the samples that matched it. Samples clear
    after every vote, so each decision is produced exactly once.
    """

    def __init__(self, samples_needed: int) -> None:
        if samples_needed < 1:
            raise ValueError("samples_needed must be >= 1")
        self._needed = samples_needed
        self._samples: list[tuple[str | None, float]] = []

    def add(self, name: str | None, score: float) -> Decision | None:
        self._samples.append((name, score))
        if len(self._samples) < self._needed:
            return None
        samples, self._samples = self._samples, []
        counts = Counter(sample_name for sample_name, _ in samples if sample_name is not None)
        if counts:
            winner, votes = counts.most_common(1)[0]
            if votes > self._needed // 2:
                scores = [s for sample_name, s in samples if sample_name == winner]
                return Decision(winner, sum(scores) / len(scores))
        return Decision(None, 0.0)

    def reset(self) -> None:
        self._samples.clear()


class IdentityTracker:
    """Owns the vote lifecycle: decide once, hold it, re-verify periodically."""

    def __init__(self, samples_needed: int, reverify_interval_s: float) -> None:
        self._voter = IdentityVoter(samples_needed)
        self._reverify_interval_s = reverify_interval_s
        self._decided_at: float | None = None

    def sampling(self, now: float) -> bool:
        """True while samples are wanted — callers can skip embedding work otherwise."""
        return self._decided_at is None or now - self._decided_at >= self._reverify_interval_s

    def observe(self, name: str | None, score: float, now: float) -> Decision | None:
        """Feed one sample; returns a Decision exactly when a vote completes."""
        if not self.sampling(now):
            return None
        self._decided_at = None
        decision = self._voter.add(name, score)
        if decision is not None:
            self._decided_at = now
        return decision

    def reset(self) -> None:
        self._voter.reset()
        self._decided_at = None
