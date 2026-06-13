"""Identity logic: cosine matching, majority voting, and re-verify timing."""

from __future__ import annotations

import numpy as np
import pytest

from assistant.perception.identity import (
    Decision,
    IdentityTracker,
    IdentityVoter,
    best_match,
    cosine_similarity,
)

SAMPLES = 7
REVERIFY_S = 90.0


def unit(*values: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


# --- cosine_similarity -------------------------------------------------------


def test_cosine_identical_is_one() -> None:
    v = unit(0.3, -0.7, 0.2)
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero() -> None:
    assert cosine_similarity(unit(1, 0), unit(0, 1)) == pytest.approx(0.0)


def test_cosine_opposite_is_minus_one() -> None:
    v = unit(0.5, 0.5)
    assert cosine_similarity(v, -v) == pytest.approx(-1.0)


def test_cosine_is_scale_invariant() -> None:
    v = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(v, 5.0 * v) == pytest.approx(1.0)


def test_cosine_zero_vector_is_zero() -> None:
    assert cosine_similarity(np.zeros(3, dtype=np.float32), unit(1, 1, 1)) == 0.0


# --- best_match --------------------------------------------------------------


def people_fixture() -> list[tuple[str, np.ndarray]]:
    return [
        ("alice", unit(1, 0, 0)),
        ("bob", unit(0, 1, 0)),
    ]


def test_best_match_picks_highest_scoring_person() -> None:
    query = unit(0.9, 0.1, 0)  # closest to alice
    match = best_match(query, people_fixture(), threshold=0.363)
    assert match is not None
    name, score = match
    assert name == "alice"
    assert score == pytest.approx(cosine_similarity(query, unit(1, 0, 0)))


def test_best_match_below_threshold_is_none() -> None:
    assert best_match(unit(0, 0, 1), people_fixture(), threshold=0.363) is None


def test_best_match_empty_people_is_none() -> None:
    assert best_match(unit(1, 0, 0), [], threshold=0.363) is None


def test_best_match_at_threshold_matches() -> None:
    query = unit(1, 0, 0)
    match = best_match(query, [("alice", query)], threshold=1.0)
    assert match == ("alice", pytest.approx(1.0))


# --- IdentityVoter -----------------------------------------------------------


def test_voter_no_decision_before_full_window() -> None:
    voter = IdentityVoter(SAMPLES)
    for _ in range(SAMPLES - 1):
        assert voter.add("alice", 0.5) is None


def test_voter_clear_majority_wins_with_mean_score() -> None:
    voter = IdentityVoter(SAMPLES)
    decisions = [voter.add("alice", s) for s in (0.4, 0.5, 0.6, 0.5)]
    decisions += [voter.add("bob", 0.9) for _ in range(3)]
    assert decisions[:-1] == [None] * (SAMPLES - 1)
    decision = decisions[-1]
    assert decision == Decision("alice", pytest.approx(0.5))


def test_voter_split_without_majority_is_unknown() -> None:
    voter = IdentityVoter(SAMPLES)
    for name in ["alice", "alice", "alice", "bob", "bob", "bob"]:
        assert voter.add(name, 0.5) is None
    assert voter.add("carol", 0.5) == Decision(None, 0.0)


def test_voter_none_majority_is_unknown() -> None:
    voter = IdentityVoter(SAMPLES)
    for _ in range(4):
        assert voter.add(None, 0.0) is None
    for _ in range(2):
        assert voter.add("alice", 0.9) is None
    assert voter.add("alice", 0.9) == Decision(None, 0.0)


def test_voter_four_of_seven_wins() -> None:
    voter = IdentityVoter(SAMPLES)
    names = ["alice", None, "alice", None, "alice", None, "alice"]
    decision = [voter.add(n, 0.5 if n else 0.0) for n in names][-1]
    assert decision == Decision("alice", pytest.approx(0.5))


def test_voter_clears_after_each_vote() -> None:
    voter = IdentityVoter(SAMPLES)
    for _ in range(SAMPLES):
        first = voter.add("alice", 0.5)
    assert first == Decision("alice", pytest.approx(0.5))
    for _ in range(SAMPLES - 1):
        assert voter.add("bob", 0.7) is None
    assert voter.add("bob", 0.7) == Decision("bob", pytest.approx(0.7))


def test_voter_reset_discards_partial_samples() -> None:
    voter = IdentityVoter(SAMPLES)
    for _ in range(SAMPLES - 1):
        voter.add("alice", 0.5)
    voter.reset()
    for _ in range(SAMPLES - 1):
        assert voter.add("bob", 0.7) is None
    assert voter.add("bob", 0.7) == Decision("bob", pytest.approx(0.7))


# --- IdentityTracker ---------------------------------------------------------


def make_tracker() -> IdentityTracker:
    return IdentityTracker(SAMPLES, REVERIFY_S)


def feed(tracker: IdentityTracker, name: str, count: int, now: float) -> list[Decision | None]:
    return [tracker.observe(name, 0.5, now) for _ in range(count)]


def test_tracker_decides_on_final_sample() -> None:
    tracker = make_tracker()
    results = feed(tracker, "alice", SAMPLES, now=0.0)
    assert results[:-1] == [None] * (SAMPLES - 1)
    assert results[-1] == Decision("alice", pytest.approx(0.5))


def test_tracker_holds_decision_until_reverify() -> None:
    tracker = make_tracker()
    feed(tracker, "alice", SAMPLES, now=0.0)
    assert not tracker.sampling(REVERIFY_S - 1.0)
    assert feed(tracker, "bob", SAMPLES, now=REVERIFY_S - 1.0) == [None] * SAMPLES


def test_tracker_revotes_after_reverify_interval() -> None:
    tracker = make_tracker()
    feed(tracker, "alice", SAMPLES, now=0.0)
    assert tracker.sampling(REVERIFY_S)
    results = feed(tracker, "bob", SAMPLES, now=REVERIFY_S)
    assert results[-1] == Decision("bob", pytest.approx(0.5))
    # The fresh decision is held again for a full interval.
    assert not tracker.sampling(REVERIFY_S + 1.0)


def test_tracker_reset_mid_vote_gives_one_fresh_decision() -> None:
    tracker = make_tracker()
    feed(tracker, "alice", SAMPLES - 1, now=0.0)
    tracker.reset()
    results = feed(tracker, "bob", SAMPLES, now=1.0)
    decisions = [r for r in results if r is not None]
    assert decisions == [Decision("bob", pytest.approx(0.5))]


def test_tracker_reset_clears_held_decision() -> None:
    tracker = make_tracker()
    feed(tracker, "alice", SAMPLES, now=0.0)
    tracker.reset()
    assert tracker.sampling(1.0)
    results = feed(tracker, "alice", SAMPLES, now=1.0)
    assert results[-1] == Decision("alice", pytest.approx(0.5))
