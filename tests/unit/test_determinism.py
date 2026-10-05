"""Hashing stability and seeded-stream independence."""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pytest

from flow_model.core.determinism import (
    canonical_json,
    derive_seed,
    hash_dataset,
    hash_file,
    rng,
    seed_global,
    spawn_streams,
    stable_hash,
)


def test_hash_is_insensitive_to_key_order():
    a = {"b": 1, "a": 2, "c": {"y": 1, "x": 2}}
    b = {"c": {"x": 2, "y": 1}, "a": 2, "b": 1}
    assert stable_hash(a) == stable_hash(b)


def test_hash_is_sensitive_to_values():
    assert stable_hash({"x": 1.0}) != stable_hash({"x": 1.0000000001})
    assert stable_hash({"x": 1}) != stable_hash({"y": 1})


def test_negative_zero_hashes_as_zero():
    assert stable_hash({"x": -0.0}) == stable_hash({"x": 0.0})


def test_non_finite_floats_hash_stably():
    assert stable_hash(float("nan")) == stable_hash(float("nan"))
    assert stable_hash(float("inf")) != stable_hash(float("-inf"))


def test_enums_dates_and_numpy_are_hashable():
    from flow_model.core.enums import Regime

    payload = {
        "regime": Regime.CHOP,
        "d": date(2020, 1, 1),
        "t": datetime(2020, 1, 1, tzinfo=timezone.utc),
        "arr": np.array([1.0, 2.0]),
        "n": np.float64(3.5),
    }
    assert stable_hash(payload) == stable_hash(payload)
    assert isinstance(canonical_json(payload), str)


def test_sets_hash_order_independently():
    assert stable_hash({1, 2, 3}) == stable_hash({3, 1, 2})


def test_list_order_matters():
    assert stable_hash([1, 2]) != stable_hash([2, 1])


def test_derive_seed_is_reproducible_and_stream_independent():
    assert derive_seed(42, "monte_carlo") == derive_seed(42, "monte_carlo")
    assert derive_seed(42, "monte_carlo") != derive_seed(42, "execution")
    assert derive_seed(42, "a", "b") != derive_seed(42, "a", "c")
    assert derive_seed(1, "x") != derive_seed(2, "x")


def test_named_rngs_are_reproducible():
    assert np.array_equal(rng(42, "mc").random(10), rng(42, "mc").random(10))


def test_named_rngs_are_independent():
    """Adding a new stochastic component must not shift an existing one."""
    a = rng(42, "monte_carlo").random(20)
    b = rng(42, "execution_jitter").random(20)
    assert not np.any(a == b)


def test_spawn_streams():
    streams = spawn_streams(7, ["a", "b"])
    assert set(streams) == {"a", "b"}
    assert not np.array_equal(streams["a"].random(5), streams["b"].random(5))


def test_hash_file(tmp_path):
    p = tmp_path / "bars.csv"
    p.write_text("ts,open,high,low,close\n1,1,2,0,1\n")
    first = hash_file(p)
    assert first == hash_file(p)
    p.write_text("ts,open,high,low,close\n1,1,2,0,2\n")
    assert hash_file(p) != first


def test_hash_dataset_distinguishes_slices():
    base = dict(symbol="NQ", start=date(2015, 1, 1), end=date(2019, 1, 1),
                rows=1000, source="csv")
    assert hash_dataset(**base) == hash_dataset(**base)
    assert hash_dataset(**{**base, "end": date(2020, 1, 1)}) != hash_dataset(**base)
    assert hash_dataset(**{**base, "rows": 1001}) != hash_dataset(**base)


def test_seed_global_is_reproducible():
    import random

    seed_global(99)
    first = (random.random(), float(np.random.random()))
    seed_global(99)
    assert (random.random(), float(np.random.random())) == pytest.approx(first)
